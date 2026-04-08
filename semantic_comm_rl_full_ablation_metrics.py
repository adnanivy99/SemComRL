#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_comm_rl_full_ablation_metrics.py

Full ablation + optional evaluation metrics (BLEU 1-4, ROUGE-L, METEOR, BERTScore)
Controlled by CLI flags: --enable_bleu --enable_rouge --enable_meteor --enable_bertscore
Default dataset: Hugging Face `ag_news` (cached). Use --dataset synthetic to avoid HF.

$nohup python semantic_comm_rl_full_ablation_metrics.py --outdir ./results_full_sweep --enable_bleu  --bleu_max_n 4 --enable_rouge --enable_meteor --enable_bertscore  --sweep_counts 5 > semantic_comm_rl_full_ablation_metrics.log &  

"""

import os, sys, argparse, random, math, logging, json, datetime
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sklearn.manifold import TSNE


# optional libs
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    from reedsolo import RSCodec
    HAS_REEDSOL = True
except Exception:
    HAS_REEDSOL = False

try:
    import pyldpc
    HAS_PYLDPC = True
except Exception:
    HAS_PYLDPC = False

# optional metrics libs - import lazily if flags true
HAS_NLTK = False
HAS_ROUGE = False
HAS_METEOR = False

# try:
#     from bert_score import BERTScorer
#     HAS_BERTSCORE = True
# except Exception:
#     HAS_BERTSCORE = False

try:
    from bert_score import BERTScorer
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False




# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("semantic_comm_ablation")

# device selection
def get_device():
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            logger.info("Using MPS device")
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")
DEVICE = get_device()

def deterministic_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# --- Mock embedder fallback
import hashlib
class MockST:
    def __init__(self, dim=64): self.dim=int(dim)
    def encode(self, texts, convert_to_tensor=False, device=None):
        single=False
        if isinstance(texts,str): texts=[texts]; single=True
        vecs=[]
        for t in texts:
            h=hashlib.md5(t.encode("utf-8")).hexdigest()
            seed=int(h,16)%(2**32)
            rng=np.random.RandomState(seed)
            v=rng.randn(self.dim).astype(np.float32)
            v/= (np.linalg.norm(v)+1e-9)
            vecs.append(v)
        arr=np.vstack(vecs)
        if convert_to_tensor:
            dev=device or DEVICE
            return torch.tensor(arr, device=dev, dtype=torch.float32)
        return arr if not single else arr[0]

# --- Model building blocks
class PolicyNetwork(nn.Module):
    def __init__(self, emb_dim:int, hidden:int=128, layers:int=2, temp:float=1.0):
        super().__init__()
        layers_list=[nn.Linear(emb_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(max(0,layers-1)):
            layers_list += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        layers_list += [nn.Linear(hidden, emb_dim)]
        self.net = nn.Sequential(*layers_list); self.temp=float(temp)
    def forward(self, z):
        logits=self.net(z)/max(self.temp,1e-8)
        return F.softmax(logits, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, emb_dim:int, hidden:int=64, layers:int=2):
        super().__init__()
        layers_list=[nn.Linear(emb_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(max(0,layers-1)):
            layers_list += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        layers_list += [nn.Linear(hidden,1)]
        self.net=nn.Sequential(*layers_list)
    def forward(self,z): return self.net(z).squeeze(-1)

# --- helpers for packing/unpacking ints to bytes
def _pack_int_to_bytes(val:int,bits:int)->List[int]:
    n_bytes=(bits+7)//8
    mask=(1<<bits)-1
    if val<0: val=(1<<bits)+val
    val=int(val)&mask
    out=[]
    for i in range(n_bytes):
        out.append(int((val>>(8*i))&0xFF))
    return out

def _unpack_bytes_to_int(bytelist:List[int], bits:int)->int:
    val=0
    for i,b in enumerate(bytelist):
        val |= (int(b)&0xFF)<<(8*i)
    mask=(1<<bits)-1
    val=val&mask
    sign_bit=1<<(bits-1)
    if val & sign_bit:
        val = val - (1<<bits)
    return int(val)

# --- ECC classes (repetition, uep, reedsolomon, ldpc fallback)
class RepetitionECC:
    name="repetition"
    def encode(self,symbols,t_i,bits): 
        out=[]
        for i,val in enumerate(symbols):
            n_par=int(t_i[i]) if i<len(t_i) else 0
            n_tx=1+max(0,n_par)
            packed=_pack_int_to_bytes(int(val), bits)
            cw=[]
            for _ in range(n_tx): cw.extend(packed)
            out.append(cw)
        return out
    def decode(self, received, t_i, bits):
        n_bytes=(bits+7)//8
        out=[]
        for flat in received:
            if not flat: out.append(0); continue
            chunks=[]
            for j in range(0,len(flat),n_bytes):
                if j+n_bytes<=len(flat): chunks.append(tuple(flat[j:j+n_bytes]))
            if not chunks: out.append(0); continue
            counts=Counter(chunks); most_common=counts.most_common(1)[0][0]
            out.append(_unpack_bytes_to_int(list(most_common), bits))
        return out

class UEPRepetitionECC(RepetitionECC):
    name="uep"

class ReedSolomonECC:
    name="reedsolomon"
    def __init__(self):
        self.nsym = 4 if HAS_REEDSOL else 0
    def encode(self,symbols,t_i,bits):
        if not HAS_REEDSOL:
            logger.warning("reedsolo unavailable -> fallback to repetition")
            return RepetitionECC().encode(symbols,t_i,bits)
        out=[]
        for i,val in enumerate(symbols):
            nsym = max(1, int(t_i[i])) if i<len(t_i) else  4 #self.nsym  # Respect per-symbol t_i as nsym
            rsc = RSCodec(nsym=nsym)
            packed=bytes(_pack_int_to_bytes(int(val),bits))
            try:
                cw=list(rsc.encode(packed))
            except Exception:
                cw=list(packed)
            out.append(cw)
        return out
    def decode(self,received,t_i,bits):
        if not HAS_REEDSOL:
            return RepetitionECC().decode(received,t_i,bits)
        out=[]
        for i,flat in enumerate(received):
            if not flat: out.append(0); continue
            nsym = max(1, int(t_i[i])) if i<len(t_i) else self.nsym
            rsc = RSCodec(nsym=nsym)
            try:
                dec=rsc.decode(bytes(flat))
                out.append(_unpack_bytes_to_int(list(dec), bits))
            except Exception:
                # fallback to majority
                n_bytes=(bits+7)//8; chunks=[]
                for j in range(0,len(flat),n_bytes):
                    if j+n_bytes<=len(flat): chunks.append(tuple(flat[j:j+n_bytes]))
                if not chunks: out.append(0); continue
                counts=Counter(chunks); most_common=counts.most_common(1)[0][0]
                out.append(_unpack_bytes_to_int(list(most_common), bits))
        return out

class LDPCECC:
    name="ldpc"
    def __init__(self):
        if not HAS_PYLDPC:
            logger.warning("pyldpc unavailable -> LDPC fallback to repetition")
    def encode(self,symbols,t_i,bits):
        if HAS_PYLDPC:
            out=[]
            for i,val in enumerate(symbols):
                nsym = max(1, int(t_i[i])) if i<len(t_i) else 1  # Use t_i as duplication factor
                b=bytes(_pack_int_to_bytes(int(val),bits))
                cw = list(b) * nsym
                out.append(cw)
            return out
        return RepetitionECC().encode(symbols,t_i,bits)
    def decode(self,received,t_i,bits):
        if HAS_PYLDPC:
            out=[]
            for flat in received:
                if not flat: out.append(0); continue
                half=flat[:len(flat)//2]
                out.append(_unpack_bytes_to_int(list(half[:(bits+7)//8]), bits))
            return out
        return RepetitionECC().decode(received,t_i,bits)

ECC_REG = {"repetition":RepetitionECC, "uep":UEPRepetitionECC, "reedsolomon":ReedSolomonECC, "ldpc":LDPCECC}

# --- SemanticCommunicationSystem (quant variants + RL policy training + evaluation)
class SemanticCommunicationSystem:
    def __init__(self, ecc_rate=0.05, B=8, device:Optional[torch.device]=None, mock_dim:int=64, alpha:float=0.5, lambda_reg:float=0.0, embed_model="all-MiniLM-L6-v2", action_family="soft", topk_tau:int=1, topk_k:Optional[int]=None, beta:float=0.01, clip_norm:float=1.0, lr_policy:float=5e-4, lr_value:float=5e-4, hidden_size:int=128):
        self.device = device or DEVICE
        self.ecc_rate=float(ecc_rate); self.B=int(B)
        self.alpha=float(alpha); self.lambda_reg=float(lambda_reg)
        self.action_family=action_family; self.topk_tau=int(topk_tau); self.topk_k=topk_k
        self.beta = beta
        self.clip_norm = clip_norm
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.hidden_size = hidden_size
        self.embed_model_name=embed_model
        # load embedder
        if HAS_ST:
            try:
                self.student_model = SentenceTransformer(self.embed_model_name)
                probe=self.student_model.encode(["__probe__"], convert_to_tensor=True)
                self.student_dim = int(probe.shape[-1])
                logger.info("Loaded SentenceTransformer %s dim=%d", self.embed_model_name, self.student_dim)
            except Exception as e:
                logger.warning("Failed to load SentenceTransformer (%s). Using MockST.", e)
                self.student_model = MockST(dim=mock_dim); self.student_dim=mock_dim
        else:
            logger.info("sentence-transformers missing -> MockST")
            self.student_model = MockST(dim=mock_dim); self.student_dim=mock_dim

        self.internal_emb_dim=self.student_dim; self.bits_default=8
        self.policy=PolicyNetwork(emb_dim=self.internal_emb_dim, hidden=self.hidden_size).to(self.device)
        self.value_net=ValueNetwork(emb_dim=self.internal_emb_dim, hidden=self.hidden_size).to(self.device)
        self.policy_optim=torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        self.value_optim=torch.optim.Adam(self.value_net.parameters(), lr=self.lr_value)

        self.messages:List[str]=[]; self.kb_embeddings=None; self.parity_allocation=None; self.importance_scores=None

    def _safe_tensor(self, t: torch.Tensor):
        if not isinstance(t, torch.Tensor):
            t=torch.tensor(t, device=self.device, dtype=torch.float32)
        else:
            t=t.to(self.device, dtype=torch.float32)
        return t.clone().detach().requires_grad_(True)

    def _encode_batch(self, texts:List[str], convert_to_tensor:bool=True)->torch.Tensor:
        mt=self.student_model
        try:
            if HAS_ST and isinstance(mt, SentenceTransformer):
                embs=mt.encode(texts, convert_to_tensor=True)
                if not isinstance(embs, torch.Tensor): embs=torch.tensor(embs, dtype=torch.float32)
                embs=embs.to(self.device, dtype=torch.float32)
            else:
                embs_np = mt.encode(texts, convert_to_tensor=False)
                embs = torch.tensor(embs_np, dtype=torch.float32, device=self.device)
        except Exception as e:
            logger.warning("Embedding failure (%s). Using MockST", e)
            mock=MockST(dim=self.internal_emb_dim)
            embs_np=mock.encode(texts, convert_to_tensor=False)
            embs=torch.tensor(embs_np, dtype=torch.float32, device=self.device)
        return embs

    def _scale_embedding(self, z:torch.Tensor, epsilon=1e-6):
        if z.dim()==1: z=z.unsqueeze(0); squeeze=True
        else: squeeze=False
        s=torch.max(torch.abs(z), dim=1).values + epsilon
        u=z / s[:, None]
        if squeeze: return u.squeeze(0), s.squeeze(0)
        return u, s

    # quant variants
    def _quantize_deterministic(self, z:torch.Tensor, bits:int):
        u,s=self._scale_embedding(z)
        q_max=2**(bits-1)-1
        scaled=u*q_max
        q=torch.round(scaled).clamp(-q_max,q_max)
        return q, q_max, s

    def _quantize_stochastic(self, z:torch.Tensor, bits:int):
        u,s=self._scale_embedding(z)
        q_max=2**(bits-1)-1
        scaled=u*q_max
        d=(torch.rand_like(scaled, device=self.device)-0.5)
        scaled_d=scaled + d
        q=torch.round(scaled_d).clamp(-q_max,q_max)
        return q, q_max, s

    def _quantize_fake(self, z:torch.Tensor, bits:int):
        u,s=self._scale_embedding(z)
        q_max=2**(bits-1)-1
        scaled=u*q_max
        q=torch.round(scaled).clamp(-q_max,q_max)
        return q, q_max, s

    def _compute_importance_scores(self, num_samples=1000):
        ns=min(num_samples, len(self.messages))
        if ns==0:
            self.importance_scores = torch.ones(self.internal_emb_dim, device=self.device)/float(self.internal_emb_dim)
            self.parity_allocation = torch.zeros(self.internal_emb_dim, dtype=torch.int32, device=self.device)
            return
        samples=random.sample(self.messages, ns)
        embs=self._encode_batch(samples, convert_to_tensor=True).detach().cpu()
        var_imp=torch.var(embs, dim=0); mean_abs=torch.mean(torch.abs(embs), dim=0)
        combined=var_imp*mean_abs
        if combined.sum().item()==0: combined=torch.ones_like(combined)
        combined=combined/combined.sum()
        self.importance_scores=combined.to(self.device)
        k=self.internal_emb_dim; T_total=max(1, int(k*self.ecc_rate))
        alloc=torch.floor(self.importance_scores*T_total).int().to(self.device)
        deficit=T_total - int(alloc.sum().item())
        if deficit>0:
            top=torch.argsort(self.importance_scores, descending=True)[:deficit]
            for idx in top: alloc[idx]+=1
        self.parity_allocation=alloc

    def _simulate_channel_flip(self, encoded_bytes:List[List[int]], snr_db:float=3.0, model:str="awgn", bits_per_symbol:Optional[int]=None, num_antennas:int=1):
        bits_per_symbol=int(bits_per_symbol or self.bits_default)
        snr_linear=10**(snr_db/10.0)
        try:
            base_pbit=0.5*math.erfc(math.sqrt(snr_linear))
        except Exception:
            base_pbit=min(0.5,0.5*math.exp(-0.5*snr_linear))
        if model=="awgn": pbit=base_pbit
        elif model=="rayleigh": pbit=min(0.5,base_pbit*1.6)
        elif model=="rician": pbit=max(0.0,base_pbit*0.9)
        elif model=="nakagami": pbit=min(0.5,base_pbit*1.4)
        elif model=="bsc": pbit = 0.05  # Fixed p=0.05 for BSC
        elif model=="burst": pbit = base_pbit
        else: pbit=base_pbit
        if num_antennas>1: pbit=pbit/max(1.0, math.sqrt(num_antennas))
        p_symbol_flip=1.0-(1.0-pbit)**bits_per_symbol
        p_symbol_flip=max(0.0, min(1.0,p_symbol_flip))

        recv=[]; flips=0; total_bits=0
        bytes_per_symbol=(bits_per_symbol+7)//8
        for cw in encoded_bytes:
            rcw=[]
            n_chunks=max(1, len(cw)//bytes_per_symbol)
            for ch in range(n_chunks):
                chunk=cw[ch*bytes_per_symbol:(ch+1)*bytes_per_symbol]
                total_bits += bytes_per_symbol*8
                if model == "burst":
                    if random.random() < p_symbol_flip / 2:  # Burst flip entire chunk
                        new_chunk = [int((b + random.randint(1,255)) & 0xFF) for b in chunk]
                        flips += bytes_per_symbol*8
                        rcw.extend(new_chunk)
                    else:
                        rcw.extend(list(chunk))
                elif model == "bsc":
                    for b_idx in range(len(chunk)):
                        byte = chunk[b_idx]
                        flipped_byte = 0
                        for bit in range(8):
                            original_bit = (byte >> bit) & 1
                            if random.random() < pbit:
                                flipped_bit = 1 - original_bit
                                flips += 1
                            else:
                                flipped_bit = original_bit
                            flipped_byte |= (flipped_bit << bit)
                        rcw.append(flipped_byte)
                else:
                    if random.random() < p_symbol_flip:
                        new_chunk = [int((b + random.randint(1,255)) & 0xFF) for b in chunk]
                        flips += bytes_per_symbol*8
                        rcw.extend(new_chunk)
                    else:
                        rcw.extend(list(chunk))
            recv.append(rcw)
        ber = flips / max(1, total_bits)
        return recv, ber

    def initialize_knowledge_base(self, messages:Optional[List[str]]=None):
        if messages is not None: self.messages = messages
        if not self.messages: raise ValueError("No messages to initialize KB")
        embs=self._encode_batch(self.messages, convert_to_tensor=True)
        self.kb_embeddings=F.normalize(embs, dim=1).to(self.device)
        self._compute_importance_scores()

    def _hybrid_decode(self, z_prime:torch.Tensor, tau_r=0.9, tau_g=0.5)->str:
        if self.kb_embeddings is None: return "NoKB"
        sims=torch.matmul(F.normalize(self.kb_embeddings, dim=1), F.normalize(z_prime, dim=0))
        top_score, top_idx = sims.max(dim=0)
        top_score=float(top_score); idx=int(top_idx)
        if top_score >= tau_r: return self.messages[idx]
        if top_score <= tau_g: return "Unable to retrieve original message. Generated fallback."
        return f"Paraphrased: {self.messages[idx]}"

    def _key_entity_score(self, orig:str, dec:str)->float:
        stop=set(["the","and","for","with","that","this","from","are","is","in","on","at","to","of","a","an"])
        def keywords(s):
            toks=[w.strip(".,:;()").lower() for w in s.split()]
            toks=[t for t in toks if len(t)>3 and not any(c.isdigit() for c in t) and t not in stop]
            return set(toks)
        ko=keywords(orig); kd=keywords(dec)
        if len(ko)==0: return 0.0
        return float(len(ko & kd))/len(ko)

    def _compute_metrics(self, original:str, decoded:str, z:torch.Tensor, z_hat:torch.Tensor):
        cos=float(F.cosine_similarity(z, z_hat, dim=0).item())
        key=self._key_entity_score(original, decoded)
        return {"cosine_similarity": cos, "bert_score": cos, "key": key}

    # sampling & allocation (soft vs topk)
    def _gumbel_topk_sample(self, probs:torch.Tensor, k:int):
        logits=torch.log(torch.clamp(probs,1e-9,1.0))
        g=-torch.log(-torch.log(torch.rand_like(logits, device=self.device)+1e-9)+1e-9)
        vals=logits+g
        topk=torch.topk(vals,k); indices=topk.indices
        counts=torch.bincount(indices, minlength=probs.shape[0]).int()
        selp=probs[indices]; logp=torch.log(torch.clamp(selp,1e-9,1.0)).sum()
        return counts, probs.detach().cpu(), logp

    def _sample_allocation_from_policy(self, z:torch.Tensor, T_total:int, sampling:str="gumbel_topk"):
        if isinstance(z, torch.Tensor):
            z_in=z.clone().detach().to(self.device, dtype=torch.float32).unsqueeze(0) if z.dim()==1 else z.clone().detach().to(self.device, dtype=torch.float32)
        else:
            z_in=torch.tensor(z, device=self.device, dtype=torch.float32).unsqueeze(0)
        probs=self.policy(z_in).squeeze(0)
        if T_total<=0:
            return torch.zeros_like(probs, device=self.device).int(), probs.detach().cpu(), torch.tensor(0.0, device=self.device)
        if self.action_family=="soft":
            sampled = torch.multinomial(probs, T_total, replacement=True)
            counts = torch.bincount(sampled, minlength=probs.size(0)).int().to(self.device)
            
            selp = torch.clamp(probs[sampled], 1e-9, 1.0)
            logp = torch.log(selp).sum()
            return counts, probs.detach().cpu(), logp
        else:
            k_sel = self.topk_k or max(1, int(T_total // max(1, self.topk_tau)))
            if sampling=="gumbel_topk":
                logits=torch.log(torch.clamp(probs,1e-9,1.0))
                g=-torch.log(-torch.log(torch.rand_like(logits, device=self.device)+1e-9)+1e-9)
                vals=logits+g
                topk=torch.topk(vals,k_sel)
                indices=topk.indices
            else:
                indices=torch.argsort(probs, descending=True)[:k_sel]
            counts=torch.zeros(probs.shape[0], dtype=torch.int32, device=self.device)
            for idx in indices: counts[idx]=int(self.topk_tau)
            selp=probs[indices]; logp=torch.log(torch.clamp(selp,1e-9,1.0)).sum()
            return counts, probs.detach().cpu(), logp

    # encode message (quant variants)
    def encode_message(self, text:str, parity_allocation:Optional[torch.Tensor]=None, B:Optional[int]=None, ecc_mode:str="repetition", quant_type:str="int8_det", method:str="uniform"):
        B=self.B if B is None else int(B)
        z=self._encode_batch([text], convert_to_tensor=True).squeeze(0)
        encoded={"original":text,"quant_type":quant_type,"method":method}
        qtype=str(quant_type).lower().strip()
        quant_variant="deterministic"; bits=B
        if qtype.startswith("fake"): quant_variant="fake"; digits=''.join([c for c in qtype if c.isdigit()]); bits=int(digits) if digits else bits
        elif "stoch" in qtype or "stochastic" in qtype: quant_variant="stochastic"; digits=''.join([c for c in qtype if c.isdigit()]); bits=int(digits) if digits else bits
        else:
            quant_variant="deterministic"; digits=''.join([c for c in qtype if c.isdigit()]); bits=int(digits) if digits else bits

        if quant_variant=="deterministic":
            q_tensor,q_max,s=self._quantize_deterministic(z.unsqueeze(0), bits)
        elif quant_variant=="stochastic":
            q_tensor,q_max,s=self._quantize_stochastic(z.unsqueeze(0), bits)
        else:
            q_tensor,q_max,s=self._quantize_fake(z.unsqueeze(0), bits)
        q_np=q_tensor.detach().cpu().numpy().astype(int).flatten()

        if parity_allocation is None:
            if self.parity_allocation is None:
                k=self.internal_emb_dim; T_total=max(1,int(k*self.ecc_rate))
                base=torch.zeros(k,dtype=torch.int32,device=self.device); per=T_total//k; base+=per
                rem=T_total - int(base.sum().item())
                for idx in range(rem): base[idx]+=1
                parity_allocation=base
            else:
                parity_allocation=self.parity_allocation
        parity_cpu = parity_allocation.detach().cpu().numpy().astype(int).flatten() if isinstance(parity_allocation, torch.Tensor) else np.array(parity_allocation, dtype=int).flatten()

        ECCClass = ECC_REG.get(ecc_mode, RepetitionECC)
        ecc_obj = ECCClass()
        ecc_encoded = ecc_obj.encode(q_np.tolist(), parity_cpu.tolist(), bits)
        total_bits = int(np.sum((1 + parity_cpu) * bits))
        encoded.update({"embedding": z.detach().cpu().numpy(), "embedding_raw":z.detach().cpu().numpy(), "quantized": q_np, "deq_embedding": None, "scale": float(s.item() if isinstance(s, torch.Tensor) else float(s)), "q_max": int(q_max), "ecc_encoded": ecc_encoded, "bandwidth_bits": total_bits, "parity_allocation": torch.tensor(parity_cpu, dtype=torch.int32, device=self.device), "bits_per_symbol": int(bits), "ecc_mode": ecc_mode, "quant_variant": quant_variant})
        return encoded

    def decode_message(self, encoded:Dict[str,Any], snr_db:float=3.0, ecc_mode:str="repetition", quant_type:Optional[str]=None, channel_model:str="awgn", num_antennas:int=1):
        qtype=(quant_type if quant_type is not None else encoded.get("quant_type","int8_det")).lower()
        quant_variant=encoded.get("quant_variant","deterministic"); bits=encoded.get("bits_per_symbol", self.bits_default)
        if quant_variant=="fake":
            deq_arr=encoded.get("deq_embedding", None)
            if deq_arr is None:
                emb=encoded.get("embedding", None)
                if emb is None:
                    return {"original": encoded.get("original"), "decoded":"Unable to decode", "metrics": {"cosine_similarity":0.0,"bert_score":0.0,"key":0.0}, "ber": None}
                deq=np.array(emb, dtype=float)
            else:
                deq=np.array(deq_arr, dtype=float)
            snr_linear=10**(snr_db/10.0)
            noise_std=math.sqrt(1.0/max(1e-9,snr_linear))
            noise=np.random.normal(0.0, noise_std, size=deq.shape).astype(float)
            noisy = deq + noise
            z_hat = torch.tensor(noisy, dtype=torch.float32, device=self.device)
            z_hat = F.normalize(z_hat.unsqueeze(0), p=2, dim=1).squeeze(0)
            orig_emb = encoded.get("embedding", encoded.get("embedding_raw", None))
            if orig_emb is None: orig_emb = torch.tensor(deq, device=self.device, dtype=torch.float32)
            elif not isinstance(orig_emb, torch.Tensor): orig_emb = torch.tensor(orig_emb, device=self.device, dtype=torch.float32)
            else: orig_emb = orig_emb.to(self.device, dtype=torch.float32)
            orig_for_metrics = F.normalize(orig_emb, p=2, dim=0)
            decoded_text = self._hybrid_decode(z_hat)
            metrics = self._compute_metrics(encoded.get("original"), decoded_text, orig_for_metrics, z_hat)
            return {"original": encoded.get("original"), "decoded": decoded_text, "metrics": metrics, "ber": None}

        noisy, ber = self._simulate_channel_flip(encoded["ecc_encoded"], snr_db=snr_db, model=channel_model, bits_per_symbol=encoded.get("bits_per_symbol", self.bits_default), num_antennas=num_antennas)
        ECCClass = ECC_REG.get(ecc_mode, RepetitionECC); ecc_obj = ECCClass()
        q_hat_list = ecc_obj.decode(noisy, encoded.get("parity_allocation", []), encoded.get("bits_per_symbol", self.bits_default))
        q_hat = torch.tensor(q_hat_list, dtype=torch.float32, device=self.device)
        z_hat = self._int_dequantize(q_hat.unsqueeze(0), encoded["q_max"], torch.tensor([encoded.get("scale",1.0)], device=self.device))
        z_hat = F.normalize(z_hat, p=2, dim=1).squeeze(0)
        decoded_text = self._hybrid_decode(z_hat)
        orig_emb = encoded.get("embedding", encoded.get("embedding_raw", None))
        if orig_emb is None: orig_emb = self._encode_batch([encoded["original"]], convert_to_tensor=True).squeeze(0)
        elif not isinstance(orig_emb, torch.Tensor): orig_emb = torch.tensor(orig_emb, device=self.device, dtype=torch.float32)
        else: orig_emb = orig_emb.to(self.device, dtype=torch.float32)
        orig_for_metrics = F.normalize(orig_emb, p=2, dim=0)
        metrics = self._compute_metrics(encoded.get("original"), decoded_text, orig_for_metrics, z_hat)
        return {"original": encoded.get("original"), "decoded": decoded_text, "metrics": metrics, "ber": ber}

    def _int_dequantize(self, q:torch.Tensor, q_max:int, s:torch.Tensor):
        return s[:, None] * (q / q_max)

    def train_parity_policy(self, train_messages:List[str], num_epochs:int=10, episodes_per_epoch:int=128, snr_db:float=3.0, sampling:str="gumbel_topk", channel_model:str="awgn", num_antennas:int=1, lr_policy:float=5e-4, lr_value:float=5e-4):
        k=self.internal_emb_dim; T_total=max(1,int(k*self.ecc_rate))
        history=[]
        self.policy_optim=torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optim=torch.optim.Adam(self.value_net.parameters(), lr=lr_value)
        for epoch in range(num_epochs):
            epoch_reward=0.0; epoch_ploss=0.0; epoch_vloss=0.0
            for ep in range(episodes_per_epoch):
                text=random.choice(train_messages)
                z=self._encode_batch([text], convert_to_tensor=True).squeeze(0).to(self.device)
                counts, probs_cpu, logp = self._sample_allocation_from_policy(z, T_total, sampling=sampling)
                counts=counts.to(self.device)
                encoded=self.encode_message(text, parity_allocation=counts, B=self.B, ecc_mode="repetition", quant_type=f"int{self.B}_det")
                decoded=self.decode_message(encoded, snr_db=snr_db, ecc_mode="repetition", channel_model=channel_model, num_antennas=num_antennas)
                cos=decoded["metrics"].get("cosine_similarity",0.0); key=decoded["metrics"].get("key",0.0)
                cos_f=float(cos) if cos is not None else 0.0; key_f=float(key) if key is not None else 0.0
                D_S=self.alpha*(1.0-cos_f)+(1.0-self.alpha)*(1.0-key_f)
                parity_cost=float(counts.sum().item())
                reward=float(-D_S - self.lambda_reg*parity_cost)
                with torch.no_grad():
                    value_pred = float(self.value_net(self._safe_tensor(z).unsqueeze(0)).detach().cpu().numpy().squeeze())
                adv = reward - value_pred
                # policy update
                self.policy_optim.zero_grad()
                entropy = - (probs_cpu.to(self.device) * torch.log(torch.clamp(probs_cpu.to(self.device),1e-9,1.0))).sum()
                policy_loss = -adv * logp.to(self.device) - 1e-3 * entropy
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_optim.step()
                # value update
                self.value_optim.zero_grad()
                val_pred_tensor = self.value_net(self._safe_tensor(z).unsqueeze(0))
                val_target = torch.tensor([reward], dtype=torch.float32, device=self.device)
                val_loss = F.mse_loss(val_pred_tensor, val_target)
                val_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.value_optim.step()
                epoch_reward+=reward; epoch_ploss+=float(policy_loss.detach().cpu().numpy()) if isinstance(policy_loss, torch.Tensor) else float(policy_loss); epoch_vloss+=float(val_loss.detach().cpu().numpy())
                history.append({"epoch":epoch+1,"episode":ep+1,"reward":reward,"D_S":D_S,"parity_cost":parity_cost})
            # derive deterministic allocation
            prob_accum=torch.zeros(k, dtype=torch.float32, device=self.device)
            ns=min(1000, len(train_messages)); samples=random.sample(train_messages, ns)
            for t in samples:
                zt=self._encode_batch([t], convert_to_tensor=True).squeeze(0).to(self.device)
                with torch.no_grad():
                    p=self.policy(zt.unsqueeze(0)).squeeze(0).cpu()
                prob_accum += p.to(self.device)
            avg=prob_accum / max(1, ns)
            if self.action_family=="soft":
                det=torch.floor(avg * int(k*self.ecc_rate)).int()
                deficit = int(k*self.ecc_rate) - int(det.sum().item())
                if deficit>0:
                    top=torch.argsort(avg, descending=True)[:deficit]
                    for idx in top: det[idx]+=1
                self.parity_allocation=det
            else:
                ksel=self.topk_k or max(1, int((k*self.ecc_rate)//max(1,self.topk_tau)))
                top=torch.argsort(avg, descending=True)[:ksel]
                det=torch.zeros(k, dtype=torch.int32, device=self.device)
                for idx in top: det[idx]=int(self.topk_tau)
                self.parity_allocation=det
            logger.info("Policy Epoch %d/%d | avg_reward %.4f | avg_policy_loss %.6f | avg_val_loss %.6f", epoch+1, num_epochs, epoch_reward/episodes_per_epoch, epoch_ploss/episodes_per_epoch, epoch_vloss/episodes_per_epoch)
        return history

    # evaluation harness (computes basic metrics; heavy metrics computed outside if enabled)
    def evaluate_methods(self, test_messages:List[str], methods:tuple=("uniform","importance","rl","no_uep","random"), snr_levels:tuple=(0,3,10), antennas:tuple=(1,), channels:tuple=("awgn","rayleigh"), ecc_schemes:tuple=("repetition",), quant_types:tuple=("int8_det",), outdir:str="./results", sample_size:int=200, n_bootstrap:int=200, seed:int=123, enable_bleu:bool=False, bleu_max_n:int=4, enable_rouge:bool=False, enable_meteor:bool=False, enable_bertscore:bool=False):
       
        global _HAS_BERTSCORE
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        os.makedirs(outdir, exist_ok=True)
        timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv=os.path.join(outdir, f"eval_summary_{timestamp}.csv")
        permsg_jsonl=os.path.join(outdir, f"eval_per_message_{timestamp}.jsonl")
        # optional metric initializations done by caller; here we assume top-level imports if flags true

        def chrF_score(ref,hyp,max_n=6,beta=2.0):
            try:
                if len((ref or "").strip())==0 and len((hyp or "").strip())==0: return 1.0
                if len((ref or "").strip())==0 or len((hyp or "").strip())==0: return 0.0
                def char_ngrams(s,n):
                    s=(s or "").replace(" ","")
                    if len(s)<n: return []
                    return [s[i:i+n] for i in range(len(s)-n+1)]
                precisions=[]; recalls=[]
                for n in range(1,max_n+1):
                    r_ngr=char_ngrams(ref,n); h_ngr=char_ngrams(hyp,n)
                    if not r_ngr and not h_ngr: continue
                    r_count=defaultdict(int); h_count=defaultdict(int)
                    for g in r_ngr: r_count[g]+=1
                    for g in h_ngr: h_count[g]+=1
                    overlap=0
                    for g,cnt in h_count.items(): overlap += min(cnt, r_count.get(g,0))
                    prec = overlap / max(1, len(h_ngr)); rec = overlap / max(1, len(r_ngr))
                    precisions.append(prec); recalls.append(rec)
                if not precisions: return 0.0
                P=sum(precisions)/len(precisions); R=sum(recalls)/len(recalls)
                if P+R==0: return 0.0
                beta2 = beta*beta
                return float((1+beta2)*P*R/(beta2*P + R + 1e-12))
            except Exception:
                return 0.0

        def bootstrap_ci(values, n_boot=n_bootstrap, alpha=0.05):
            vals=np.array(values, dtype=float); n=len(vals)
            if n==0: return (0.0,0.0,0.0)
            means=[]
            for _ in range(n_boot):
                sample=np.random.choice(vals, size=n, replace=True); means.append(sample.mean())
            lo=float(np.percentile(means, 100*alpha/2)); hi=float(np.percentile(means, 100*(1-alpha/2)))
            return float(vals.mean()), lo, hi

        k=self.internal_emb_dim; T_total=max(1,int(k*self.ecc_rate))
        allocs={}
        allocs["uniform"] = torch.tensor(self._make_uniform_alloc(k,T_total), dtype=torch.int32, device=self.device)
        if self.importance_scores is None: self._compute_importance_scores()
        allocs["importance"]=self.parity_allocation
        allocs["rl"]=self.parity_allocation
        allocs["no_uep"]=torch.zeros(k, dtype=torch.int32, device=self.device)
        allocs["random"]=torch.randint(0, int(T_total/k)+1, (k,), dtype=torch.int32, device=self.device)
        sum_rand = allocs["random"].sum().item()
        if sum_rand > 0:
            allocs["random"] = torch.floor(allocs["random"] / sum_rand * T_total).int()
        summary_rows=[]; permsg_fh=open(permsg_jsonl,"w",encoding="utf-8")
        total_runs=len(methods)*len(channels)*len(snr_levels)*len(antennas)*len(ecc_schemes)*len(quant_types)
        run_idx=0

        # optionally import heavy libs lazily
        if enable_bleu:
            try:
                import nltk
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                HAS_NLTK=True
                smooth=SmoothingFunction().method1
            except Exception:
                HAS_NLTK=False; logger.warning("nltk BLEU unavailable")
        else:
            HAS_NLTK=False

        if enable_rouge:
            try:
                from rouge_score import rouge_scorer
                rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                HAS_ROUGE=True
            except Exception:
                HAS_ROUGE=False; logger.warning("rouge-score unavailable")
        else:
            HAS_ROUGE=False

        if enable_meteor:
            try:
                import nltk
                nltk.download('wordnet', quiet=True)
                from nltk.translate.meteor_score import meteor_score
                HAS_METEOR=True
            except Exception:
                HAS_METEOR=False; logger.warning("METEOR unavailable")
        else:
            HAS_METEOR=False

        if enable_bertscore:
            try:
                from bert_score import BERTScorer
                # configure device string
                device_str = "cuda" if torch.cuda.is_available() else ("mps" if str(DEVICE)=="mps" else "cpu")
                bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device_str)
                _HAS_BERTSCORE=True
            except Exception:
                _HAS_BERTSCORE=False; logger.warning("BERTScore unavailable")

        for channel in channels:
            for snr in snr_levels:
                for ant in antennas:
                    for ecc in ecc_schemes:
                        for qtype in quant_types:
                            run_idx += 1
                            logger.info("Eval [%d/%d] channel=%s snr=%s ant=%s ecc=%s quant=%s", run_idx, total_runs, channel, snr, ant, ecc, qtype)
                            subset = test_messages if len(test_messages)<=sample_size else random.sample(test_messages, sample_size)
                            all_metrics = {m: {'cos': [], 'chrf': [], 'ent_frac': [], 'ber': [], 'dS': [], 'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [], 'rougeL': [], 'meteor': [], 'bert': []} for m in methods}
                            for txt in subset:
                                for method in methods:
                                    try:
                                        alloc = allocs[method]
                                        encoded=self.encode_message(txt, parity_allocation=alloc, B=self.B, quant_type=qtype, ecc_mode=ecc)
                                        decoded=self.decode_message(encoded, snr_db=snr, ecc_mode=ecc, channel_model=channel, num_antennas=ant, quant_type=qtype)
                                    except Exception as e:
                                        logger.warning("Processing failed for text: %s... (%s)", str(txt)[:80], e)
                                        decoded={"decoded":"DECODE_ERROR","metrics":{"cosine_similarity":0.0,"bert_score":0.0,"key":0.0},"ber":None}
                                    hyp=decoded.get("decoded") or ""
                                    if not isinstance(hyp,str): hyp=str(hyp)
                                    ref=txt if isinstance(txt,str) else str(txt)
                                    metrics=decoded.get("metrics",{}) or {}
                                    cos=metrics.get("cosine_similarity",0.0) or 0.0
                                    key_val=metrics.get("key",0.0) or 0.0
                                    chrf=chrF_score(ref,hyp) or 0.0
                                    ent_frac=0.0
                                    ber_val=decoded.get("ber", 0.0) or 0.0
                                    cos_f, key_f, chrf_f, ent_f, ber_f = float(cos), float(key_val), float(chrf), float(ent_frac), float(ber_val)
                                    D_S = self.alpha*(1.0-cos_f) + (1.0-self.alpha)*(1.0-ent_f)
                                    all_metrics[method]['cos'].append(cos_f)
                                    all_metrics[method]['chrf'].append(chrf_f)
                                    all_metrics[method]['ent_frac'].append(ent_f)
                                    all_metrics[method]['ber'].append(ber_f)
                                    all_metrics[method]['dS'].append(D_S)
                                    # BLEU
                                    if HAS_NLTK:
                                        try:
                                            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                                            smooth=SmoothingFunction().method1
                                            tokens_ref = [ref.split()]
                                            tokens_hyp = hyp.split()
                                            b1 = sentence_bleu(tokens_ref, tokens_hyp, weights=(1,0,0,0), smoothing_function=smooth)
                                            b2 = sentence_bleu(tokens_ref, tokens_hyp, weights=(0.5,0.5,0,0), smoothing_function=smooth) if bleu_max_n>=2 else 0.0
                                            b3 = sentence_bleu(tokens_ref, tokens_hyp, weights=(1/3,1/3,1/3,0), smoothing_function=smooth) if bleu_max_n>=3 else 0.0
                                            b4 = sentence_bleu(tokens_ref, tokens_hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth) if bleu_max_n>=4 else 0.0
                                        except Exception:
                                            b1=b2=b3=b4=0.0
                                    else:
                                        b1=b2=b3=b4=0.0
                                    all_metrics[method]['bleu1'].append(b1)
                                    all_metrics[method]['bleu2'].append(b2)
                                    all_metrics[method]['bleu3'].append(b3)
                                    all_metrics[method]['bleu4'].append(b4)
                                    # ROUGE-L
                                    if HAS_ROUGE:
                                        try:
                                            score = rouge_scorer_obj.score(ref, hyp)['rougeL'].fmeasure
                                        except Exception:
                                            score=0.0
                                    else:
                                        score=0.0
                                    all_metrics[method]['rougeL'].append(score)
                                    # METEOR
                                    if HAS_METEOR:
                                        try:
                                            from nltk.translate.meteor_score import meteor_score
                                            m = meteor_score([ref], hyp)
                                        except Exception:
                                            m = 0.0
                                    else:
                                        m = 0.0
                                    all_metrics[method]['meteor'].append(m)
                                    # BERTScore
                                    if _HAS_BERTSCORE:
                                        try:
                                            P,R,F1 = bert_scorer.score([hyp],[ref])
                                            bert_f1 = float(F1.mean().cpu().numpy()) if hasattr(F1,'mean') else float(F1[0])
                                        except Exception:
                                            bert_f1 = 0.0
                                    else:
                                        bert_f1=0.0
                                    all_metrics[method]['bert'].append(bert_f1)
                                    permsg={"method":method,"channel":channel,"snr":snr,"antennas":ant,"ecc":ecc,"quant_type":qtype,"original":ref,"decoded":hyp,"cosine":cos_f,"chrf":chrf_f,"ber":ber_f,"D_S":D_S,"bleu1":b1,"bleu2":b2,"bleu3":b3,"bleu4":b4,"rougeL":score,"meteor":m,"bertscore":bert_f1}
                                    permsg_fh.write(json.dumps(permsg)+"\n")
                            permsg_fh.flush()
                            # aggregate & CIs per method
                            for method in methods:
                                mean_cos, cos_lo, cos_hi = bootstrap_ci(all_metrics[method]['cos'])
                                mean_chrf, chrf_lo, chrf_hi = bootstrap_ci(all_metrics[method]['chrf'])
                                mean_entity, ent_lo, ent_hi = bootstrap_ci(all_metrics[method]['ent_frac'])
                                mean_ber, ber_lo, ber_hi = bootstrap_ci(all_metrics[method]['ber'])
                                mean_dS, dS_lo, dS_hi = bootstrap_ci(all_metrics[method]['dS'])
                                mean_b1, b1_lo, b1_hi = bootstrap_ci(all_metrics[method]['bleu1'])
                                mean_b4, b4_lo, b4_hi = bootstrap_ci(all_metrics[method]['bleu4'])
                                mean_rouge, r_lo, r_hi = bootstrap_ci(all_metrics[method]['rougeL'])
                                mean_meteor, me_lo, me_hi = bootstrap_ci(all_metrics[method]['meteor'])
                                mean_bert, bert_lo, bert_hi = bootstrap_ci(all_metrics[method]['bert'])
                                # Wilcoxon p-values (RL vs uniform)
                                p_dS = p_cos = None
                                if "rl" in all_metrics and "uniform" in all_metrics:
                                    try:
                                        p_dS = wilcoxon(all_metrics["rl"]["dS"], all_metrics["uniform"]["dS"]).pvalue
                                        p_cos = wilcoxon(all_metrics["rl"]["cos"], all_metrics["uniform"]["cos"]).pvalue
                                    except ValueError as e:
                                        logger.warning("Wilcoxon failed: %s", e)
                                        p_dS = p_cos = float('nan')
                                row={"alpha":self.alpha,"lambda_reg":self.lambda_reg,"method":method,"channel":channel,"snr":snr,"antennas":ant,"ecc":ecc,"quant_type":qtype,"n_samples":len(all_metrics[method]['cos']),"mean_cosine":mean_cos,"ci_cos_lo":cos_lo,"ci_cos_hi":cos_hi,"mean_chrf":mean_chrf,"ci_chrf_lo":chrf_lo,"ci_chrf_hi":chrf_hi,"mean_entity_frac":mean_entity,"mean_ber":mean_ber,"mean_D_S":mean_dS,"mean_bleu1":mean_b1,"ci_bleu1_lo":b1_lo,"ci_bleu1_hi":b1_hi,"mean_bleu4":mean_b4,"ci_bleu4_lo":b4_lo,"ci_bleu4_hi":b4_hi,"mean_rougeL":mean_rouge,"mean_meteor":mean_meteor,"mean_bertscore":mean_bert, "p_dS_vs_uniform":p_dS, "p_cos_vs_uniform":p_cos, "ecc_rate":self.ecc_rate}
                                summary_rows.append(row)
        permsg_fh.close()
        try:
            import pandas as pd
            df=pd.DataFrame(summary_rows)
            df.to_csv(summary_csv, index=False)
            logger.info("Saved eval summary to %s", summary_csv)
        except Exception as e:
            logger.warning("Failed to save CSV: %s", e)
        return summary_rows

    def _make_uniform_alloc(self,k:int,T_total:int)->List[int]:
        base=np.zeros(k,dtype=np.int32); per=T_total//k; base+=per
        rem=T_total-base.sum()
        for idx in range(rem): base[idx]+=1
        return base.tolist()

# --- dataset helpers
def load_hf_texts(name="ag_news", split="train", limit=10000, cache_dir:Optional[str]=None):
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not available")
    ds = load_dataset(name, split=split, cache_dir=cache_dir)
    texts=[]
    for ex in ds:
        if isinstance(ex, dict):
            for k in ("text","content","sentence","article","description"):
                if k in ex and isinstance(ex[k], str) and ex[k].strip():
                    texts.append(ex[k].strip()); break
        elif isinstance(ex,str):
            texts.append(ex)
        if len(texts)>=limit: break
    return texts

def generate_synthetic_dataset(n=10000, seed=123):
    random.seed(seed); np.random.seed(seed)
    pool=["Traffic alert: {event} on Highway {h} {dir}","Weather warning: {phenomenon} expected in {region}","Sensor alert: {sensor} reading exceeded threshold at {loc}","Home automation: Set {device} to {value} at {time}","Medical alert: Patient {pid} heart rate above {hr} bpm"]
    events=["accident","congestion","pile-up","vehicle breakdown"]; dirs=["Northbound","Southbound","eastbound","westbound"]
    phenomena=["heavy snow","thunderstorms","dense fog"]; regions=["northern regions","downtown","coastal zone"]
    sensors=["temperature sensor","vibration sensor","pressure sensor"]; devices=["thermostat","sprinkler","garage door"]
    values=["21°C","on","off","auto"]; times=["18:00","07:30","22:15"]; pids=["P001","P002","P123"]; locs=["substation SS-4","door D-7","server room"]
    msgs=[]
    for _ in range(n):
        tpl=random.choice(pool); mapping={}
        if "{event}" in tpl: mapping["event"]=random.choice(events)
        if "{h}" in tpl: mapping["h"]=random.randint(1,199)
        if "{dir}" in tpl: mapping["dir"]=random.choice(dirs)
        if "{phenomenon}" in tpl: mapping["phenomenon"]=random.choice(phenomena)
        if "{region}" in tpl: mapping["region"]=random.choice(regions)
        if "{sensor}" in tpl: mapping["sensor"]=random.choice(sensors)
        if "{loc}" in tpl: mapping["loc"]=random.choice(locs)
        if "{device}" in tpl: mapping["device"]=random.choice(devices)
        if "{value}" in tpl: mapping["value"]=random.choice(values)
        if "{time}" in tpl: mapping["time"]=random.choice(times)
        if "{pid}" in tpl: mapping["pid"]=random.choice(pids)
        if "{hr}" in tpl: mapping["hr"]=random.randint(60,180)
        msgs.append(tpl.format(**mapping))
    return msgs

# --- plotting helper
def plot_ablation(x_vals, means, lo, hi, xlabel, ylabel, outpath):
    plt.figure(figsize=(8,5))
    try:
        plt.plot(x_vals, means, marker='o'); plt.fill_between(x_vals, lo, hi, alpha=0.2)
    except Exception:
        plt.plot(range(len(means)), means, marker='o'); plt.fill_between(range(len(means)), lo, hi, alpha=0.2)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(f"{ylabel} vs {xlabel}"); plt.grid(True); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close(); logger.info("Saved %s", outpath)

# --- full ablation runner
# --- full ablation runner ---------------------------------------------------------
def run_full_ablation(
        outdir="./results", seed=123, n_messages=4000, train_epochs=3, episodes=128,
        eval_sample_size=400, n_bootstrap=200, sweep_counts=5, dataset="hf",
        hf_name="ag_news", cache_dir=None, quant_bits_list=None, channels_list=None,
        ecc_list=None, action_families=None, quant_variants=None,
        enable_bleu=False, bleu_max_n=4, enable_rouge=False, enable_meteor=False,
        enable_bertscore=False, max_runs: Optional[int] = None):

    deterministic_seed(seed)
    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------------------------- data
    if dataset == "hf":
        if not HAS_DATASETS:
            logger.warning("datasets not available → synthetic")
            msgs = generate_synthetic_dataset(n=n_messages, seed=seed)
        else:
            logger.info("Loading HF %s (limit=%d)", hf_name, n_messages)
            msgs = load_hf_texts(name=hf_name, split="train",
                                 limit=n_messages, cache_dir=cache_dir)
    else:
        logger.info("Using synthetic dataset")
        msgs = generate_synthetic_dataset(n=n_messages, seed=seed)

    # --------------------------------------------------------------------- axes
    alphas      = np.linspace(0.0, 1.0, sweep_counts)
    lambdas     = np.logspace(-3, 0, sweep_counts)
    snrs        = np.linspace(0, 12, sweep_counts)
    ecc_rates   = np.linspace(0.0, 0.2, sweep_counts)

    quant_bits_list = quant_bits_list or [4, 8, 12, 16]
    channels_list   = channels_list   or ["awgn","rayleigh","rician","nakagami","bsc","burst"]
    ecc_list        = ecc_list        or ["repetition","uep","reedsolomon","ldpc"]
    action_families = action_families or ["soft","topk"]
    quant_variants  = quant_variants  or ["deterministic","stochastic","fake"]

    results = []                                 # ← **only once**

    # --------------------------------------------------------------------- helper
    def run_config(cfg):
        logger.info("Run config: %s", cfg)
        scs = SemanticCommunicationSystem(
            ecc_rate   = float(cfg.get("ecc_rate",   0.05)),
            B          = int  (cfg.get("B",          8)),
            mock_dim   = 64,
            alpha      = float(cfg.get("alpha",      0.5)),
            lambda_reg = float(cfg.get("lambda_reg", 0.0)),
            action_family = cfg.get("action_family","soft"),
            topk_tau   = int  (cfg.get("topk_tau",   1)),
            topk_k     = cfg.get("topk_k", None)
        )
        scs.initialize_knowledge_base(msgs)

        scs.train_parity_policy(
            msgs,
            num_epochs        = int(cfg.get("train_epochs", train_epochs)),
            episodes_per_epoch= int(cfg.get("episodes",    episodes)),
            snr_db            = float(cfg.get("train_snr_db", 3.0)),
            sampling          = "gumbel_topk",
            channel_model     = cfg.get("train_channel","awgn")
        )

        qv = cfg.get("quant_variant","deterministic")
        qb = int(cfg.get("quant_bits",8))
        qtype = f"int{qb}_det" if qv=="deterministic" else \
                f"int{qb}_stoch" if qv=="stochastic" else f"fake{qb}"

        summary = scs.evaluate_methods(
            msgs, methods=("rl","uniform","importance","no_uep","random"),
            snr_levels=(int(cfg.get("eval_snr_db",3)),),
            antennas=(1,), channels=(cfg.get("eval_channel","awgn"),),
            ecc_schemes=(cfg.get("ecc","repetition"),),
            quant_types=(qtype,),
            outdir=outdir, sample_size=eval_sample_size,
            n_bootstrap=n_bootstrap, seed=seed,
            enable_bleu=enable_bleu, bleu_max_n=bleu_max_n,
            enable_rouge=enable_rouge, enable_meteor=enable_meteor,
            enable_bertscore=enable_bertscore
        )
        for subrow in summary:
            rowcfg = dict(cfg)
            rowcfg.update(subrow)
            results.append(rowcfg)
        return results  # Return all per-method rows

    # --------------------------------------------------------------------- defaults
    default = {
        "alpha":0.5, "lambda_reg":1e-2, "eval_snr_db":3, "ecc_rate":0.05,
        "quant_bits":8, "train_epochs":train_epochs, "episodes":episodes, "B":8,
        "train_snr_db":3.0, "train_channel":"awgn"
    }

    # --------------------------------------------------------------------- early sweeps
    for a in alphas:
        cfg = dict(default)
        cfg.update({"alpha":float(a), "eval_channel":"awgn","ecc":"repetition",
                    "quant_variant":"deterministic"})
        run_config(cfg)

    for lam in lambdas:
        cfg = dict(default)
        cfg.update({"lambda_reg":float(lam), "eval_channel":"awgn","ecc":"repetition",
                    "quant_variant":"deterministic"})
        run_config(cfg)

    for s in snrs:
        cfg = dict(default)
        cfg.update({"eval_snr_db":int(s), "eval_channel":"awgn","ecc":"repetition",
                    "quant_variant":"deterministic"})
        run_config(cfg)

    for e in ecc_rates:
        cfg = dict(default)
        cfg.update({"ecc_rate":float(e), "eval_channel":"awgn","ecc":"repetition",
                    "quant_variant":"deterministic"})
        run_config(cfg)

    for q in quant_bits_list:
        for qv in quant_variants:
            cfg = dict(default)
            cfg.update({"quant_bits":int(q), "quant_variant":qv,
                        "eval_channel":"awgn","ecc":"repetition","action_family":"soft"})
            run_config(cfg)

    # --------------------------------------------------------------------- cross-product
    import itertools, random
    full_grid = [
        {
            "alpha":a, "lambda_reg":lam, "eval_snr_db":int(s), "ecc_rate":e,
            "eval_channel":ch, "ecc":ecc, "action_family":af,
            "quant_bits":q, "quant_variant":qv,
            "train_snr_db":3.0, "train_channel":"awgn", "B":8,
            "train_epochs":train_epochs, "episodes":episodes
        }
        for a,lam,s,e,ch,ecc,af,q,qv in itertools.product(
            alphas, lambdas, snrs, ecc_rates,
            channels_list, ecc_list, action_families,
            quant_bits_list, quant_variants)
    ]

    if max_runs is not None and 0 < max_runs < len(full_grid):
        random.seed(seed)
        full_grid = random.sample(full_grid, max_runs)
        logger.info("Sampled %d configs (max-runs=%d)", len(full_grid), max_runs)

    for idx, cfg in enumerate(full_grid, 1):
        logger.info("Ablation [%d/%d] %s", idx, len(full_grid),
                    " | ".join(f"{k}={v}" for k,v in cfg.items()))
        run_config(cfg)

    # --------------------------------------------------------------------- save
    out_json = os.path.join(outdir,
               f"ablation_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved raw results → %s (%d runs)", out_json, len(results))

    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(outdir, "ablation_results_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Saved CSV summary → %s", csv_path)
    except Exception as e:
        logger.warning("CSV save failed: %s", e)
    return results

    if results:
        dfv = pd.DataFrame(results).dropna(subset=["mean_cosine"])
        best_cos = dfv.loc[dfv["mean_cosine"].idxmax()].to_dict()
        best_ds  = dfv.loc[dfv["mean_D_S"].idxmin()].to_dict()
        best = {"best_by_cosine":best_cos, "best_by_D_S":best_ds}
        with open(os.path.join(outdir, "best_config_summary.json"),"w") as f:
            json.dump(best, f, indent=2)
        print("\n=== Best Configs ===")
        print("Cosine :", {k:v for k,v in best_cos.items()
                          if k in ("alpha","lambda_reg","mean_cosine")})
        print("D_S    :", {k:v for k,v in best_ds.items()
                          if k in ("alpha","lambda_reg","mean_D_S")})

# --- CLI
def parse_args():
    p = argparse.ArgumentParser(description="RL ablation + optional metrics")
    p.add_argument("--outdir", type=str, default="./results", help="output directory")
    p.add_argument("--dataset", choices=("hf","synthetic"), default="hf")
    p.add_argument("--hf_name", type=str, default="ag_news")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--n_messages", type=int, default=4000)
    p.add_argument("--train_epochs", type=int, default=3)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--eval_sample_size", type=int, default=400)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--sweep_counts", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--quick", action="store_true", help="small quick run")
    p.add_argument("--enable_bleu", action="store_true", help="compute BLEU1-4")
    p.add_argument("--bleu_max_n", type=int, default=4, help="max n for BLEU (1-4)")
    p.add_argument("--enable_rouge", action="store_true", help="compute ROUGE-L")
    p.add_argument("--enable_meteor", action="store_true", help="compute METEOR")
    p.add_argument("--enable_bertscore", action="store_true", help="compute BERTScore")
    p.add_argument("--max-runs", type=int, default=200,
                  help="Maximum number of ablation configurations to run. "
                       "If omitted, runs full Cartesian product.")
    return p.parse_args()

def main():
    args = parse_args()
    if args.quick:
        args.n_messages = min(1000, args.n_messages)
        args.train_epochs = min(1, args.train_epochs)
        args.episodes = min(32, args.episodes)
        args.eval_sample_size = min(100, args.eval_sample_size)
        args.sweep_counts = min(3, args.sweep_counts)

    deterministic_seed(args.seed)
    run_full_ablation(
        outdir=args.outdir,
        seed=args.seed,
        n_messages=args.n_messages,
        train_epochs=args.train_epochs,
        episodes=args.episodes,
        eval_sample_size=args.eval_sample_size,
        n_bootstrap=args.n_bootstrap,
        sweep_counts=args.sweep_counts,
        dataset=args.dataset,
        hf_name=args.hf_name,
        cache_dir=args.cache_dir,
        enable_bleu=args.enable_bleu,
        bleu_max_n=args.bleu_max_n,
        enable_rouge=args.enable_rouge,
        enable_meteor=args.enable_meteor,
        enable_bertscore=args.enable_bertscore,
        max_runs=args.max_runs  # ← NOW PASSED
    )

if __name__ == "__main__":
    main()