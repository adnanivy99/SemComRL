# SemComRL

**SemComRL** is a Python research software framework for **reinforcement-learned unequal error protection (UEP)** in **text-based semantic communication**.

The software implements the method introduced in the associated research article:

**Moirangthem Tiken Singh, Adnan Arif, _Reinforcement-learned unequal error protection for quantized semantic embeddings_, Array.**

It provides a modular experimental pipeline for semantic embedding, quantization, adaptive redundancy allocation, channel simulation, decoding, evaluation, and large-scale ablation studies.

---

## Overview

Conventional communication systems aim to recover bits accurately. In contrast, **semantic communication** aims to preserve the **meaning** of the transmitted message.

SemComRL studies this problem by representing text as sentence embeddings, quantizing those embeddings, and then allocating protection **adaptively across embedding dimensions** using **reinforcement learning**. The framework supports controlled experiments under bandwidth and noise constraints and enables reproducible evaluation across different coding schemes, channel models, and quantization settings.

---

## Main features

- Reinforcement-learned unequal error protection for quantized semantic embeddings
- Text embedding using `sentence-transformers`
- Deterministic, stochastic, and fake quantization modes
- Repetition-based and comparative ECC support
- Channel simulation for:
  - AWGN
  - Rayleigh
  - Rician
  - Nakagami
  - BSC
  - Burst channels
- Hybrid semantic decoding using embedding similarity and retrieval
- Multiple evaluation metrics, including:
  - cosine similarity
  - chrF
  - BLEU
  - ROUGE-L
  - METEOR
  - BERTScore
- Automated full ablation runner for systematic experiments
- Support for both Hugging Face datasets and synthetic data generation

---

## Software architecture

The framework follows the pipeline below:

1. **Input text message**
2. **Sentence embedding**
3. **Embedding normalization and quantization**
4. **RL-based parity/redundancy allocation**
5. **Error control coding**
6. **Noisy channel simulation**
7. **Decoding and dequantization**
8. **Hybrid semantic retrieval**
9. **Metric computation and result logging**

---

## Core components

The main script is:

```bash
semantic_comm_rl_full_ablation_metrics.py

## Installation

```bash
pip install -r requirements.txt
