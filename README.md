# Transformer-Lab# Transformer-Lab

**Status:** In Progress 🚧  
This lab includes experimental Transformer-based models for generating creative text. The repo contains early-stage implementations using Multi-Head Attention (MHA) and Multi-Latent Attention (MLA), with supporting scripts for training and querying.

---

## 🧠 Overview

The core idea is to develop and compare transformer models that can learn to generate story-like sequences from prompt inputs. The architecture includes:

- Custom tokenization (SentencePiece)
- RoPE (Rotary Positional Embeddings)
- Flexible model architecture (MHA vs MLA)
- Story chunking for dataset processing
- Simple inference interface

---

## 🧪 Testing Models

| Script | Description |
|--------|-------------|
| `testing_2_MHA.py` | Testing script using MHA-based transformer |
| `testing_2_MLA.py` | Testing script using MLA-based transformer |

Both scripts train a lightweight decoder-only transformer (`TinyTransformer`) to generate movie plot-like texts.

---

## 🔄 Pulling System (Prompt-Response)

| Script | Description |
|--------|-------------|
| `Pulling_2_MHA.py` | Runs inference (prompt in → story out) using the MHA model |
| `Pulling_2_MLA.py` | Same but with the MLA model |

This works like a mini-chat between a user and the model. The pulling scripts simulate conversations by passing queries and retrieving generated outputs.

---

## ⚙️ Key Features

- Self-contained scripts with on-the-fly dependency installs
- Logs all progress to `training.log`
- Early stopping and checkpoint saving
- Model generates and saves story outputs
- Tokenizer trained if not already present

---

## 📂 Files & Output

- `training_metrics.txt`: Training/validation loss & perplexity
- `training.log`: Logging details per epoch
- `generated_stories.txt`: Output from the model
- `movie_tokenizer.model`: Saved tokenizer model
- `/checkpoints/`: Saved checkpoints every few epochs

---

## 🔮 Planned Improvements

- Modularize code (split model, data, utils)
- Add more structured config loading
- Integrate web interface for story generation
- Improve MLA attention mechanism
- Benchmark performance between MHA and MLA

---

## 📅 System Context

- Development Date: April 2025
- Environment: PyTorch + custom RoPE
- Dataset: JSON file (`Movie_XD__420.json`) with `body` field for plot content

---

---

## 📌 Note

This is an early-stage lab environment. Code will be modified and cleaned later. Stay tuned for updates :) 

---




