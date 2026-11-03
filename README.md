---
library_name: pytorch
license: other
tags:
  - kv-cache
  - llm
  - rag
  - vector-database
  - model-compression
  - qwen2
  - llama-3
pipeline_tag: text-generation
---

# HyperCache AI — Unified KV-Cache & Vector DB Acceleration SDK

[![PyPI - Version](https://img.shields.io/pypi/v/hypercache-ai?color=blue&label=PyPI%20Package)](https://pypi.org/project/hypercache-ai/)
[![Python Versions](https://img.shields.io/badge/python-3.8%20%E2%80%93%203.13%2B-blue)](https://pypi.org/project/hypercache-ai/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HyperCache--AI-yellow)](https://huggingface.co/Walid-Kotb-1/HyperCache-AI)
[![Commercial License](https://img.shields.io/badge/License-Commercial%20PRO-green)](https://kotbster7.gumroad.com/l/hypercache-ai)

**Cut LLM KV-Cache VRAM by 58.2% (`+30%` faster decoding) AND compress Vector Database / RAG Embeddings by 57.3% (`2.34x` faster search) with zero model retraining.**

```bash
pip install hypercache-ai
```

👉 **[Get Your Instant Annual PRO License Key on Gumroad ($99/yr Launch Offer)](https://kotbster7.gumroad.com/l/hypercache-ai)**

![HyperCache AI Benchmark](./Screenshot_Qwen2.5-3B-Instruct.png)

HyperCache AI is a dual-engine runtime acceleration SDK for **PyTorch / Hugging Face LLMs** (`Qwen2.5`, `Llama-3`, `Mistral`, `Gemma`) and **Enterprise Vector Databases** (`Pinecone`, `Milvus`, `Qdrant`, `FAISS`, `pgvector`):
1. **Engine 1 — LLM KV-Cache Runtime Compressor:** Preserves full-resolution semantic blocks while condensing historical background tokens on the fly, achieving **100% verbatim Needle-in-a-Haystack recall** while cutting GPU KV-Cache memory in half.
2. **Engine 2 — Vector DB & RAG Embedding Compressor:** Compresses high-dimensional embedding vectors (e.g., `1536-D` OpenAI / Cohere / BGE embeddings down to `656-D`), cutting Vector Database RAM bills by **57.3%** while preserving **100.0% Top-1 match accuracy** and **99.92% cosine similarity correlation**.

---

## Verified Production Benchmarks

### 1. Live 3B & 7B LLM Needle-in-a-Haystack Retrieval (NVIDIA Tesla T4 — FP16)
*Task: Retrieve a confidential vault authorization key (`'HC-ENTERPRISE-9940-X'`) and reserve liquidity threshold (`90.37 million USD`) buried at token ~160 inside a 2,765-token cloud infrastructure report.*

#### A) `Qwen/Qwen2.5-3B-Instruct` (100% Native GPU Resident on Tesla T4)

| Compression Method | Cached Tokens | KV-Cache VRAM | Decoding Speed | Exact Answer Retrieved? | Actual Model Output |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Uncompressed Baseline (100% RAM)** | 2,764 | 97.17 MB | 15.2 tok/s | **Yes (100%)** | `The primary vault authorization key is 'HC-ENTERPRISE-9940-X' and the reserve liquidity threshold is 90.37 million USD` |
| **2. HyperCache AI SDK (`-58.2% RAM`)** | **1,156** | **40.64 MB** | **19.7 tok/s (+30%)** | **Yes (100%)** | `The primary vault authorization key is 'HC-ENTERPRISE-9940-X' and the reserve liquidity threshold is 90.37 million USD` |
| **3. Standard Sliding Window (`-58.2% RAM`)** | 1,156 | 40.64 MB | 19.1 tok/s | **Failed (0%)** | `The primary vault authorization key is the exact primary vault authorization key...` *(Hallucinates & loops)* |

#### B) `Qwen/Qwen2.5-7B-Instruct` (Large-Scale VRAM Reduction)
* **Uncompressed Baseline:** `2,764 tokens` (**`151.16 MB VRAM`**) → Retrieves `'HC-ENTERPRISE-9940-X'` and `90.37 million USD`.
* **HyperCache AI SDK (`-58.2% RAM`):** **`1,156 tokens`** (**`63.22 MB VRAM`**) → Retrieves the **exact same verbatim output** while saving **`87.94 MB VRAM`** per single request!

---

### 2. Vector Database & RAG Embedding Benchmark (`5,000` Docs, `1536-D` → `656-D`)

| Vector DB Compression Method (At 42.7% RAM Budget) | Compressed Dim | RAM Saved | Top-1 Match (↑) | Recall@10 (↑) | Cosine Score Correlation (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Standard Dimension Truncation** | 656-D | 57.29% | 99.0% | 22.90% | 66.28% |
| **2. HyperCache Vector DB Engine (Ours)** | **656-D** | **57.29%** | **100.0%** | **95.20%** | **99.92%** |

---

### 3. Multi-Head Long-Context Attention Fidelity (`N = 4,096` Tokens, 16 Heads)

| Compression Method (At Identical 42.5% RAM Budget) | Output Cosine Similarity (↑) | Relative L2 Error (↓) | Fact Recall (↑) | Theoretical Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **1. Standard Sliding Window** | 66.73% | 115.84% | 40.0% | 2.35x |
| **2. Uniform Strided Subsampling** | 63.47% | 114.45% | 20.0% | 2.35x |
| **3. HyperCache AI Engine (Ours)** | **99.96%** | **2.93%** | **100.0%** | **2.35x** |

---

## Instant 30-Second Installation & Evaluation (Free Trial Key Included)

You can install HyperCache AI directly from **Official PyPI** and evaluate both engines right now on your own machine or in Google Colab using our public evaluation key:
* **Public Evaluation Key:** `HYPERCACHE-FREE-TRIAL-2026`
  *(Hardware-locked to **15 free evaluation compressions** per machine, up to **4,096 tokens/dimensions** per call).*

### Step 1: One-Line Installation via `pip`
**Universal Python Version Support (3.8 – 3.13+):** `hypercache-ai` decrypts and compiles the runtime engine in volatile RAM directly on your target Python interpreter, ensuring seamless compatibility across **Python 3.8 through 3.13+** (including Google Colab, Kaggle, Linux, macOS, and Windows).

```bash
pip install hypercache-ai torch transformers accelerate
```

### Step 2: Use Both Engines in Python (3 Lines of Code)
```python
import hypercache_sdk_client

# 1. Unlock the AES-256-GCM dual engine in RAM
engine = hypercache_sdk_client.load_hypercache_engine(
    license_key="HYPERCACHE-FREE-TRIAL-2026"
)

# 2A. Compress HuggingFace LLM KV-Cache on the fly (58.2% VRAM saved)
compressed_kv = engine.compress_torch_cache(
    past_key_values, _inp_ids=input_ids, _mdl=model, _q_len=48
)

# 2B. Compress Vector DB / RAG Document & Query Embeddings (1536-D -> 656-D, 57.3% RAM saved)
compressed_docs    = engine.fit_compress_embeddings(doc_embeddings_1536d)
compressed_queries = engine.compress_query_embeddings(query_embeddings_1536d)
```

### Step 3: Or Run the Full Benchmark Suite
```bash
python quickstart_demo.py
```

---

## Annual Licensing, Early-Bird Pricing & Seat Management

HyperCache AI is distributed as an **AES-256-GCM authenticated commercial vault** (`hypercache-ai`) that decrypts strictly in volatile RAM upon license verification.

| Feature | Free Evaluation Trial | Developer / Startup PRO | Enterprise Cluster PRO | Enterprise Unlimited (K8s) |
| :--- | :---: | :---: | :---: | :---: |
| **Annual Price (Early-Bird Launch)** | **Free** | ~~$199/yr~~ **$99 / year** | ~~$999/yr~~ **$399 / year** | ~~$2,999/yr~~ **$1,499 / year** |
| **Purchase Instant License Key** | `HYPERCACHE-FREE-TRIAL-2026` | **[Buy Developer PRO ($99/yr)](https://kotbster7.gumroad.com/l/hypercache-ai)** | **[Buy Cluster PRO ($399/yr)](https://kotbster7.gumroad.com/l/hypercache-ai)** | **[Buy Unlimited K8s ($1,499/yr)](https://kotbster7.gumroad.com/l/hypercache-ai)** |
| **Active Machine Seats** | 1 Machine (Non-Transferable) | **2 Transferable Seats** *(Dev Laptop + 1 Prod Server)* | **10 Transferable Server Seats** | **Unlimited Nodes / Containers** *(Kubernetes Auto-Scaling)* |
| **Machine Transfer (`deactivate`)** | No | **Yes (`deactivate_current_machine()`)** | **Yes (`deactivate_current_machine()`)** | **Automatic (Zero Seat Lock)** |
| **Total Runs Allowed** | **15 Runs** (Hardware-Locked) | **Unlimited (24/7 Production)** | **Unlimited (24/7 Production)** | **Unlimited (24/7 Production)** |
| **Max Context / Vector DB Size** | Up to **4,096 Tokens / Dims** | **Unlimited (32k–1M+ Tokens)** | **Unlimited (32k–1M+ Tokens)** | **Unlimited (Cluster-Wide)** |

👉 **[Click Here to Purchase Your Instant Annual PRO License Key on Gumroad](https://kotbster7.gumroad.com/l/hypercache-ai)**

### Moving Your PRO License Between Machines
Both **Developer PRO (2 Seats)** and **Enterprise Cluster PRO (10 Seats)** allow you to transfer active seats between workstations and cloud servers at any time with a single line of Python:
```python
# Frees the current machine seat immediately so you can activate on a new server:
print(engine.deactivate_current_machine())
```
*(For dynamic Kubernetes / Docker auto-scaling environments where pods spin up and down continuously, choose **Enterprise Unlimited (K8s)** for zero seat-tracking overhead).*
