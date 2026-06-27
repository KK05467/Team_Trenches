<div align="center">
  
# 🧠 DeepThinker Multi-Agent Hub

### A Fully Local Multi-Agent AI System with Dual Sandbox Verification & Dynamic Hardware Scaling

*Running an orchestrated fleet of specialized LLMs on any hardware — from Intel iGPUs to NVIDIA H100s*

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18.0%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Architecture Multi-Agent MoE](https://img.shields.io/badge/Architecture-Multi--Agent%20MoE-purple)
![Hardware Adaptive](https://img.shields.io/badge/Hardware-Adaptive-orange)

</div>

---

DeepThinker is a **production-grade, fully local multi-agent AI system** that runs multiple heavyweight LLMs on consumer hardware — from Intel Iris Xe integrated GPUs to NVIDIA data center cards. It routes every query through a dynamic pipeline of specialized agents that reason, verify, code, fix, search the web, and generate interactive 3D visualizations — **all without a single API call to the cloud.**

The system auto-detects your hardware at startup and dynamically scales **context windows, batch sizes, scraping depth, memory thresholds, and model configurations** to extract maximum performance from whatever GPU/CPU you have.

> [!CAUTION]
> ### ⚠️ Experimental Status
> This project was developed as a submission for the **India Agentic AI Open Hackathon 2026**. The dual-sandbox architecture, Reflexion loops, GPU-aware scaling, and Dynamic Memory Allocator push consumer hardware to its absolute limits.

---

## ✨ What Makes This Different

Most "local AI" projects run a single model and call it agentic. DeepThinker runs an **orchestrated pipeline of 5 specialized models** with a loop-based Actor-Critic architecture, a live polyglot code execution sandbox, real-time web search with deep scraping, and a 3D visualization engine — all adapting to your hardware.

| Feature | DeepThinker | Typical Local LLM App |
|---|---|---|
| Number of models | 5 (dynamically routed V2) | 1 |
| Code execution | ✅ Isolated polyglot sandbox (9 languages) | ❌ |
| Self-correction | ✅ Reflexion + Nuclear Reset loops | ❌ |
| Web search | ✅ 4 modes: Off, Simple, Prediction, Extreme | ❌ or cloud API |
| UI Artifacts | ✅ Claude-style Frontend Sandboxing | ❌ |
| Memory / RAG | ✅ ChromaDB + SQLite vector store | ❌ |
| 3D Visualization | ✅ Three.js / Plotly.js auto-generation | ❌ |
| Hardware scaling | ✅ Auto-adapts to GPU compute capability | Usually NVIDIA only |
| Cloud dependency | ❌ 100% local | Often cloud-backed |

---

## 🤖 The Agent Pipeline (6-Way Architecture)

```
                       User Prompt + Uploaded Image
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │ Qwen-2.5-VL Vision Parser │ (Extract text/diagrams)
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    Phi-3.5-Mini Router    │ (Intent & Mode Classifier)
                      └─────────────┬─────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬────────────────┬──────────────┐
    ▼               ▼               ▼               ▼                ▼              ▼
  [PATH A]       [PATH B]        [PATH C]        [PATH D]         [PATH E]       [PATH F]
   SIMPLE         CODING        REASONING       PREDICTION      EXTREME SEARCH    3D VIZ
    │               │               │               │                │              │
    ▼               ▼               ▼               ▼                ▼              ▼
Phi-3.5-Mini   OpenCodeDS      VibeThinker 3B  OpenCodeDS       DeepSeek R1    OpenCodeDS
 Direct Ans    Actor-Critic     Reasoning Plan  ML Regression   Deep Analysis   WebGL & JS
    │               │               │               │                │              │
    ▼               ▼               ▼               ▼                ▼              ▼
Web Search    Execution       Sandbox         Data Cleaning    OpenCodeDS     Reflexion
 (Simple)     Sandbox         Verification     Loop (SciPy)    Plotly Charts  Sandbox Loop
    │               │               │               │                │              │
    └───────────────┴───────────────┴───────────────┴────────────────┴──────────────┘
                                    │
                                    ▼
                             Streamed Response
                         (Answer + Code + 3D Visual)
```

---

## 🧩 The Five Agents

| Agent | Model | Role | VRAM |
|---|---|---|---|
| **Master Router** | Phi-3.5-Mini-Instruct Q6_K | Intent classification, search query optimization, prompt compression, 3D gate checks | ~2.8 GB |
| **Math/Logic Reasoning** | WeiboAI VibeThinker 3B Q6_K | Curricular reinforcement learning-trained reasoning engine for mathematics, physics & logic | ~2.5 GB |
| **Reasoning Engine** | DeepSeek-R1-Distill-Qwen-7B Q6_K | Chain-of-thought logic, math proofs, code writing, deep analysis, Actor-Critic debate | ~6.0 GB |
| **3D Visualization** | OpenCodeInterpreter-DS-6.7B Q6_K | Plotly 3D charts, Three.js simulations, interactive HTML artifact generation, ML scripting | ~5.0 GB |
| **Vision Parser** | Qwen2.5-VL-7B-Instruct Q6_K_XL | Multimodal image/document OCR, circuit diagrams, handwritten equations | ~7.0 GB |

All models are loaded on-demand (lazy loading) and managed by the Dynamic Memory Allocator. Only one model occupies VRAM at a time in EVM mode.

---

## 🛡️ Core Systems

### 1 — Dynamic Memory Allocator (DMA)

A custom memory manager built for running multiple large models on consumer hardware:

- **Auto-Calibration:** Detects total RAM and VRAM at startup, reserves 25% of RAM and 40% of VRAM as safety buffers
- **LRU Eviction:** When memory is tight, evicts the least recently used model first — one at a time, until room is available
- **iGPU Unified Memory Guard:** Prevents `corrupted size vs. prev_size` glibc heap crashes on Intel Iris Xe by blocking simultaneous large allocations
- **Lazy Loading:** Models load only when the pipeline needs them, never all at once
- **Multi-Backend VRAM Detection:** Reads VRAM from `torch.cuda` (NVIDIA), `torch.xpu` (Intel), and Linux sysfs `/sys/class/drm/` (AMD/Intel without ROCm)
- **Automatic CPU Fallback:** If GPU context creation fails, transparently retries with `n_gpu_layers=0` to keep the pipeline alive

### 2 — GPU-Aware Dynamic Scaling

The entire system auto-scales to the detected GPU compute capability:

| Hardware | Compute Cap | Context Ceiling | Scrape Depth | Batch Size |
|---|---|---|---|---|
| **H100/B200** (80 GB) | SM ≥ 9.0 | **65,536 tokens** | 8,000 chars/page | Default (2048) |
| **A100/RTX 4090** (40-48 GB) | SM ≥ 8.0 | **32,768 tokens** | 8,000 chars/page | Default (2048) |
| **P100/T4/V100** (16 GB) | SM < 8.0 | **8,192 tokens** | ~2,949 chars/page | 512 (n_ubatch=256) |
| **Intel Iris Xe iGPU** | N/A | **16,384 tokens** | ~5,898 chars/page | Default |
| **CPU only** | N/A | **16,384 tokens** | ~5,898 chars/page | Default |

This scaling chain flows through every subsystem: **GPU Compute Capability → Context Ceiling → Web Scrape char_limit → Prompt Budget → Semantic Cruncher thresholds → Generation token allocation.**

On older GPUs (SM < 8.0), the system also:
- Restricts llama.cpp `n_batch` to 512 and `n_ubatch` to 256 to prevent quadratic self-attention VRAM spikes
- Explicitly disables `flash_attn` to avoid buggy fallback paths

### 3 — Enterprise VRAM Multiplexing (EVM)

A hardware-aware hot-swap technology for environments where System RAM far exceeds GPU VRAM (e.g., Kaggle P100 with 32 GB RAM + 16 GB VRAM):

**Auto-Activation Criteria:**
- System RAM ≥ 24 GB
- GPU VRAM ≤ 16 GB (discrete NVIDIA/CUDA GPU)

When active, EVM:
- Loads all models into the OS page cache (System RAM)
- Aggressively hot-swaps models one-by-one into GPU VRAM via PCIe
- Evicts dormant agents to guarantee the active model gets **100% of VRAM** for its KV cache
- Reduces RAM safety threshold to 5% (from 25%) and VRAM safety to 5% (from 40%)

The **Load All Models** sidebar button pre-loads all models into the OS page cache, reducing subsequent hot-swaps to **1-2 seconds** instead of minutes.

### 4 — Polyglot Execution Sandbox

A three-layer isolated code execution environment supporting **9 programming languages:**

**Supported Languages:** Python, C, C++, Java, JavaScript, Bash, Go, Rust, TypeScript

**Three Layers of Protection:**
1. **Process Isolation:** Code runs in a subprocess — crashes can't kill the server
2. **Restricted Builtins:** Custom `__import__` whitelist blocks `open`, `exec`, `eval`, `os`, `subprocess`, `socket`
3. **Linux Resource Limits:** 2 GB RAM cap, 300s CPU limit, 200 child process limit, 100 MB file write limit

**Additional Features:**
- **Auto-pip Package Installer:** Detects `ModuleNotFoundError`, auto-installs the missing pip package, and retries execution
- **Unrestricted Fallback:** If the restricted sandbox blocks a legitimate library, falls back to unrestricted execution with a `⚠️ [Unrestricted Fallback]` prefix
- **GUI Detection:** Identifies `pygame`, `tkinter`, `turtle`, etc. and returns a friendly message instead of hanging
- **Infinite Loop Detection:** Catches `while True`, `for(;;)`, `.mainloop()` patterns

**Whitelisted Science Libraries (Restricted Mode):**
`numpy`, `sympy`, `scipy`, `pandas`, `plotly`, `sklearn`, `statsmodels`, `pint`, `z3` (theorem prover), `networkx`, `astropy`, `Bio` (Biopython), `rdkit` (cheminformatics), `rocketpy`, `qiskit`, `qutip`, `cryptography`, `scapy`, `requests`, `urllib`

### 5 — Reflexion Self-Fix Loop

When generated code fails, the system doesn't just report the error — it fixes it:

1. **Shallow Fix:** Attempt a targeted correction based on the error traceback
2. **Deep Escalation:** Rewrite the entire script with a stronger prompt including the failure context
3. **Nuclear Reset:** Extract lessons from all failures and start over from scratch with a new plan
4. Up to **2 reflexion loops** and **1 Nuclear Reset cycle** before returning best-effort output

### 6 — Long-Term RAG Memory

- **ChromaDB** vector database for semantic similarity search (primary)
- **SQLite** with cosine similarity and keyword search as fallback
- Stores **compact solution summaries** — not raw code dumps — to avoid context bloat
- Stores **mistake-fix patterns** from the Reflexion loop to prevent regressing on similar tasks
- **Smart Deduplication:** Uses NLP stopword filtering and punctuation stripping to merge near-identical tasks (>80% content word overlap)
- **Concurrency-safe:** Dynamic connection pooling with 30-second lock timeouts for SQLite

### 7 — Web Search Integration

Real-time web search with deep scraping and intelligent deduplication:

**Search Provider Chain:** Google Custom Search API → SearXNG (4 instance fallback) → DuckDuckGo Library → DuckDuckGo HTML Scraper

**Search Modes:**
*   **Off:** Direct response generation.
*   **🌐 Search (Simple):** Scrapes 3-5 pages with precise query optimization.
*   **🔮 Predict (Prediction):** Decoupled ML-data pipeline scraping up to 20 sources, extracting dataset columns, handling NaNs, dates, and computing regression metrics via scikit-learn in the sandbox.
*   **🔬 Extreme (Extreme WebSearch):** Ingests massive data chunks and routes to DeepSeek R1-7B for comparative analysis before OpenCodeInterpreter builds Plotly visualizations.

---

## 📁 Project Structure

```
Team_Trenches/
├── backend/
│   ├── app.py            # FastAPI server, REST endpoints, SSE streaming, cancel/offload controls
│   ├── orchestrator.py   # Core multi-agent pipeline, DMA, GPU-aware scaling, Reflexion loops
│   ├── sandbox.py        # Polyglot code execution sandbox (9 languages), auto-pip installer
│   ├── memory.py         # Long-term RAG memory (ChromaDB + SQLite), deduplication engine
│   ├── search.py         # Web search (Google → SearXNG → DuckDuckGo), deep scraper
│   └── downloader.py     # Model downloader with HuggingFace Hub, retry logic, shard support
├── frontend/
│   └── src/
│       └── App.jsx       # React UI: live streaming, 3D artifacts, glassmorphic design
├── models/               # Downloaded GGUF weights (git-ignored, ~20.5 GB total)
│   ├── text/
│   │   ├── router/       # Phi-3.5-Mini-Instruct Q6_K (~2.8 GB)
│   │   ├── vibethinker/  # WeiboAI VibeThinker 3B Q6_K (~2.5 GB)
│   │   ├── deepseek_r1/  # DeepSeek-R1-Distill-Qwen-7B Q6_K (~6.0 GB)
│   │   └── opencode/     # OpenCodeInterpreter-DS-6.7B Q6_K (~5.0 GB)
│   └── image_to_text/
│       └── qwen_vl/      # Qwen2.5-VL-7B-Instruct Q6_K_XL + mmproj (~8.3 GB)
├── start.sh              # One-command Linux launcher (backend + frontend)
├── STARTUP.md            # Cross-platform startup guide
├── README_SETUP.md       # Hardware acceleration installation guide
└── requirements.txt      # Python dependencies
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | `POST` | Main chat endpoint — streams SSE status updates + final response |
| `/api/status` | `GET` | System health, model status, RAM/CPU/GPU usage, EVM state |
| `/api/download/{model_key}` | `POST` | Trigger background model download from HuggingFace |
| `/api/cancel` | `POST` | Stop active generation (models stay loaded in memory) |
| `/api/offload` | `POST` | Unload all models from RAM/VRAM |
| `/api/load_all` | `POST` | Pre-load all downloaded models into RAM page cache (EVM only) |
| `/api/settings` | `POST` | Update context length, temperature, GPU layers, web search mode |
| `/api/memory/count` | `GET` | Number of stored long-term memories |
| `/api/memory/clear` | `POST` | Reset the vector database and memory store |
| `/api/unload` | `POST` | Unload all models to free system RAM |

---

## 🆕 Pipeline V2 Optimizations (June 2026)

- **VibeThinker 3B Integration:** WeiboAI's reasoning-focused model registered as the dedicated engine for competitive math and STEM logic.
- **Tri-State Web Search Mode Selector:** Replacing the binary switch with Off / Search / Predict / Extreme modes in both frontend and backend.
- **Dedicated ML Prediction Pipeline:** Python Scikit-Learn validation loops with NaN correction, date parsers, and custom `PREDICTIVE_METRICS` outputs.
- **Extreme WebSearch Pipeline:** Direct ingestion of up to 20 web documents with DeepSeek R1 reasoning followed by automated Plotly chart generation.
- **Full Qwen-2.5-VL 7B Multimodal Pipeline:** Verified vision OCR transcribing handwritten text, circuit diagrams, and math formulas directly into the query context.

---

## 👥 Team

**Team Trenches** — India Agentic AI Open Hackathon 2026

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
