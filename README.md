# DeepThinker Multi-Agent Hub
This is a production-grade AI routing engine capable of running multiple Heavyweight LLMs (Phi-3.5, VibeThinker, DeepSeek-R1, OpenCodeInterpreter) simultaneously on an **Intel Iris Xe Integrated GPU (iGPU)**.

## 🚀 The Intel Iris Xe iGPU Fixes
Running complex AI models natively on Iris Xe historically causes severe driver crashes. The following stability fixes have been permanently applied:

### 1. The `Error 45` Level Zero Workaround
When PyTorch attempts massive Attention Mechanism calculations, the Intel Level Zero API will instantly crash with `UR_RESULT_ERROR_INVALID_ARGUMENT (Error 45)`. 
**Fix:** In `backend/app.py`, we explicitly inject `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`. This forces the driver to execute memory allocations instantaneously, completely bypassing the crash.

### 2. The PyTorch Tokenizer Corruption (`untagged enum ModelWrapper`)
When loading local safetensors models like **VibeThinker 1.5B**, the Rust-based HuggingFace tokenizer can corrupt or fail to parse `tokenizer.json` files generated from older architectures. 
**Fix:** For Qwen-based models like VibeThinker, the pipeline requires replacing the corrupted `tokenizer.json` directly from the official `Qwen/Qwen2.5-1.5B-Instruct` repository.

### 3. The `UR error` (OpenCL Kernel Failure) on Transformers
Models loaded via HuggingFace `AutoModelForCausalLM` natively compile standard PyTorch operations. On an iGPU, complex `float16` sliding window attention operations will trigger a deep driver `UR error`.
**Fix:** The **Phi-3.5 Router**, **DeepSeek**, and **OpenCodeInterpreter** have been migrated entirely to the **GGUF format**. The pipeline executes them via the `llama.cpp` Vulkan Bridge, achieving ultra-high speeds while dodging the PyTorch driver completely.

## 🛠️ Usage
**Start the Backend:**
```bash
source venv/bin/activate
python backend/app.py
```
**Start the Frontend:**
```bash
cd frontend
npm run dev
```
