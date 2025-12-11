# Code Setup & Configuration

Complete setup instructions, GPU requirements, and how to run the pipeline with detailed parameter documentation.

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Hugging Face API key:**
Set `HF_API_KEY` (used by vLLM and tokenizers):

```powershell
# PowerShell
$env:HF_API_KEY = 'your_hf_api_key_here'

# Or create .env file
# HF_API_KEY=your_hf_api_key_here
```

---

## GPU & Hardware Requirements

The project uses `vllm` for efficient LLM inference. Models run **sequentially** (not in parallel), so you only need enough VRAM for the largest model.

**Memory per model (loaded individually):**
- **LLaMA 3.1-8B**: ~16–18 GB VRAM
- **Mistral 7B**: ~14–16 GB VRAM
- **Qwen 7B**: ~14–16 GB VRAM

**Minimum GPU needed:** **18 GB** (for LLaMA 3.1-8B, the largest model)  
**Recommended:** **20–24 GB** (for headroom with batching & vLLM's KV cache)

**Suitable GPUs:**
- ✓ NVIDIA A100 (40GB+), A40 (48GB), RTX 4090 (24GB), RTX 6000 Ada (48GB)
- ⚠ Smaller GPUs (RTX 3090, 12–20GB) may hit OOM with vLLM overhead

**vLLM Configuration:**
- Current: `max_model_len=2048`, `tensor_parallel_size=1` (single GPU, no sharding)
- For multi-GPU: increase `tensor_parallel_size` to shard a single model across GPUs
- Models still run sequentially; multi-GPU is for larger models or higher throughput

---

## How to Run

**Basic run (evaluation only, default):**
```powershell
python main.py
```

**Full pipeline (with flags):**
```powershell
python -c "from main import main; main(paraphrasing_force=True, nl2sql_force=True, evaluate=True, run_mistral=True)"
```

**Pipeline parameters:**
- `dataset_force` — Regenerate dataset (`data/interim/generated_queries.csv`)
- `paraphrasing_force` — Re-run paraphrasing (LLaMA only) → `data/processed/output_paraphrased.csv`
- `nl2sql_force` — Run NL→SQL generation for selected models
- `evaluate` — Merge results → `result/results.csv` and `result/structured_result.json`
- `run_llama`, `run_qwen`, `run_mistral` — Which models to run (independent)
- `threshold` — Paraphrase acceptance score (0.0–1.0), default 0.7
- `max_retries` — Retry paraphrasing if score < threshold, default 1

**Pipeline stages (all optional):**
1. Dataset prep (`dataset_force`)
2. Paraphrasing (`paraphrasing_force`) — **LLaMA only**
3. NL→SQL generation (`nl2sql_force`) — All selected models
4. Evaluation (`evaluate`) — Merge & aggregate

**Output files:**
- `result/llama_results.csv`, `result/qwen_results.csv`, `result/mistral_results.csv` (per-model)
- `result/results.csv` (merged results)
- `result/structured_result.json` (structured output)
- `logs/main.log`, `logs/llama.log`, `logs/qwen.log`, `logs/mistral.log`

---

## Important Notes

- **Paraphrasing is LLaMA-only:** Only `models/llama.py` implements `paraphrase_sentence()`. Paraphrasing always uses LLaMA, regardless of NL→SQL model selection.
- **Mistral & Qwen:** Used for NL→SQL generation only (no paraphrasing function).
- **`regenerate_paraphrase()` is a stub:** Currently returns original question; can be enhanced with retry logic.
- **Trust remote code:** Models use `trust_remote_code=True`; audit if needed.
- **Database structure:** Ensure `data/database/` contains subdirectories with `*.sqlite` files for each database.

---

## Data

The project uses the [Spider dataset](https://yale-lily.github.io/spider). Expected structure:
```
data/database/
├── academic/academic.sqlite
├── airline/airline.sqlite
└── ... (100+ databases)
```

---

For detailed function documentation, see [`FUNCTIONS_README.md`](FUNCTIONS_README.md) and [`UTILS_README.md`](UTILS_README.md).
