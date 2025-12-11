# ParaphrasingProject

A pipeline to evaluate paraphrasing quality for NL→SQL tasks. The code paraphrases natural-language questions, generates SQL from both original and paraphrased questions using multiple LLM backends (LLaMA, Mistral, Qwen), compares generated SQL against ground-truth, and aggregates results.

## Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Set Hugging Face API key:**
```powershell
$env:HF_API_KEY = 'your_hf_api_key_here'
# or create .env file: HF_API_KEY=your_key
```

**3. Run:**
```powershell
python main.py
```

---

## Setup & Configuration

### GPU Requirements

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

### How to Run

**Basic run (evaluation only, default):**
```powershell
python main.py
```

**Full pipeline (with flags):**
```powershell
python -c "from main import main; main(paraphrasing_force=True, nl2sql_force=True, evaluate=True, run_mistral=True)"
```

**Pipeline parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_force` | bool | `False` | Regenerate dataset (`data/interim/generated_queries.csv`) |
| `paraphrasing_force` | bool | `False` | Re-run paraphrasing (LLaMA only) → `data/processed/output_paraphrased.csv` |
| `nl2sql_force` | bool | `True` | Run NL→SQL generation for selected models |
| `evaluate` | bool | `False` | Merge results → `result/results.csv` and `structured_result.json` |
| `run_llama` | bool | `False` | Run LLaMA NL→SQL generation |
| `run_qwen` | bool | `False` | Run Qwen NL→SQL generation |
| `run_mistral` | bool | `True` | Run Mistral NL→SQL generation |
| `threshold` | float | `0.7` | Paraphrase acceptance score (LLaMA stage only) |
| `max_retries` | int | `1` | Retry paraphrasing if score < threshold |

**Output files:**
- `result/llama_results.csv`, `result/qwen_results.csv`, `result/mistral_results.csv` (per-model)
- `result/results.csv` (merged results)
- `result/structured_result.json` (structured output)
- `logs/main.log`, `logs/llama.log`, `logs/qwen.log`, `logs/mistral.log`

### Important Notes

- **Paraphrasing is LLaMA-only:** Only `models/llama.py` implements `paraphrase_sentence()`. Paraphrasing always uses LLaMA, regardless of NL→SQL model selection.
- **Mistral & Qwen:** Used for NL→SQL generation only (no paraphrasing function).
- **`regenerate_paraphrase()` is a stub:** Currently returns original question; can be enhanced with retry logic.
- **Trust remote code:** Models use `trust_remote_code=True`; audit if needed.
- **Database structure:** Ensure `data/database/` contains subdirectories with `*.sqlite` files for each database.

---

## Documentation

## Project Structure

```
ParaphrasingProject/
├── main.py                          # Orchestrator: controls pipeline stages
├── models/
│   ├── llama.py                     # LLaMA model wrapper (paraphrasing + NL→SQL)
│   ├── mistral.py                   # Mistral model wrapper (NL→SQL only)
│   ├── qwen.py                      # Qwen model wrapper (NL→SQL only)
│   └── prompt_templates.py          # System prompts for paraphrasing & SQL generation
├── src/
│   ├── prepare_dataset.py           # Dataset preparation
│   └── utils/
│       ├── sql_utils.py             # Database operations, schema extraction, SQL comparison
│       ├── logger.py                # Logging setup
│       ├── paraphrase_score.py      # Paraphrase quality scoring
│       └── __init__.py
├── data/
│   ├── database/                    # SQLite databases (one per dataset DB)
│   ├── interim/                     # Generated queries before paraphrasing
│   └── processed/                   # Paraphrased questions + scores
├── result/                          # Model outputs, aggregated results, structured JSON
├── logs/                            # Log files per stage
├── FUNCTIONS_README.md              # Function reference
├── UTILS_README.md                  # Utility function reference
├── requirements.txt                 # Python dependencies
└── README.md                        # This file (main documentation)
```

---

## Pipeline Overview

The project runs in 4 stages (all optional):

1. **Dataset Preparation** — Generate initial dataset from source.
2. **Paraphrasing** — LLaMA generates paraphrases and scores them (`paraphrasing_force` flag).
3. **NL→SQL Generation** — All selected models (LLaMA, Mistral, Qwen) generate SQL from original and paraphrased questions (`nl2sql_force` flag).
4. **Evaluation** — Merge results from all models and create aggregated outputs (`evaluate` flag).

**Example runs:**
- Paraphrase only: `main(paraphrasing_force=True, nl2sql_force=False)`
- Run Mistral NL→SQL only: `main(nl2sql_force=True, run_mistral=True, run_llama=False, run_qwen=False)`
- Full pipeline: `main(paraphrasing_force=True, nl2sql_force=True, evaluate=True)`

See [`CODE_README.md`](CODE_README.md) for detailed parameter documentation.

---

## Models

- **LLaMA 3.1-8B-Instruct** — Used for paraphrasing + NL→SQL
- **Mistral 7B-Instruct** — Used for NL→SQL only
- **Qwen (XiYanSQL)** — Used for NL→SQL only

All models use `vllm` for efficient inference.

---

## Dataset

The project uses the [Spider dataset](https://yale-lily.github.io/spider). Expected structure:
```
data/database/
├── academic/
│   └── academic.sqlite
├── airline/
│   └── airline.sqlite
└── ... (100+ databases)
```

---

## Key Files at a Glance

| File | Role |
|------|------|
| `main.py` | Pipeline orchestrator; controls stages and model selection |
| `models/llama.py` | LLaMA wrapper (only module with paraphrasing) |
| `models/mistral.py`, `qwen.py` | NL→SQL generation (no paraphrasing) |
| `src/utils/sql_utils.py` | Database I/O, schema extraction, SQL execution & comparison |
| `result/results.csv` | Merged results from all models |
| `result/structured_result.json` | Structured output with correctness flags |

---

## Documentation Reference

- **[`FUNCTIONS_README.md`](FUNCTIONS_README.md)** — Complete function reference for `main.py` and model modules (`llama.py`, `mistral.py`, `qwen.py`). Includes function signatures, parameters, return values, and usage examples.
- **[`UTILS_README.md`](UTILS_README.md)** — Utility function documentation for `src/utils/`. Covers database operations, schema extraction, SQL comparison, logging, and paraphrase scoring.

---

