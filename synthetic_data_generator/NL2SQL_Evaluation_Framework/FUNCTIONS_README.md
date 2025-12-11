# Functions Reference

This document describes all functions in the main orchestrator (`main.py`) and model modules (`models/llama.py`, `models/qwen.py`, `models/mistral.py`). Each section covers function signatures, parameters, return values, and purpose.

**For utility functions** (database operations, logging, schema extraction, SQL comparison), see [`UTILS_README.md`](UTILS_README.md).

---

## `main.py`

### `ensure_dirs(*paths: Path)`
**Purpose:** Create one or more directories recursively (like `mkdir -p`).

**Parameters:**
- `*paths: Path` — Variable number of `pathlib.Path` objects to create.

**Returns:** None

**Example:**
```python
ensure_dirs(Path("data/interim"), Path("result"), Path("logs"))
```

---

### `main(dataset_force=False, paraphrasing_force=False, nl2sql_force=True, evaluate=False, run_llama=False, run_qwen=False, run_mistral=True, threshold=0.7, max_retries=1)`

**Purpose:** Orchestrator function. Controls the entire pipeline: dataset prep → paraphrasing (LLaMA only) → NL→SQL generation (per selected model) → evaluation & aggregation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_force` | bool | `False` | If `True`, regenerate the interim dataset (`data/interim/generated_queries.csv`). |
| `paraphrasing_force` | bool | `False` | If `True`, re-run paraphrasing (LLaMA only) and overwrite `data/processed/output_paraphrased.csv`. |
| `nl2sql_force` | bool | `True` | If `True`, run the NL→SQL generation step for all selected models (LLaMA, Qwen, Mistral). |
| `evaluate` | bool | `False` | If `True`, merge per-model results and create aggregated `results.csv` and `structured_result.json`. |
| `run_llama` | bool | `False` | If `True`, run NL→SQL generation using the LLaMA model. |
| `run_qwen` | bool | `False` | If `True`, run NL→SQL generation using the Qwen model. |
| `run_mistral` | bool | `True` | If `True`, run NL→SQL generation using the Mistral model. All three models are independent. |
| `threshold` | float | `0.7` | Paraphrase acceptance score threshold. If `score < threshold`, retry paraphrasing up to `max_retries` times. |
| `max_retries` | int | `1` | Number of retry attempts if paraphrase score falls below `threshold`. |

**Returns:** None

**Flow:**
1. Calls `src.prepare_dataset.main()` if dataset doesn't exist or `dataset_force=True`.
2. **Paraphrasing (LLaMA only)**: Iterates through dataset, calls `paraphrase_sentence()` (from `models/llama.py`) for each row, scores with `score_paraphrase()`, retries if low score using `regenerate_paraphrase()`.
3. Saves paraphrased output to `data/processed/output_paraphrased.csv`.
4. **NL→SQL (all selected models)**: For each selected model (`run_llama`, `run_qwen`, `run_mistral`), calls `generate_sql_from_dataframe()` which generates SQL from both original and paraphrased questions and writes per-model CSVs.
5. If `evaluate=True`, merges per-model results into a single DataFrame and JSON output.

**Outputs:**
- `data/interim/generated_queries.csv` (if dataset generation)
- `data/processed/output_paraphrased.csv` (paraphrased questions + scores)
- `result/llama_results.csv`, `result/qwen_results.csv`, `result/mistral_results.csv` (per-model)
- `result/results.csv` (merged results)
- `result/structured_result.json` (structured output with model correctness flags)
- `logs/main.log`, `logs/llama.log`, `logs/qwen.log`, `logs/mistral.log`

---

## `models/llama.py`

### `get_tokenizer() -> AutoTokenizer`
**Purpose:** Returns a singleton tokenizer instance for the LLaMA model. Lazy-loads on first call.

**Parameters:** None

**Returns:** `transformers.AutoTokenizer` instance for `meta-llama/Meta-Llama-3.1-8B-Instruct`.

**Global state:** Sets `_tokenizer_nl2nl` on first call; subsequent calls return the same instance.

**Example:**
```python
tok = get_tokenizer()
chat_template = tok.apply_chat_template([...], tokenize=False)
```

---

### `get_llm_nl2nl() -> LLM`
**Purpose:** Returns a singleton `vllm.LLM` instance for the LLaMA model. Lazy-loads on first call.

**Parameters:** None

**Returns:** `vllm.LLM` instance configured for LLaMA-3.1-8B-Instruct with `max_model_len=2048`, `tensor_parallel_size=1`.

**Global state:** Sets `_llm_nl2nl` on first call; subsequent calls return the same instance.

**Example:**
```python
llm = get_llm_nl2nl()
outputs = llm.generate(chat_input, sampling_params=_sampling_nl2nl)
```

---

### `paraphrase_sentence(sentence: str, schema: str = "") -> str`
**Purpose:** Paraphrases a natural-language question while preserving SQL semantics.

**Parameters:**
- `sentence: str` — Original NL question to paraphrase.
- `schema: str` — (optional) Database schema for context. Default is empty string.

**Returns:** `str` — Paraphrased question. Falls back to original if an error occurs.

**Details:**
- Uses `PARAPHRASE_SYSTEM_PROMPT` to guide the model.
- Constructs a chat message with system prompt + user query (including schema).
- Calls `llm.generate()` with `_sampling_nl2nl` (temperature=0, max_tokens=100).
- On exception, logs error and returns the original sentence.

**Example:**
```python
from pathlib import Path
from models.llama import paraphrase_sentence
from src.utils.sql_utils import extract_schema

db_path = Path("data/database/academic/academic.sqlite")
schema = extract_schema(db_path)
question = "How many authors are there?"
paraphrased = paraphrase_sentence(question, schema)
print(paraphrased)
```

---

### `generate_sql(nl_question: str, schema: str) -> str`
**Purpose:** Generates a SQL query from a natural-language question and database schema.

**Parameters:**
- `nl_question: str` — Natural-language question.
- `schema: str` — Extracted database schema as text.

**Returns:** `str` — Generated SQL query as plain text. Empty string on error.

**Details:**
- Uses `SQL_GEN_SYSTEM_PROMPT` to instruct the model.
- Sends a chat message: system prompt + user query (including schema and question).
- Calls `llm.generate()` with `_sampling_nl2nl`.
- If output is JSON-formatted (starts with `[`, ends with `]`), parses and extracts first query; otherwise returns raw output.
- On exception, logs error and returns empty string.

**Example:**
```python
from pathlib import Path
from models.llama import generate_sql
from src.utils.sql_utils import extract_schema

db_path = Path("data/database/academic/academic.sqlite")
schema = extract_schema(db_path)
question = "How many authors are there?"
sql = generate_sql(question, schema)
print(sql)
```

---

### `regenerate_paraphrase(question: str, schema: str) -> str`
**Purpose:** (Currently a stub) Intended to regenerate paraphrase if score is too low.

**Parameters:**
- `question: str` — Original question.
- `schema: str` — Database schema.

**Returns:** `str` — Currently returns the original question unchanged.

**Note:** This is a placeholder and should be implemented to retry paraphrasing or use a different prompt.

---

### `generate_sql_from_dataframe(paraphrased_df: pd.DataFrame, database_path: Path, *, logger, result_path: Optional[Path] = None, store_sql: bool = True, checkpoint_every: Optional[int] = None) -> pd.DataFrame`

**Purpose:** Batch process a DataFrame: generate SQL for original & paraphrased questions, compare against ground-truth, and save results.

**Parameters:**
- `paraphrased_df: pd.DataFrame` — Input DataFrame with columns: `db_name`, `natural_language`, `sql_query`, `paraphrased_nl`.
- `database_path: Path` — Root path to database folders (e.g., `data/database/`).
- `logger` — Logger instance (keyword-only, required).
- `result_path: Optional[Path]` — Directory to save output CSV and checkpoints. Default: `None`.
- `store_sql: bool` — If `True`, include generated SQL queries in output columns. Default: `True`.
- `checkpoint_every: Optional[int]` — Save intermediate checkpoint CSV every N rows. Default: `None` (no checkpoints).

**Returns:** `pd.DataFrame` — Processed DataFrame with columns:
- `row_id` — Stable row identifier (added if missing).
- `db_name`, `natural_language`, `sql_query`, `paraphrased_nl`, `paraphrased_score`.
- `llama_para_correct` — Boolean; true if SQL from paraphrased question matches ground truth.
- `llama_original_correct` — Boolean; true if SQL from original question matches ground truth.
- (optional) `llama_query_from_para`, `llama_query_from_original` — Generated SQL strings (if `store_sql=True`).

**Details:**
- Iterates through DataFrame rows.
- For each row: extracts schema once per DB (cached), calls `generate_sql()` for both original and paraphrased questions, compares with `compare_sql()`.
- Logs progress every 50 rows.
- If `checkpoint_every` is set, saves intermediate CSV at specified intervals.
- Final DataFrame is reordered to a standard column order and saved to `result_path / "llama_results.csv"` if `result_path` is provided.

**Example:**
```python
from pathlib import Path
from models.llama import generate_sql_from_dataframe
from src.utils.logger import setup_logger
import pandas as pd

df = pd.read_csv("data/processed/output_paraphrased.csv")
result_df = generate_sql_from_dataframe(
    paraphrased_df=df,
    database_path=Path("data/database"),
    logger=setup_logger("llama_logger", Path("logs/llama.log")),
    result_path=Path("result"),
    store_sql=True,
    checkpoint_every=50
)
print(result_df.head())
```

---

## `models/qwen.py`

### `get_tokenizer() -> AutoTokenizer`
**Purpose:** Returns a singleton tokenizer instance for the Qwen model. Lazy-loads on first call.

**Parameters:** None

**Returns:** `transformers.AutoTokenizer` instance for `XGenerationLab/XiYanSQL-QwenCoder-7B-2504`.

**Global state:** Sets `_tokenizer_nl2nl` on first call; subsequent calls return the same instance.

---

### `get_llm_nl2nl() -> LLM`
**Purpose:** Returns a singleton `vllm.LLM` instance for the Qwen model. Lazy-loads on first call.

**Parameters:** None

**Returns:** `vllm.LLM` instance configured for XiYanSQL-QwenCoder-7B-2504 with `max_model_len=2048`, `tensor_parallel_size=1`.

**Global state:** Sets `_llm_nl2nl` on first call; subsequent calls return the same instance.

---

### `paraphrase_sentence(sentence: str, schema: str = "") -> str`
**Purpose:** Paraphrases a natural-language question while preserving SQL semantics (same as LLaMA version).

**Parameters:**
- `sentence: str` — Original NL question to paraphrase.
- `schema: str` — (optional) Database schema for context. Default is empty string.

**Returns:** `str` — Paraphrased question. Falls back to original if an error occurs.

**Details:**
- Uses `PARAPHRASE_SYSTEM_PROMPT`.
- Constructs chat message with system prompt + user query (including schema).
- Calls `llm.generate()` with `_sampling_nl2nl`.
- On exception, logs error and returns the original sentence.

---

### `generate_sql(nl_question: str, schema: str) -> str`
**Purpose:** Generates a SQL query from a natural-language question and database schema (same as LLaMA version).

**Parameters:**
- `nl_question: str` — Natural-language question.
- `schema: str` — Extracted database schema as text.

**Returns:** `str` — Generated SQL query as plain text. Empty string on error.

**Details:**
- Uses `SQL_GEN_SYSTEM_PROMPT`.
- Calls `llm.generate()` with `_sampling_nl2nl`.
- Parses JSON-formatted output if applicable; otherwise returns raw output.
- On exception, logs error and returns empty string.

---

### `generate_sql_from_dataframe(paraphrased_df: pd.DataFrame, database_path: Path, *, logger, result_path: Optional[Path] = None, store_sql: bool = True, checkpoint_every: Optional[int] = None) -> pd.DataFrame`

**Purpose:** Batch process a DataFrame: generate SQL for original & paraphrased questions, compare against ground-truth, and save results.

**Parameters:** (Same as `models/llama.py`)
- `paraphrased_df: pd.DataFrame` — Input DataFrame.
- `database_path: Path` — Root path to database folders.
- `logger` — Logger instance (keyword-only).
- `result_path: Optional[Path]` — Output directory. Default: `None`.
- `store_sql: bool` — Include generated SQL in output. Default: `True`.
- `checkpoint_every: Optional[int]` — Checkpoint interval. Default: `None`.

**Returns:** `pd.DataFrame` — Processed DataFrame with Qwen-specific columns:
- `qwen_para_correct` — Boolean; true if SQL from paraphrased question matches ground truth.
- `qwen_original_correct` — Boolean; true if SQL from original question matches ground truth.
- (optional) `qwen_query_from_para`, `qwen_query_from_original` — Generated SQL strings.

**Details:**
- Same flow as LLaMA, but with Qwen-prefixed column names.
- Saves to `result_path / "qwen_results.csv"`.

---

## `models/mistral.py`

### `get_tokenizer() -> AutoTokenizer`
**Purpose:** Returns a singleton tokenizer instance for the Mistral model. Lazy-loads on first call.

**Parameters:** None

**Returns:** `transformers.AutoTokenizer` instance for `mistralai/Mistral-7B-Instruct-v0.3`.

**Global state:** Sets `_tokenizer_nl2nl` on first call; subsequent calls return the same instance.

---

### `get_llm_nl2nl() -> LLM`
**Purpose:** Returns a singleton `vllm.LLM` instance for the Mistral model. Lazy-loads on first call.

**Parameters:** None

**Returns:** `vllm.LLM` instance configured for Mistral-7B-Instruct-v0.3 with `max_model_len=2048`, `tensor_parallel_size=1`.

**Global state:** Sets `_llm_nl2nl` on first call; subsequent calls return the same instance.

---

### `generate_sql(nl_question: str, schema: str) -> str`
**Purpose:** Generates a SQL query from a natural-language question and database schema (same as LLaMA/Qwen).

**Parameters:**
- `nl_question: str` — Natural-language question.
- `schema: str` — Extracted database schema as text.

**Returns:** `str` — Generated SQL query as plain text. Empty string on error.

**Details:**
- Uses `SQL_GEN_SYSTEM_PROMPT`.
- Calls `llm.generate()` with `_sampling_nl2nl`.
- Parses JSON-formatted output if applicable; otherwise returns raw output.
- On exception, logs error and returns empty string.

**Note:** Unlike LLaMA and Qwen, `mistral.py` does **not** provide a `paraphrase_sentence()` function. Paraphrasing is handled by the LLaMA or Qwen models.

---

### `generate_sql_from_dataframe(paraphrased_df: pd.DataFrame, database_path: Path, *, logger, result_path: Optional[Path] = None, store_sql: bool = True, checkpoint_every: Optional[int] = None) -> pd.DataFrame`

**Purpose:** Batch process a DataFrame: generate SQL for original & paraphrased questions, compare against ground-truth, and save results.

**Parameters:** (Same as other models)
- `paraphrased_df: pd.DataFrame` — Input DataFrame.
- `database_path: Path` — Root path to database folders.
- `logger` — Logger instance (keyword-only).
- `result_path: Optional[Path]` — Output directory. Default: `None`.
- `store_sql: bool` — Include generated SQL in output. Default: `True`.
- `checkpoint_every: Optional[int]` — Checkpoint interval. Default: `None`.

**Returns:** `pd.DataFrame` — Processed DataFrame with Mistral-specific columns:
- `mistral_para_correct` — Boolean; true if SQL from paraphrased question matches ground truth.
- `mistral_original_correct` — Boolean; true if SQL from original question matches ground truth.
- (optional) `mistral_query_from_para`, `mistral_query_from_original` — Generated SQL strings.

**Details:**
- Same flow as LLaMA/Qwen, but with Mistral-prefixed column names.
- Saves to `result_path / "mistral_results.csv"`.

---

## Usage Summary

### Typical workflow:

1. **Import & configure:**
   ```python
   from main import main
   from src.utils.logger import setup_logger
   from src.utils.sql_utils import extract_schema
   ```

2. **Run the pipeline:**
   ```python
   main(
       dataset_force=False,
       paraphrasing_force=False,
       nl2sql_force=True,
       evaluate=True,
       run_llama=False,
       run_qwen=False,
       run_mistral=True,
       threshold=0.7,
       max_retries=1
   )
   ```

3. **Or, run from command line:**
   ```powershell
   python main.py
   ```

4. **Or, use a specific model directly (with automatic schema extraction):**
   ```python
   from pathlib import Path
   from models.llama import generate_sql
   from src.utils.sql_utils import extract_schema
   
   db_path = Path("data/database/academic/academic.sqlite")
   schema = extract_schema(db_path)  # Extracts schema automatically
   question = "How many authors are there?"
   sql = generate_sql(question, schema)
   print(sql)
   ```

---

## Notes

- All model tokenizers and LLM instances are **singletons** (lazy-loaded), so they are created once and reused across calls.
- **Sampling parameters** are fixed: `temperature=0, max_tokens=100` for deterministic SQL generation.
- **Error handling:** All functions catch exceptions and return graceful fallbacks (original text or empty string).
- **Logging:** Comprehensive logging via `setup_logger()` for debugging and monitoring progress.
- **Schema caching:** `generate_sql_from_dataframe()` caches schemas per DB to avoid repeated I/O.
- **Utilities:** For database operations (`extract_schema()`, `compare_sql()`, etc.) and logging, see [`UTILS_README.md`](UTILS_README.md).

---

File created by the project helper.
