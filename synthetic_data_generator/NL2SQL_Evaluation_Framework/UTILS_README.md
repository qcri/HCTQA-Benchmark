# Utilities Reference

This document describes the utility modules in `src/utils/`. These are helper functions used throughout the project for database operations, logging, and paraphrase scoring.

---

## `src/utils/sql_utils.py`

Core database and SQL utilities for schema extraction, query execution, and SQL comparison.

### `validate_db_path(db_path: Path) -> None`
**Purpose:** Validate that a SQLite database file exists at the given path.

**Parameters:**
- `db_path: Path` — Path to the SQLite database file.

**Returns:** None

**Raises:** `FileNotFoundError` if the file does not exist.

---

### `connect_to_db(db_path: Path) -> sqlite3.Connection`
**Purpose:** Open a SQLite database connection with automatic resource cleanup support.

**Parameters:**
- `db_path: Path` — Path to the SQLite database file.

**Returns:** `sqlite3.Connection` — An open connection object (use with `with` or `closing()` for proper cleanup).

---

### `get_table_names(conn: sqlite3.Connection) -> List[str]`
**Purpose:** Retrieve all table names from the connected database.

**Parameters:**
- `conn: sqlite3.Connection` — An open database connection.

**Returns:** `List[str]` — List of table names.

---

### `table_exists(conn: sqlite3.Connection, table_name: str) -> bool`
**Purpose:** Check if a specific table exists in the database.

**Parameters:**
- `conn: sqlite3.Connection` — An open database connection.
- `table_name: str` — Name of the table to check.

**Returns:** `bool` — `True` if the table exists, `False` otherwise.

---

### `extract_schema(db_path: Path) -> str`
**Purpose:** Extract and format the full schema of a SQLite database (table names and column information).

**Parameters:**
- `db_path: Path` — Path to the SQLite database file.

**Returns:** `str` — Formatted schema string with table and column names. Format:
```
Database name: <db_name>
Table: table1 (col1, col2, col3)
Table: table2 (col_a, col_b)
...
```

**Details:**
- Queries `sqlite_master` to get all tables.
- Uses `PRAGMA table_info()` to extract column names per table.
- Called by `generate_sql_from_dataframe()` to provide context for NL→SQL generation.
- Schemas are cached per database to avoid repeated I/O.

**Example:**
```python
from pathlib import Path
from src.utils.sql_utils import extract_schema

db_path = Path("data/database/academic/academic.sqlite")
schema = extract_schema(db_path)
print(schema)
# Output:
# Database name: academic
# Table: author (author_id, author_name)
# Table: paper (paper_id, title, author_id)
```

---

### `get_sample_rows_from_tables(db_path: Path, tables: List[str], limit: int = 3) -> Dict`
**Purpose:** Retrieve sample rows from specified tables for context in prompts or debugging.

**Parameters:**
- `db_path: Path` — Path to the SQLite database file.
- `tables: List[str]` — List of table names to sample from.
- `limit: int` — Number of sample rows per table. Default: `3`.

**Returns:** `Dict[str, Tuple[List[str], List[Tuple]]]` — Dictionary mapping table names to `(column_names, sample_rows)` tuples. Returns an empty dict if queries fail.

**Example:**
```python
samples = get_sample_rows_from_tables(
    Path("data/database/academic/academic.sqlite"),
    ["author", "paper"],
    limit=2
)
# samples = {
#     "author": (["author_id", "author_name"], [(1, "Alice"), (2, "Bob")]),
#     "paper": (["paper_id", "title", "author_id"], [(1, "Paper A", 1), (2, "Paper B", 2)])
# }
```

---

### `run_query(db_path: Path, query: str) -> pd.DataFrame`
**Purpose:** Execute a SQL query on the database and return results as a pandas DataFrame.

**Parameters:**
- `db_path: Path` — Path to the SQLite database file.
- `query: str` — SQL query string (SELECT, etc.).

**Returns:** `pd.DataFrame` — DataFrame with query results. Returns an empty DataFrame if the query fails.

**Example:**
```python
df = run_query(Path("data/database/academic/academic.sqlite"), "SELECT * FROM author LIMIT 5;")
print(df)
```

---

### `extract_tables(sql: str) -> List[str]`
**Purpose:** Extract all table names referenced in a SQL query using `sql_metadata.Parser`.

**Parameters:**
- `sql: str` — A SQL query string.

**Returns:** `List[str]` — List of table names mentioned in the query.

**Example:**
```python
tables = extract_tables("SELECT a.author_name FROM author a JOIN paper p ON a.author_id = p.author_id;")
# Returns: ["author", "paper"]
```

---

### `trim_schema(schema_text: str, max_tables: int = 5) -> str`
**Purpose:** Reduce schema size by keeping only the first N tables (useful for fitting schemas into token limits for prompts).

**Parameters:**
- `schema_text: str` — Full schema string (output from `extract_schema()`).
- `max_tables: int` — Maximum number of tables to keep. Default: `5`.

**Returns:** `str` — Trimmed schema with only the first N tables.

---

### `trim_by_tokens(text: str, max_tokens: int = 3000, model: str = "gpt-4o") -> str`
**Purpose:** Truncate text to fit within a token limit for a specific model (using `tiktoken`).

**Parameters:**
- `text: str` — Text to trim (e.g., schema or prompt).
- `max_tokens: int` — Maximum number of tokens allowed. Default: `3000`.
- `model: str` — Model name for tokenizer selection (e.g., "gpt-4o", "gpt-3.5-turbo"). Default: `"gpt-4o"`.

**Returns:** `str` — Trimmed text that fits within the token limit.

---

### `compare_df(df1: pd.DataFrame, df2: pd.DataFrame) -> bool`
**Purpose:** Compare two DataFrames for semantic equality, ignoring row/column order and NaN values.

**Parameters:**
- `df1: pd.DataFrame` — First DataFrame.
- `df2: pd.DataFrame` — Second DataFrame.

**Returns:** `bool` — `True` if DataFrames are semantically equal (same data, possibly different order), `False` otherwise.

**Details:**
- Checks if shapes match.
- Attempts element-wise comparison.
- Falls back to string-based comparison to handle NaNs and ordering.
- Useful for SQL result comparison when row order differs.

---

### `compare_sql(db_path: Path, query1: str, query2: str) -> bool`
**Purpose:** Execute two SQL queries on the same database and compare their results for equality.

**Parameters:**
- `db_path: Path` — Path to the SQLite database file.
- `query1: str` — First SQL query.
- `query2: str` — Second SQL query.

**Returns:** `bool` — `True` if both queries return equivalent results, `False` otherwise.

**Details:**
- If both queries are identical strings, returns `True` immediately.
- Executes both queries and compares results using `compare_df()`.
- Catches exceptions and returns `False` if either query fails.
- **Used extensively in `generate_sql_from_dataframe()` to verify if generated SQL produces the same results as the ground-truth SQL.**

**Example:**
```python
from pathlib import Path
from src.utils.sql_utils import compare_sql

db_path = Path("data/database/academic/academic.sqlite")
ground_truth_sql = "SELECT COUNT(*) FROM author;"
generated_sql = "SELECT COUNT(author_id) FROM author;"

is_correct = compare_sql(db_path, ground_truth_sql, generated_sql)
print(is_correct)  # True if results match
```

---

## `src/utils/logger.py`

Logging configuration for the project.

### `setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger`
**Purpose:** Create and configure a logger with file output.

**Parameters:**
- `name: str` — Logger name (e.g., "main_logger", "llama_logger").
- `log_file: Path` — Path where log messages are written.
- `level` — Logging level. Default: `logging.INFO`.

**Returns:** `logging.Logger` — Configured logger instance.

**Details:**
- Creates the log file directory if it doesn't exist.
- Formats log messages as: `<timestamp> | <level> | <message>`
- Prevents duplicate handlers if called multiple times with the same name.

**High-level usage:**
```python
from src.utils.logger import setup_logger
from pathlib import Path

logger = setup_logger("main_logger", Path("logs/main.log"))
logger.info("Pipeline started.")
logger.error("An error occurred.")
```

---

## `src/utils/paraphrase_score.py`

Paraphrase quality scoring.

### `score_paraphrase(paraphrased: str, original: str) -> float`
**Purpose:** Score how well a paraphrase matches the original question semantically.

**Parameters:**
- `paraphrased: str` — Paraphrased question.
- `original: str` — Original question.

**Returns:** `float` — Score between 0 and 1 (higher is better). Currently returns `1.0` (stub implementation).

**Note:** This function is a placeholder. A real implementation should compute semantic similarity (e.g., using embeddings, BLEU, or other metrics). Used in `main.py` to decide whether to retry paraphrasing if `score < threshold`.

**High-level usage:**
```python
from src.utils.paraphrase_score import score_paraphrase

original = "How many authors are there?"
paraphrased = "What is the total number of authors?"
score = score_paraphrase(paraphrased, original)
print(f"Score: {score}")  # Currently always returns 1.0
```

---

## Summary

| Function | Purpose | Key Parameters |
|----------|---------|-----------------|
| `extract_schema()` | Extract DB schema (tables & columns) | `db_path` |
| `compare_sql()` | Compare two SQL queries by execution | `db_path`, `query1`, `query2` |
| `run_query()` | Execute a SQL query and get results | `db_path`, `query` |
| `get_table_names()` | List all tables in database | `conn` |
| `get_sample_rows_from_tables()` | Get sample data from tables | `db_path`, `tables`, `limit` |
| `trim_schema()` | Reduce schema to first N tables | `schema_text`, `max_tables` |
| `trim_by_tokens()` | Truncate text to token limit | `text`, `max_tokens`, `model` |
| `compare_df()` | Compare two DataFrames | `df1`, `df2` |
| `setup_logger()` | Configure file-based logging | `name`, `log_file`, `level` |
| `score_paraphrase()` | Score paraphrase quality (stub) | `paraphrased`, `original` |

---

File created by the project helper.
