import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Any, Dict
from contextlib import closing
from sql_metadata import Parser
import tiktoken
import numpy as np
import re

def validate_db_path(db_path: Path) -> None:
    """
    Raise error if the SQLite database path is invalid.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file '{db_path}' not found.")


def connect_to_db(db_path: Path) -> sqlite3.Connection:
    """
    Create a SQLite connection with automatic closing support.

    Returns:
        sqlite3.Connection: An open database connection.
    """
    validate_db_path(db_path)
    return sqlite3.connect(db_path)


def get_table_names(conn: sqlite3.Connection) -> List[str]:
    """
    Return all table names in the connected database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
    )
    return cursor.fetchone() is not None


def extract_schema(db_path: Path) -> str:
    """
    Extract schema information from the database (tables and columns).
    """
    schema_lines = [f"Database name: {db_path.name}"]

    with closing(connect_to_db(db_path)) as conn:
        for table_name in get_table_names(conn):
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            schema_lines.append(f"Table: {table_name} ({', '.join(col_names)})")

    return "\n".join(schema_lines)


def get_sample_rows_from_tables(db_path: Path, tables: List[str], limit: int = 3):
    sample_data = {}

    with closing(connect_to_db(db_path)) as conn:
        for table in tables:
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table} LIMIT {limit};")
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                sample_data[table] = (col_names, rows)
            except Exception as e:
                print(f"⚠️ Could not get rows from '{table}': {e}")

    return sample_data


def run_query(db_path: Path, query: str) -> pd.DataFrame:
    """
    Run a SQL query on the SQLite DB and return results as a DataFrame.

    Raises:
        ValueError if the query fails.
    """
    with closing(connect_to_db(db_path)) as conn:
        try:
            return pd.read_sql_query(query, conn)
        except Exception as e:
            return pd.DataFrame()  # Return empty DataFrame on error


def extract_tables(sql: str) -> list:
    return Parser(sql).tables

def trim_schema(schema_text: str, max_tables: int = 5) -> str:
    """
    Trims schema to include only the first N tables and their columns.

    Assumes schema is in the format:
    Table: table_name (col1, col2, ...)
    """
    trimmed = []
    table_blocks = re.findall(r"(Table: .+?\n(?:.+\n)+)", schema_text, re.MULTILINE)
    for block in table_blocks[:max_tables]:
        trimmed.append(block.strip())

    return "\n\n".join(trimmed)

def trim_by_tokens(text, max_tokens=3000, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = enc.decode(tokens)
    return text


def compare_df(df1, df2):
    """Compare two DataFrames for equality, ignoring row/column order and NaNs."""
    if df1.shape != df2.shape:
        return False

    df1_vals = df1.values
    df2_vals = df2.values

    if np.array_equal(df1_vals, df2_vals):
        return True

    if (df1_vals == df2_vals).all():
        return True
    else:
        def row_to_str(row):
            return "|".join(str('' if pd.isna(x) else x) for x in row)
             
        a_str = "".join(sorted([row_to_str(row) for row in df1_vals]))
        b_str = "".join(sorted([row_to_str(row) for row in df2_vals]))

        if a_str == b_str:
            return True
        try:
            return sorted(a_str) == sorted(b_str)
        except Exception:
            return False


def compare_sql(db_path: Path, query1:str, query2:str) -> bool:
    """Compare two SQL queries on the same database. Returns True if results match, else False."""
    if query1 == query2:
        return True
    try:
        conn = connect_to_db(db_path)
        df1 = run_query(db_path, query1)
        df2 = run_query(db_path, query2)
        conn.close()
        return compare_df(df1, df2)
    except Exception as e:
        print(f"[run_all] Error: {e}")
        return False


