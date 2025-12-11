import os
import numpy as np
import time
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer
from dotenv import load_dotenv
from pathlib import Path
from src.utils.sql_utils import extract_schema, compare_sql
import csv
from typing import Optional, Dict
from .prompt_templates import (
    PARAPHRASE_SYSTEM_PROMPT,
    SQL_GEN_SYSTEM_PROMPT,
    REGENERATE_SQL_PROMPT,
)

from multiprocessing import set_start_method

load_dotenv()  # Load environment variables from .env file

# Set threading layer early
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['HF_TOKEN'] = os.getenv("HF_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HF_API_KEY")


# === Change model and tokenizer here ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# Global variables (lazy-loaded)
_llm_nl2nl = None  # Singleton LLM instance
_sampling_nl2nl = SamplingParams(temperature=0, max_tokens=100)  # Sampling params for LLM
_tokenizer_nl2nl = None  # Singleton tokenizer instance

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_tokenizer():
    """Return a singleton tokenizer instance for paraphrasing."""
    global _tokenizer_nl2nl
    if _tokenizer_nl2nl is None:
        _tokenizer_nl2nl = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            trust_remote_code=True
        )
    return _tokenizer_nl2nl

def get_llm_nl2nl():
    """Return a singleton LLM instance for paraphrasing."""
    global _llm_nl2nl
    if _llm_nl2nl is None:
        _llm_nl2nl = LLM(
            model=MODEL_NAME,
            max_model_len=2048,
            tokenizer=TOKENIZER_NAME,
            hf_token=os.environ['HF_TOKEN'],
            trust_remote_code=True,
            tensor_parallel_size=1  # Parallel
        )
    return _llm_nl2nl


def paraphrase_sentence(sentence: str, schema: str = "") -> str:
    """
    Paraphrase an NL question while guaranteeing SQL-equivalent semantics.
    Falls back to the original if any semantic-drift guard fails.
    """
    tokenizer = get_tokenizer()

    usr_msg = (
        f"Here is the table schema for context:\n{schema}\n\n"
        f"Original question:\n{sentence}"
    )

    chat_input = tokenizer.apply_chat_template(
        [{"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
         {"role": "user", "content": usr_msg}],
        tokenize=False,
        add_generation_prompt=True
    )

    llm = get_llm_nl2nl()
    try:
        outputs = llm.generate(chat_input, sampling_params=_sampling_nl2nl)
        para = outputs[0].outputs[0].text.strip()
        return para
    except Exception as e:
        print(f"[paraphrase_sentence] Error: {e}")
        return sentence


def generate_sql(nl_question: str, schema: str) -> str:
    """
    Generate a SQL query from a natural language question and schema using LLaMA model.

    Args:
        nl_question (str): Natural language question.
        schema (str): Extracted database schema as text.

    Returns:
        str: Generated SQL query (as plain text).
    """
    tokenizer = get_tokenizer()
    llm = get_llm_nl2nl()

    user_prompt = (
        f"Here is the database schema:\n{schema}\n\n"
        f"Now, write a valid SQL query for the following question:\n{nl_question}"
    )

    chat_input = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SQL_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        outputs = llm.generate(chat_input, sampling_params=_sampling_nl2nl)
        raw_output = outputs[0].outputs[0].text.strip()

        if raw_output.startswith("[") and raw_output.endswith("]"):
            import json
            parsed = json.loads(raw_output)
            return parsed[0] if isinstance(parsed, list) and parsed else raw_output

        return raw_output  # fallback

    except Exception as e:
        print(f"[generate_sql] Error: {e}")
        return ""


def regenerate_paraphrase(question:str, schema:str) -> str:
    return question


def generate_sql_from_dataframe(
    paraphrased_df: pd.DataFrame,
    database_path: Path,
    *,
    logger,
    result_path: Optional[Path] = None,
    store_sql: bool = True,
    checkpoint_every: Optional[int] = None,
) -> pd.DataFrame:
    """
    Runs Llama over the entire dataframe:
      - generates SQL for original & paraphrased questions
      - evaluates correctness
      - writes Llama* columns into the same df
      - optionally saves CSV (llama_results.csv) and checkpoints every N rows
    """

    # Stable key for later merges
    if "row_id" not in paraphrased_df.columns:
        paraphrased_df.insert(0, "row_id", range(len(paraphrased_df)))

    # Ensure output columns exist (avoid dtype churn)
    cols = ["llama_para_correct", "llama_original_correct"]
    if store_sql:
        cols += ["llama_query_from_para", "llama_query_from_original"]
    for c in cols:
        if c not in paraphrased_df.columns:
            paraphrased_df[c] = pd.NA

    # Cache schemas per DB (big speedup)
    schema_cache: Dict[str, str] = {}

    def get_schema(db_name: str, db_full_path: Path) -> str:
        if db_name not in schema_cache:
            schema_cache[db_name] = extract_schema(db_path=db_full_path)
        return schema_cache[db_name]

    for i, row in paraphrased_df.iterrows():
        db_name = row["db_name"]
        paraphrased_question = row["paraphrased_nl"]
        original_sql = row["sql_query"]
        original_question = row["natural_language"]
        db_full_path = database_path / db_name / f"{db_name}.sqlite"

        try:
            schema = get_schema(db_name, db_full_path)

            query_para = generate_sql(paraphrased_question, schema)
            query_original = generate_sql(original_question, schema)

            para_result = compare_sql(db_full_path, original_sql, query_para)
            original_result = compare_sql(db_full_path, original_sql, query_original)

            paraphrased_df.at[i, "llama_para_correct"] = bool(para_result)
            paraphrased_df.at[i, "llama_original_correct"] = bool(original_result)

            if store_sql:
                paraphrased_df.at[i, "llama_query_from_para"] = query_para
                paraphrased_df.at[i, "llama_query_from_original"] = query_original

            if (i + 1) % 50 == 0:
                logger.info(f"[Llama] processed {i+1} rows...")

            if checkpoint_every and (i + 1) % checkpoint_every == 0 and result_path:
                ckpt_file = result_path / "llama_results_intermediate.csv"
                # Force robust CSV for checkpoints too
                paraphrased_df.to_csv(
                    ckpt_file, index=False,
                    quoting=csv.QUOTE_ALL, escapechar='\\', lineterminator='\r\n'
                )
                logger.info(f"[Llama] checkpoint saved at row {i+1}: {ckpt_file}")

        except Exception as e:
            logger.error(f"[Row {i}] llama NL2SQL error: {e}")
            paraphrased_df.at[i, "llama_para_correct"] = False
            paraphrased_df.at[i, "llama_original_correct"] = False
            if store_sql:
                paraphrased_df.at[i, "llama_query_from_para"] = None
                paraphrased_df.at[i, "llama_query_from_original"] = None

    # Build final_df with only existing columns
    base_order = [
        "row_id",
        "db_name",
        "natural_language",
        "sql_query",
        "paraphrased_nl",
        "paraphrased_score",
    ]
    sql_cols = ["llama_query_from_para", "llama_query_from_original"] if store_sql else []
    flag_cols = ["llama_para_correct", "llama_original_correct"]

    new_order = [c for c in (base_order + sql_cols + flag_cols) if c in paraphrased_df.columns]
    final_df = paraphrased_df.loc[:, new_order].copy()

    if result_path:
        out = result_path / "llama_results.csv"
        # Robust CSV settings prevent “broken” rows in Excel/simple viewers
        final_df.to_csv(
            out, index=False,
            quoting=csv.QUOTE_ALL, escapechar='\\', lineterminator='\r\n'
        )
        logger.info(f"Llama evaluation complete. Results saved to: {out}")

    return final_df


if __name__=="__main__":
    pass


