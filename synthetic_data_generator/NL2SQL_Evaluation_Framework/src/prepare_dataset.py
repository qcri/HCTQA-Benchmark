import os
import json
import re
import pandas as pd
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from itertools import islice
from src.utils.sql_utils import extract_schema, get_sample_rows_from_tables, run_query, extract_tables, trim_schema, trim_by_tokens
from src.utils.logger import setup_logger

# --- Environment & Constants ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI")
client = OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "database"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim"
LOG_PATH = PROJECT_ROOT / "logs" / "prepare_dataset.log"
OUTPUT_CSV_PATH = OUTPUT_PATH / "generated_queries.csv"

logger = setup_logger("prepare_dataset", LOG_PATH)

# --- LLM  ---
def generate_sql_query(schema, model="gpt-4o"):
    system_prompt = (
        "You are an expert SQL query generator.\n"
        "Given a database schema and example rows, generate a list of SQL queries.\n"
        "- Only generate syntactically correct SQL queries for SQLite.\n"
        "- Do not hallucinate tables or columns.\n"
        "- Use values seen in the schema or sample rows.\n"
        "- Output ONLY a JSON array of SQL query strings.\n"
        "- No explanations or formatting."
    )

    user_prompt = (
        f"Here is the database schema. Generate 2 queries:\n{schema}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    raw_text = response.choices[0].message.content.strip()

    # Preprocessing the response 
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json|sql)?\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("Error while deoding response.")
        return []


def regenerate_sql(wrong_query, sample_rows, model="gpt-4o"):
    sample_context = ""
    for table, (cols, rows) in sample_rows.items():
        sample_context += f"\nTable: {table} ({', '.join(cols)})\n"
        for row in rows:
            sample_context += " | ".join(str(val) for val in row) + "\n"

    system_prompt = (
        "You are an expert SQL query fixer.\n"
        "- You are given an incorrect sql query. \n"
        "- The query might look syntactically fine but can fail due to incorrect values.\n"
        "- You are provided with sample rows from relevant tables to guide correction.\n"
        "- Only use values shown in sample rows.\n"
        "- Fix incorrect queries and return it.\n"
        "- Output JSON array with exactly one fixed SQL string."
    )
    user_prompt = (
        f"Here is the incorrect SQL query:\n{wrong_query}\n\n"
        f"Sample rows:\n{sample_context}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=800
        )

        raw_output = response.choices[0].message.content.strip()
        if raw_output.startswith("```"):
            raw_output = re.sub(r"```(json)?", "", raw_output).strip("`").strip()

        parsed = json.loads(raw_output)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], str):
            return parsed[0]
        return wrong_query
    except Exception as e:
        logger.error(f"Error during SQL regeneration: {e}")
        return wrong_query

def generate_nl(sql_query, result, model="gpt-4o"):
    if hasattr(result, 'to_string'):
        result_str = result.to_string(index=False)
    else:
        result_str = str(result)

    system_prompt = (
        "You are a helpful assistant that converts SQL queries and their results into natural language questions.\n"
        "- Use only visible columns and values.\n"
        "- Do not add new columns.\n"
        "- Output a single clear question."
    )
    user_prompt = f"SQL Query:\n{sql_query}\n\nResult:\n{result_str}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=800
        )
        output = response.choices[0].message.content.strip()
        return re.sub(r"```(text)?", "", output).strip("`").strip()
    except Exception as e:
        logger.error(f"Error generating natural language: {e}")
        return "Failed to generate NL."

# --- Main Dataset Builder ---
def main():
    rows = []
    for db_file in DB_PATH.iterdir():
        db_full_path = DB_PATH / db_file / f"{db_file.name}.sqlite"
        logger.info(f"Processing DB: {db_file.name}")
        
        schema_text = extract_schema(db_full_path)
        schema_text = trim_schema(schema_text, max_tables=5)
        schema_text = trim_by_tokens(schema_text, max_tokens=3000)
        
        try:
            sql_queries = generate_sql_query(schema_text)
        except Exception as e:
            logger.error(f"Failed to generate SQL for {db_file.name}: {e}")
            continue

        for i, query in enumerate(sql_queries):
            logger.info(f"Running Query {i+1}: {query}")
            sample_rows = get_sample_rows_from_tables(db_full_path, extract_tables(query))
            result = run_query(db_full_path, query)

            if result.empty:
                logger.warning("Original query returned empty. Attempting fix...")
                query = regenerate_sql(query, sample_rows)
                result = run_query(db_full_path, query)
                if result.empty:
                    logger.warning("Regenerated query also returned empty. Skipping.")
                    continue
                logger.info("Regenerated query succeeded.")
                logger.info(f"Corrected Query: {query}")
            else:
                logger.info("Original query succeeded.")

            nl_question = generate_nl(query, result)
            logger.info(f"Generated NL: {nl_question}")

            rows.append({
                "db_name": db_file.name,
                "natural_language": nl_question,
                "sql_query": query
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info(f" Saved dataset to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
