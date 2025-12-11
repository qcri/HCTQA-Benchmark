from pathlib import Path
import pandas as pd
from time import time
import json

from src.prepare_dataset import main as prepare_dataset
from src.utils.logger import setup_logger
from src.utils.sql_utils import extract_schema
from src.utils.paraphrase_score import score_paraphrase
from models.llama import generate_sql_from_dataframe as nl2sql_llama, paraphrase_sentence, regenerate_paraphrase
from models.qwen import generate_sql_from_dataframe as nl2sql_qwen
from models.mistral import generate_sql_from_dataframe as nl2sql_mistral


def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)



def main(
    dataset_force: bool = False,
    paraphrasing_force: bool = False,
    nl2sql_force: bool = True,
    evaluate: bool = False,
    run_llama: bool = False,
    run_qwen: bool = False,
    run_mistral: bool = True,
    threshold: float = 0.7,
    max_retries: int = 1,
):
    PROJECT_ROOT = Path().resolve()

    # Paths
    DATA_PATH = PROJECT_ROOT / "data"
    INTERIM = DATA_PATH / "interim"
    PROCESSED = DATA_PATH / "processed"
    DATABASE = DATA_PATH / "database"
    RESULT = PROJECT_ROOT / "result"
    LOG_PATH = PROJECT_ROOT / "logs"

    ensure_dirs(PROCESSED, RESULT, LOG_PATH)

    dataset_path = INTERIM / "generated_queries.csv"
    paraphrased_csv_path = PROCESSED / "output_paraphrased.csv"

    # loggers
    logger = setup_logger("main_logger", log_file=LOG_PATH / "main.log")

    # Step 1: Prepare dataset
    if dataset_force or not dataset_path.exists():
        logger.info("Generating interim dataset...")
        prepare_dataset()
        logger.info("Interim dataset created.")
    else:
        logger.info("Interim dataset already exists. Skipping...")

    # Step 2: Paraphrasing
    if paraphrasing_force or not paraphrased_csv_path.exists():
        logger.info("Starting paraphrasing...")
        t0 = time()
        dataset_df = pd.read_csv(dataset_path)

        # stable row id to use later for merges
        if "row_id" not in dataset_df.columns:
            dataset_df.insert(0, "row_id", range(len(dataset_df)))

        for i, row in dataset_df.iterrows():
            db_name = row["db_name"]
            question = row["natural_language"]
            db_full_path = DATABASE / db_name / f"{db_name}.sqlite"

            try:
                schema = extract_schema(db_path=db_full_path)
                paraphrased = paraphrase_sentence(question, schema)
                score = score_paraphrase(paraphrased=paraphrased, original=question)

                if score < threshold:
                    for attempt in range(max_retries):
                        logger.warning(f"[Row {i}] Low paraphrase score ({score:.2f}). Retry {attempt + 1}...")
                        paraphrased = regenerate_paraphrase(question, schema)
                        # keep arg order consistent with first call
                        score = score_paraphrase(paraphrased=paraphrased, original=question)
                        logger.info(f"[Row {i}] Retry {attempt + 1} score: {score:.2f}")
                        if score >= threshold:
                            break

            except Exception as e:
                logger.warning(f"[Row {i}] Paraphrasing error: {e}")
                paraphrased = question
                score = 0.0

            dataset_df.loc[i, "paraphrased_nl"] = paraphrased
            dataset_df.loc[i, "paraphrased_score"] = score

            logger.info(f"[Row {i}] Original: {question}")
            logger.info(f"[Row {i}] Paraphrased: {paraphrased}")
            logger.info(f"[Row {i}] Score: {score:.2f}")

        dataset_df.to_csv(paraphrased_csv_path, index=False)
        logger.info(f"Paraphrasing complete in {time() - t0:.2f}s. Saved to {paraphrased_csv_path}")
    else:
        logger.info("Paraphrased dataset already exists. Skipping...")

    # Step 3: NL2SQL generation
    if nl2sql_force:
        logger.info("Starting NL2SQL generation...")
        paraphrased_df = pd.read_csv(paraphrased_csv_path)
        if "row_id" not in paraphrased_df.columns:
            paraphrased_df.insert(0, "row_id", range(len(paraphrased_df)))

        # --- LLaMA ---
        if run_llama:
            t = time()
            logger.info("LLaMA evaluation...")
            try:
                _ = nl2sql_llama(
                    paraphrased_df=paraphrased_df.copy(),
                    database_path=DATABASE,
                    logger=setup_logger("llama_logger", log_file=LOG_PATH / "llama.log"),
                    result_path=RESULT,
                    checkpoint_every=None,
                    store_sql=True,
                )
                logger.info(f"LLaMA evaluation completed in {time() - t:.2f}s. Results: {RESULT}/llama_results.csv")
            except Exception as e:
                logger.error(f"LLaMA evaluation error: {e}")
               
        else:
            logger.info("Skipping LLaMA.")

        # --- Qwen ---
        if run_qwen:
            t = time()
            logger.info("Qwen evaluation...")
            try:
                _ = nl2sql_qwen(
                    paraphrased_df=paraphrased_df.copy(),
                    database_path=DATABASE,
                    logger=setup_logger("qwen_logger", log_file=LOG_PATH / "qwen.log"),
                    result_path=RESULT,
                    checkpoint_every=None,
                    store_sql=True,
                )
                logger.info(f"Qwen evaluation completed in {time() - t:.2f}s. Results: {RESULT}/qwen_results.csv")
            except Exception as e:
                logger.error(f"Qwen evaluation error: {e}")
        else:
            logger.info("Skipping Qwen.")

        # --- Mistral ---
        if run_mistral:
            t = time()
            logger.info("Mistral evaluation...")
            try:
                _ = nl2sql_mistral(
                    paraphrased_df=paraphrased_df.copy(),
                    database_path=DATABASE,
                    logger=setup_logger("mistral_logger", log_file=LOG_PATH / "mistral.log"),
                    result_path=RESULT,
                    checkpoint_every=None,
                    store_sql=True,
                )
                logger.info(f"Mistral evaluation completed in {time() - t:.2f}s. Results: {RESULT}/mistral_results.csv")
            except Exception as e:
                logger.error(f"Mistral evaluation error: {e}")
        else:
            logger.info("Skipping Mistral.")
    else:
        logger.info("Skipping NL2SQL generation.")

    # Step 4: Evaluation/merge
    if evaluate:
        logger.info("Comparing results from all models...")
        t_eval = time()

        # read available results safely
        dfs = []
        keys = []

        llama_csv = RESULT / "llama_results.csv"
        qwen_csv = RESULT / "qwen_results.csv"
        mistral_csv = RESULT / "mistral_results.csv"

        if llama_csv.exists():
            dfs.append(pd.read_csv(llama_csv))
            keys.append("llama_")
        if qwen_csv.exists():
            dfs.append(pd.read_csv(qwen_csv))
            keys.append("qwen_")
        if mistral_csv.exists():
            dfs.append(pd.read_csv(mistral_csv))
            keys.append("mistral_")

        base = pd.read_csv(paraphrased_csv_path)
        if "row_id" not in base.columns:
            base.insert(0, "row_id", range(len(base)))

        final_df = base.copy()

        # merge on row_id if present in model outputs; else fall back to index join
        for df in dfs:
            if "row_id" in df.columns:
                final_df = final_df.merge(df.filter(regex="^(row_id|llama_|qwen_|mistral_)"), on="row_id", how="left")
            else:
                final_df = final_df.join(df.filter(regex="^(llama_|qwen_|mistral_)"))

        # ensure flags exist
        for col in [
            "llama_original_correct","llama_para_correct",
            "qwen_original_correct","qwen_para_correct",
            "mistral_original_correct","mistral_para_correct",
        ]:
            if col not in final_df.columns:
                final_df[col] = False

        all_ok = (
            final_df["llama_original_correct"].eq(True)
            & final_df["llama_para_correct"].eq(True)
            & final_df["qwen_original_correct"].eq(True)
            & final_df["qwen_para_correct"].eq(True)
            & final_df["mistral_original_correct"].eq(True)
            & final_df["mistral_para_correct"].eq(True)
        )
        
        final_df["all_models_correct"] = all_ok.fillna(False).astype("int8")

        out = RESULT / "results.csv"
        final_df.to_csv(out, index=False)
        logger.info(f"Evaluation complete in {time() - t_eval:.2f}s. Saved to: {out}")

        output_list = []
        for _, row in final_df.iterrows():

            # build dictionaries (ordered keys: llama, qwen, mistral)
            original_dict = {
                "llama": row.get("llama_original_correct", False),
                "qwen": row.get("qwen_original_correct", False),
                "mistral": row.get("mistral_original_correct", False),
            }
            para_dict = {
                "llama": row.get("llama_para_correct", False),
                "qwen": row.get("qwen_para_correct", False),
                "mistral": row.get("mistral_para_correct", False),
            }

            # compute final_paraphased_output
            final_paraphased_output = False
            true_in_original = [m for m, v in original_dict.items() if v]
            if true_in_original and all(para_dict.get(m, False) for m in true_in_original):
                final_paraphased_output = True

            # JSON row
            output_list.append({
                "row_id": row.get("row_id", ""),
                "db_name": row.get("db_name", ""),
                "NLQorg": row.get("natural_language", ""),
                "NLQparap": row.get("paraphrased_nl", ""),
                "SQLorg": row.get("sql_query", ""),
                "NLQorgOutput": {
                    "llama": row.get("llama_para_correct", False),
                    "qwen": row.get("qwen_para_correct", False),
                    "mistral": row.get("mistral_para_correct", False),
                },
                "NLQparapOutput": {
                    "llama": row.get("llama_original_correct", False),
                    "qwen": row.get("qwen_original_correct", False),
                    "mistral": row.get("mistral_original_correct", False),
                },
                # replaced "correct" with this
                "final_paraphased_output": final_paraphased_output,
            })

        # Save as JSON
        json_out = RESULT / "structured_result.json"
        with open(json_out, "w") as f:
            json.dump(output_list, f, indent=2)

if __name__ == "__main__":
    main(dataset_force = False,  # force dataset generation
    paraphrasing_force = False,  # force paraphrasing
    nl2sql_force = False,     # force NL2SQL generation
    evaluate = True,   # evaluate results
    run_llama = False,  # run LLaMA model
    run_qwen = False,  # run Qwen model
    run_mistral = False,  # run Mistral model
    threshold = 0.7,  # paraphrase score threshold
    max_retries = 1,  # max retries for paraphrasing if score is low
    )
