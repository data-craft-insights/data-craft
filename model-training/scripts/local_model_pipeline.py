 #model-training/scripts/local_model_pipeline.py

"""
Local runner for the model pipeline DAG.

Goal:
- Use the SAME logic as model_pipeline_dag.py
- But run it sequentially in Python, without Airflow
- Use the same user_queries.txt (150 queries)

This is just for debugging before running in Docker/Airflow.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# ---------- PATH SETUP ----------
THIS_FILE = Path(__file__).resolve()
MODEL_TRAINING_DIR = THIS_FILE.parents[1]       # .../model-training
ROOT_DIR = MODEL_TRAINING_DIR.parent           # repo root
SCRIPTS_DIR = MODEL_TRAINING_DIR / "scripts"
DATA_DIR = MODEL_TRAINING_DIR / "data"

sys.path.insert(0, str(SCRIPTS_DIR))

# ---------- OPTIONAL: .env ----------
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    print("✓ .env loaded")
except Exception:
    print("⚠ python-dotenv not installed or .env missing – continuing")

# ---------- IMPORTS FROM YOUR SCRIPTS ----------
from data_loader import ModelDataLoader
from feature_engineering import FeatureEngineer
from metadata_manager import MetadataManager
from prompts import build_prompt, FEW_SHOT_EXAMPLES
from model_evaluator import ModelEvaluator
from bias_detector import BiasDetector
from model_selector import ModelSelector
from response_saver import ResponseSaver
from query_executor import QueryExecutor
# If these exist & you want to test them too, uncomment:
# from hyperparameter_tuner import HyperparameterTuner
# from sensitivity_analysis import SensitivityAnalyzer

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ---------- CONFIG (mirrors DAG, but via env vars where possible) ----------

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "datacraft-478300")
REGION = os.getenv("REGION", "us-east1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "model-datacraft")
DATASET_ID = os.getenv("BQ_DATASET", "datacraft_ml")

DATASET_NAME = "orders"

MODELS_TO_EVALUATE = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

QUERIES_FILE = DATA_DIR / "user_queries.txt"

# local outputs (not /opt/airflow)
OUTPUT_BASE_DIR = ROOT_DIR / "local_outputs" / "model-training"
EVALUATION_DIR = OUTPUT_BASE_DIR / "evaluation"
BIAS_DIR = OUTPUT_BASE_DIR / "bias"
SELECTION_DIR = OUTPUT_BASE_DIR / "model-selection"
BEST_MODEL_DIR = OUTPUT_BASE_DIR / "best-model-responses"

for p in [EVALUATION_DIR, BIAS_DIR, SELECTION_DIR, BEST_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ---------- HELPER FUNCTIONS (copied from DAG, slightly adapted) ----------

def _call_gemini_model(prompt: str, model_name: str) -> str:
    """Call specific Gemini model (same as DAG)."""
    vertexai.init(project=PROJECT_ID, location=REGION)

    model = GenerativeModel(
        model_name,
        generation_config=GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
    )

    response = model.generate_content(prompt)
    return response.text


def _parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini response (same logic as DAG)."""
    import re

    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    response_text = response_text.strip()

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
            except Exception:
                result = {
                    "sql_query": "SELECT * FROM dataset LIMIT 100;",
                    "visualization": {"type": "table", "title": "Results"},
                    "explanation": "Fallback response",
                }
        else:
            result = {
                "sql_query": "SELECT * FROM dataset LIMIT 100;",
                "visualization": {"type": "table", "title": "Results"},
                "explanation": "Fallback response",
            }

    # Ensure required fields
    result.setdefault("sql_query", "SELECT * FROM dataset LIMIT 100;")
    result.setdefault("visualization", {"type": "table", "title": "Results"})
    result.setdefault("explanation", "Generated query")

    return result


# ---------- PIPELINE STEPS (XCom → simple Python variables) ----------

def step_1_load_data():
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA TO BIGQUERY")
    print("="*70)

    loader = ModelDataLoader(
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
    )

    table_name = f"{DATASET_NAME}_processed"

    if loader.table_exists(table_name):
        print(f"✓ Table {table_name} already exists in BigQuery")
        info = loader.get_table_info(table_name)
        print(f"  Rows: {info.get('num_rows', 0)}")
        return loader, table_name

    df = loader.load_processed_data_from_gcs(DATASET_NAME, stage="validated")
    table_id = loader.load_to_bigquery(df, DATASET_NAME, table_suffix="_processed")
    print(f"✓ Loaded {len(df):,} rows into {table_id}")
    return loader, f"{DATASET_NAME}_processed"


def step_2_generate_features(loader):
    print("\n" + "="*70)
    print("STEP 2: GENERATE FEATURES & METADATA")
    print("="*70)

    df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=10000)
    print(f"✓ Loaded {len(df):,} rows for feature engineering")

    schema = loader.load_dataset_profile_from_gcs(DATASET_NAME)
    engineer = FeatureEngineer(df, schema, DATASET_NAME)

    metadata = engineer.generate_metadata()
    llm_context = engineer.create_llm_context()
    summary = engineer.get_feature_summary()

    print("✓ Metadata & LLM context generated")
    return metadata, llm_context, summary


def step_3_store_metadata(metadata, llm_context):
    print("\n" + "="*70)
    print("STEP 3: STORE METADATA IN BIGQUERY")
    print("="*70)

    manager = MetadataManager(PROJECT_ID, DATASET_ID)
    manager.store_metadata(DATASET_NAME, metadata, llm_context)
    retrieved = manager.get_metadata(DATASET_NAME)

    if not retrieved:
        raise RuntimeError("❌ Failed to verify stored metadata")
    print("✓ Metadata stored and verified")
    return manager, retrieved


def step_4_read_queries():
    print("\n" + "="*70)
    print("STEP 4: READ USER QUERIES")
    print("="*70)

    if not QUERIES_FILE.exists():
        raise FileNotFoundError(f"Query file not found: {QUERIES_FILE}")

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not queries:
        raise ValueError("No queries found in user_queries.txt")

    print(f"✓ Read {len(queries)} queries")
    for i, q in enumerate(queries[:10], 1):
        print(f"  {i}. {q}")
    if len(queries) > 10:
        print(f"  ... and {len(queries) - 10} more")

    return queries


def step_5_process_queries_with_models(queries, llm_context):
    print("\n" + "="*70)
    print("STEP 5: PROCESS QUERIES WITH MULTIPLE MODELS")
    print("="*70)

    all_model_responses = {}

    print(f"Evaluating {len(MODELS_TO_EVALUATE)} models on {len(queries)} queries...\n")

    for model_name in MODELS_TO_EVALUATE:
        print("\n" + "="*70)
        print(f"MODEL: {model_name}")
        print("="*70)

        model_responses = []

        for idx, user_query in enumerate(queries, 1):
            print(f"\n[{idx}/{len(queries)}] {user_query}")
            try:
                start = time.time()
                prompt = build_prompt(user_query, llm_context, FEW_SHOT_EXAMPLES)
                response_text = _call_gemini_model(prompt, model_name)
                elapsed = time.time() - start

                parsed = _parse_gemini_response(response_text)

                model_responses.append(
                    {
                        "query_number": idx,
                        "user_query": user_query,
                        "sql_query": parsed["sql_query"],
                        "visualization": parsed["visualization"],
                        "explanation": parsed["explanation"],
                        "raw_response": response_text,
                        "response_time": elapsed,
                        "status": "success",
                    }
                )
                print(f"  ✓ Success ({elapsed:.2f}s)")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                model_responses.append(
                    {
                        "query_number": idx,
                        "user_query": user_query,
                        "status": "failed",
                        "error": str(e),
                        "response_time": 0,
                    }
                )

        success_count = sum(r["status"] == "success" for r in model_responses)
        print(f"\n✓ Completed {model_name}: {success_count}/{len(queries)} successful")

        all_model_responses[model_name] = model_responses

    return all_model_responses


def step_6_evaluate_models(all_model_responses, loader):
    print("\n" + "="*70)
    print("STEP 6: EVALUATE ALL MODELS")
    print("="*70)

    test_df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=1000)
    evaluator = ModelEvaluator(PROJECT_ID, DATASET_ID, str(EVALUATION_DIR))

    all_reports = {}

    for model_name, responses in all_model_responses.items():
        print("\n" + "="*70)
        print(f"EVALUATING: {model_name}")
        print("="*70)

        metrics = evaluator.evaluate_model_responses(
            model_name=model_name,
            responses=responses,
            test_dataframe=test_df,
        )
        evaluator.save_evaluation_report(metrics, model_name)
        all_reports[model_name] = metrics

    evaluator.generate_comparison_report(all_reports)
    print(f"✓ Evaluation complete for {len(all_reports)} models")

    return all_reports


def step_7_detect_bias(all_model_responses, metadata):
    print("\n" + "="*70)
    print("STEP 7: BIAS DETECTION FOR ALL MODELS")
    print("="*70)

    detector = BiasDetector(str(BIAS_DIR))
    all_bias_reports = {}

    for model_name, responses in all_model_responses.items():
        print("\n" + "="*70)
        print(f"BIAS: {model_name}")
        print("="*70)

        bias_report = detector.detect_all_biases(
            model_name=model_name,
            responses=responses,
            dataset_metadata=metadata,
        )

        detector.save_bias_report(bias_report, model_name)
        all_bias_reports[model_name] = bias_report

    detector.generate_bias_comparison(all_bias_reports)
    print(f"✓ Bias detection complete for {len(all_bias_reports)} models")

    return all_bias_reports


def step_8_select_best_model(evaluation_reports, bias_reports):
    print("\n" + "="*70)
    print("STEP 8: SELECT BEST MODEL")
    print("="*70)

    selector = ModelSelector(str(SELECTION_DIR))
    selection_report = selector.select_best_model(evaluation_reports, bias_reports)
    report_file = selector.save_selection_report(selection_report)
    summary_text = selector.generate_selection_summary(selection_report)

    best_model_name = selection_report["best_model"]["name"]

    print(summary_text)
    print(f"\n✓ Best model: {best_model_name}")
    print(f"✓ Selection report: {report_file}")

    return best_model_name, selection_report


def step_9_execute_and_validate(best_model_name, all_model_responses):
    print("\n" + "="*70)
    print("STEP 9: EXECUTE & VALIDATE SQL QUERIES")
    print("="*70)

    best_responses = all_model_responses.get(best_model_name, [])
    executor = QueryExecutor(PROJECT_ID, DATASET_ID)

    queries_to_execute = [
        {
            "query_number": r["query_number"],
            "user_query": r["user_query"],
            "sql_query": r["sql_query"],
            "visualization": r["visualization"],
        }
        for r in best_responses
        if r.get("status") == "success"
    ]

    print(f"Executing {len(queries_to_execute)} queries from {best_model_name}...")

    execution_results = []
    for item in queries_to_execute:
        res = executor.execute_and_validate(
            user_query=item["user_query"],
            sql_query=item["sql_query"],
            table_name=f"{DATASET_NAME}_processed",
            visualization=item["visualization"],
        )
        res["query_number"] = item["query_number"]
        execution_results.append(res)

    results_dir = BEST_MODEL_DIR / "execution_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = executor.save_execution_results(
        results=execution_results,
        output_dir=str(results_dir),
    )

    total = len(execution_results)
    executed = sum(r["execution_status"] == "success" for r in execution_results)
    valid = sum(bool(r["results_valid"]) for r in execution_results)

    metrics = {
        "total_queries": total,
        "successfully_executed": executed,
        "valid_results": valid,
        "execution_success_rate": (executed / total * 100) if total else 0,
        "result_validity_rate": (valid / executed * 100) if executed else 0,
        "overall_accuracy": (valid / total * 100) if total else 0,
    }

    print("\nExecution Summary:")
    print(f"  Total queries:         {total}")
    print(f"  Executed successfully: {executed}")
    print(f"  Valid results:         {valid}")
    print(f"  Overall accuracy:      {metrics['overall_accuracy']:.1f}%")
    print(f"  Results file:          {results_file}")

    return execution_results, metrics


def step_10_save_best_responses(
    best_model_name,
    all_model_responses,
    execution_results,
    selection_report,
    accuracy_metrics,
):
    print("\n" + "="*70)
    print("STEP 10: SAVE BEST MODEL RESPONSES")
    print("="*70)

    best_responses = all_model_responses.get(best_model_name, [])
    if not best_responses:
        raise ValueError(f"No responses for best model {best_model_name}")

    # merge execution results
    by_query = {r["query_number"]: r for r in execution_results}
    merged = []
    for r in best_responses:
        qnum = r["query_number"]
        exec_res = by_query.get(qnum)
        merged_item = r.copy()
        if exec_res:
            merged_item["execution_status"] = exec_res.get("execution_status")
            merged_item["results_valid"] = exec_res.get("results_valid")
            merged_item["result_count"] = exec_res.get("result_count")
            merged_item["natural_language_answer"] = exec_res.get(
                "natural_language_answer"
            )
            merged_item["validation_checks"] = exec_res.get("validation_checks")
        merged.append(merged_item)

    saver = ResponseSaver(PROJECT_ID, BUCKET_NAME, str(BEST_MODEL_DIR))
    save_result = saver.save_best_model_responses(
        model_name=best_model_name,
        responses=merged,
        selection_report=selection_report,
    )

    # accuracy metrics file
    acc_file = BEST_MODEL_DIR / "accuracy_metrics.json"
    with acc_file.open("w") as f:
        json.dump(accuracy_metrics, f, indent=2)

    metadata_file = saver.save_metadata_for_deployment(
        model_name=best_model_name,
        selection_report=selection_report,
    )

    print(f"✓ Saved {save_result['files_saved']} files")
    print(f"✓ Local directory: {save_result['local_directory']}")
    print(f"✓ Deployment metadata: {metadata_file}")
    print(f"✓ Accuracy: {accuracy_metrics['overall_accuracy']:.1f}%")

    return save_result, metadata_file


def step_11_final_summary(
    best_model_name,
    selection_report,
    accuracy_metrics,
    save_result,
):
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    summary = {
        "pipeline_completion_time": datetime.now().isoformat(),
        "dataset": DATASET_NAME,
        "models_evaluated": len(MODELS_TO_EVALUATE),
        "models_list": MODELS_TO_EVALUATE,
        "best_model": {
            "name": best_model_name,
            "score": selection_report["best_model"]["composite_score"],
            "performance": selection_report["best_model"]["performance_score"],
            "bias": selection_report["best_model"]["bias_score"],
        },
        "accuracy": accuracy_metrics,
        "outputs": {
            "local_directory": save_result["local_directory"],
            "files_saved": save_result["files_saved"],
        },
    }

    summary_file = OUTPUT_BASE_DIR / "pipeline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Best model: {best_model_name}")
    print(f"  Composite score: {summary['best_model']['score']:.2f}")
    print(f"  Overall accuracy: {accuracy_metrics['overall_accuracy']:.1f}%")
    print(f"Summary file: {summary_file}")


# ---------- MAIN ENTRYPOINT ----------

def main():
    print("\n" + "="*70)
    print("LOCAL MODEL PIPELINE RUN")
    print("="*70)

    # 1–3: data + metadata
    loader, table_name = step_1_load_data()
    metadata, llm_context, feature_summary = step_2_generate_features(loader)
    manager, retrieved_metadata = step_3_store_metadata(metadata, llm_context)

    # 4: queries
    queries = step_4_read_queries()

    # 5: LLM calls
    all_model_responses = step_5_process_queries_with_models(queries, retrieved_metadata["llm_context"])

    # 6: evaluation
    evaluation_reports = step_6_evaluate_models(all_model_responses, loader)

    # 7: bias
    bias_reports = step_7_detect_bias(all_model_responses, metadata)

    # 8: selection
    best_model_name, selection_report = step_8_select_best_model(
        evaluation_reports, bias_reports
    )

    # 9: execute & validate
    execution_results, accuracy_metrics = step_9_execute_and_validate(
        best_model_name, all_model_responses
    )

    # 10: save best responses
    save_result, metadata_file = step_10_save_best_responses(
        best_model_name,
        all_model_responses,
        execution_results,
        selection_report,
        accuracy_metrics,
    )

    # 11: summary
    step_11_final_summary(
        best_model_name,
        selection_report,
        accuracy_metrics,
        save_result,
    )

    print("\n✅ Local pipeline run completed.\n")


if __name__ == "__main__":
    main() 
