"""Airflow DAG orchestrating the training pipeline.

Stages:

    ingest -> validate -> preprocess -> train -> register

Each task is a ``PythonOperator`` calling a thin wrapper that delegates to the
reusable modules under ``housing.*``. XCom carries the paths to artifacts
produced by the previous task so every stage stays independently testable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "mia-mlops-ii",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

PROCESSED_DIR = Path("/opt/airflow/project/data/processed")


def _ingest_task(**context):
    from housing.data.ingest import ingest_dataset

    result = ingest_dataset(upload_to_s3=True)
    context["ti"].xcom_push(key="raw_path", value=result["local_path"])
    return result


def _validate_task(**context):
    from housing.data.validate import validate_file

    raw_path = context["ti"].xcom_pull(task_ids="ingest", key="raw_path")
    report = validate_file(raw_path)
    if not report.passed:
        raise ValueError(f"Data validation failed: {report.errors}")
    context["ti"].xcom_push(key="report", value=json.dumps(report.to_dict()))
    return report.to_dict()


def _preprocess_task(**context):
    from housing.config import get_config
    from housing.features.preprocess import load_and_split, persist_splits

    raw_path = context["ti"].xcom_pull(task_ids="ingest", key="raw_path")
    cfg = get_config()
    splits = load_and_split(csv_path=raw_path, config=cfg)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = persist_splits(splits, PROCESSED_DIR)
    context["ti"].xcom_push(key="split_paths", value=json.dumps(paths))
    return paths


def _train_task(**context):
    import joblib
    from housing.config import get_config, get_settings
    from housing.features.preprocess import load_splits
    from housing.models.train import train_all

    cfg = get_config()
    settings = get_settings()
    paths = json.loads(context["ti"].xcom_pull(task_ids="preprocess", key="split_paths"))
    splits = load_splits(paths, target_col=settings.target_column)

    summary = train_all(splits=splits, config=cfg)

    # Also persist the best pipeline locally so the API has a fallback.
    import mlflow
    import mlflow.sklearn

    best_uri = f"runs:/{summary['best_run_id']}/model"
    pipeline = mlflow.sklearn.load_model(best_uri)
    local_model_path = Path(settings.local_model_path)
    local_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, local_model_path)

    context["ti"].xcom_push(key="best_run_id", value=summary["best_run_id"])
    context["ti"].xcom_push(key="best_model_name", value=summary["best_model_name"])
    return summary


def _register_task(**context):
    from housing.models.register import register_model

    run_id = context["ti"].xcom_pull(task_ids="train", key="best_run_id")
    version = register_model(run_id=run_id)
    return {"version": version, "run_id": run_id}


with DAG(
    dag_id="california_housing_training",
    description="End-to-end training pipeline for the California Housing regressor.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,  # trigger manually or from the UI
    catchup=False,
    tags=["mlops", "california-housing", "regression", "training"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=_ingest_task)
    validate = PythonOperator(task_id="validate", python_callable=_validate_task)
    preprocess = PythonOperator(task_id="preprocess", python_callable=_preprocess_task)
    train = PythonOperator(task_id="train", python_callable=_train_task)
    register = PythonOperator(task_id="register", python_callable=_register_task)

    ingest >> validate >> preprocess >> train >> register
