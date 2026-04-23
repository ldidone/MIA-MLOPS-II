"""Run the full training pipeline without Airflow.

This is useful for:

* Rapid iteration while developing a new model family.
* Rehearsing the pipeline from inside the ``api`` container.
* The evaluator running the project without starting Airflow.

Usage::

    docker compose exec api python -m scripts.run_training_local

It performs ``ingest -> validate -> train -> register`` in a single process,
dropping to a local pickle so the API can pick the model up on reload.
"""

from __future__ import annotations

import argparse

import joblib
import mlflow
import mlflow.sklearn

from housing.config import get_config, get_settings
from housing.data.ingest import ingest_dataset
from housing.data.validate import validate_file
from housing.features.preprocess import load_and_split
from housing.models.register import register_model
from housing.models.train import train_all
from housing.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training pipeline locally.")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Re-use the CSV already sitting in data/raw/.",
    )
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Train and evaluate but do not move the champion alias.",
    )
    args = parser.parse_args()

    cfg = get_config()
    settings = get_settings()

    if not args.skip_ingest:
        logger.info("Ingesting dataset…")
        ingest_dataset(upload_to_s3=True)

    raw_path = settings.project_root / cfg["paths"]["raw_csv"]
    logger.info("Validating %s", raw_path)
    report = validate_file(raw_path)
    if not report.passed:
        raise SystemExit(f"Data validation failed: {report.errors}")

    splits = load_and_split(csv_path=raw_path, config=cfg)
    summary = train_all(splits=splits, config=cfg)
    logger.info("Training summary: %s", summary)

    # Mirror to local pickle so the API can load it offline.
    best_uri = f"runs:/{summary['best_run_id']}/model"
    pipeline = mlflow.sklearn.load_model(best_uri)
    local_model = settings.project_root / "models" / "model.pkl"
    local_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, local_model)
    logger.info("Saved champion pipeline to %s", local_model)

    if not args.skip_register:
        version = register_model(summary["best_run_id"])
        logger.info("Registered version=%s as %s", version, settings.model_alias)


if __name__ == "__main__":
    main()
