"""Helpers around the MLflow client.

Centralising this keeps environment-variable wiring and tracking URI logic in
one place, so the rest of the codebase simply does::

    from housing.utils.mlflow_utils import configure_mlflow, get_client

    configure_mlflow()
    client = get_client()
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlflow
from mlflow.tracking import MlflowClient

from housing.config import get_settings
from housing.utils.logging import get_logger

logger = get_logger(__name__)


def configure_mlflow() -> None:
    """Set tracking URI and required S3/MinIO env vars for MLflow."""
    settings = get_settings()

    # MLflow relies on standard AWS_* env vars to talk to MinIO. We set them
    # here so that both the tracking server and the client agree on the
    # artifact endpoint.
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
    os.environ.setdefault("AWS_DEFAULT_REGION", settings.aws_default_region)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.debug("Configured MLflow tracking URI=%s", settings.mlflow_tracking_uri)


@lru_cache(maxsize=1)
def get_client() -> MlflowClient:
    """Return a cached :class:`MlflowClient` pointing at the tracking server."""
    configure_mlflow()
    return MlflowClient()


def ensure_experiment(name: str | None = None) -> str:
    """Create the experiment if it does not exist and return its ID."""
    settings = get_settings()
    experiment_name = name or settings.mlflow_experiment_name
    configure_mlflow()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info("Creating MLflow experiment %s", experiment_name)
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def model_uri(stage_or_alias: str | None = None) -> str:
    """Build a ``models:/<name>@<alias>`` URI for the registered model."""
    settings = get_settings()
    alias = stage_or_alias or settings.model_alias
    return f"models:/{settings.model_name}@{alias}"
