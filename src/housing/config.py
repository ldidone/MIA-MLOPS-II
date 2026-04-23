"""Project configuration.

Two layers of configuration are combined:

1. **Static config** (``conf/config.yaml``): feature lists, model hyper-parameters,
   CV folds, etc. It does not change between environments.
2. **Runtime settings** (:class:`Settings`, ``pydantic-settings``): environment
   dependent values like MLflow/MinIO URLs, credentials and logging levels.

Both are exposed through :func:`get_settings` and :func:`get_config` to keep the
rest of the codebase decoupled from how configuration is sourced.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


class Settings(BaseSettings):
    """Environment-driven runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000")
    mlflow_experiment_name: str = Field(default="california_housing_regression")
    model_name: str = Field(default="california_housing_regressor")
    model_alias: str = Field(default="champion")

    # S3 / MinIO
    s3_endpoint_url: str = Field(default="http://minio:9000")
    mlflow_s3_endpoint_url: str = Field(default="http://minio:9000")
    aws_access_key_id: str = Field(default="minioadmin")
    aws_secret_access_key: str = Field(default="minioadmin")
    aws_default_region: str = Field(default="us-east-1")
    minio_data_bucket: str = Field(default="data-lake")
    minio_mlflow_bucket: str = Field(default="mlflow-artifacts")

    # Dataset
    target_column: str = Field(default="MedHouseVal")

    # API / app
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_url: str = Field(default="http://api:8000")
    app_mode: str = Field(default="api")  # "api" or "embedded"
    local_model_path: str = Field(default=str(PROJECT_ROOT / "models" / "model.pkl"))

    # Misc
    log_level: str = Field(default="INFO")
    project_root: Path = Field(default=PROJECT_ROOT)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings()


@lru_cache(maxsize=1)
def get_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the static YAML configuration.

    Parameters
    ----------
    path:
        Optional override of the config file path (useful for tests).
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def feature_columns(config: dict[str, Any] | None = None) -> list[str]:
    """Return the ordered list of *explicitly configured* feature columns.

    When the feature lists in ``conf/config.yaml`` are empty (the default) the
    authoritative column list is derived from the dataframe at runtime by
    :func:`housing.features.preprocess.resolve_feature_groups`, and
    persisted alongside the trained model.
    """
    cfg = config or get_config()
    features = cfg.get("features") or {}
    return (
        list(features.get("numeric") or [])
        + list(features.get("categorical") or [])
        + list(features.get("binary") or [])
    )
