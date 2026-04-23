"""Thin wrappers around model loading for inference.

Two loading strategies are supported:

* MLflow Model Registry alias (preferred when running inside docker-compose).
* Local joblib/pickle fallback (useful for tests and for the Streamlit Space
  deployment where no MLflow server is available).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from housing.config import get_settings
from housing.utils.logging import get_logger
from housing.utils.mlflow_utils import configure_mlflow, model_uri

logger = get_logger(__name__)


class ChampionModel:
    """Thin wrapper exposing :py:meth:`predict` for a regression pipeline.

    Works whether the underlying object is a fitted sklearn Pipeline loaded via
    ``mlflow.sklearn.load_model`` or a generic ``pyfunc`` model.
    """

    def __init__(self, inner: Any, metadata: dict[str, Any]):
        self._inner = inner
        self.metadata = metadata

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._inner.predict(df), dtype=float)


def load_from_registry() -> ChampionModel:
    """Load the current champion from the MLflow Registry."""
    configure_mlflow()
    settings = get_settings()
    uri = model_uri()
    logger.info("Loading model from MLflow: %s", uri)
    model = mlflow.sklearn.load_model(uri)

    # Fetch version metadata for /model/info.
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
        metadata = {
            "source": "mlflow",
            "name": settings.model_name,
            "alias": settings.model_alias,
            "version": mv.version,
            "run_id": mv.run_id,
            "status": mv.status,
        }
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Could not fetch model metadata: %s", exc)
        metadata = {"source": "mlflow", "uri": uri}

    return ChampionModel(inner=model, metadata=metadata)


def load_from_disk(path: str | Path | None = None) -> ChampionModel:
    """Load a joblib/pickle file as a last-resort fallback."""
    settings = get_settings()
    p = Path(path or settings.local_model_path)
    logger.info("Loading model from disk: %s", p)
    model = joblib.load(p)
    return ChampionModel(
        inner=model,
        metadata={"source": "local", "path": str(p)},
    )


def load_model() -> ChampionModel:
    """Load the champion with graceful fallback to the local pickle file."""
    settings = get_settings()
    if settings.app_mode.lower() == "embedded":
        return load_from_disk()
    try:
        return load_from_registry()
    except Exception as exc:
        logger.warning("MLflow load failed (%s). Falling back to disk.", exc)
    return load_from_disk()
