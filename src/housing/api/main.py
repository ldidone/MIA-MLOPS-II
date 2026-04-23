"""FastAPI application serving the champion regressor."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from housing.api.model_loader import ModelStore
from housing.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from housing.config import get_config, get_settings
from housing.utils.feature_metadata import read_feature_metadata
from housing.utils.logging import get_logger

logger = get_logger(__name__)

store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API — loading champion model…")
    store.load()
    if not store.loaded:
        logger.warning("API starting without a model; /health will report degraded.")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="California Housing - Regression API",
    description=(
        "Production-style REST service that serves the champion regressor for "
        "scikit-learn's California Housing dataset. Loads the latest "
        "`@champion` model from the MLflow Registry, with a local pickle fallback."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_store(request: Request) -> ModelStore:  # noqa: ARG001 - uniform DI signature
    return store


def _read_metadata():
    settings = get_settings()
    cfg = get_config()
    meta_path = settings.project_root / cfg["paths"]["feature_metadata"]
    return read_feature_metadata(meta_path)


def _feature_names() -> list[str]:
    """Return the authoritative list of feature names the model was trained on."""
    meta = _read_metadata()
    if meta and meta.feature_names:
        return meta.feature_names
    # Fallback: if metadata is missing (first boot or evaluator), try to read
    # the column names the sklearn ColumnTransformer remembers.
    if store.loaded:
        cols = getattr(store.model._inner, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
        pre = getattr(store.model._inner, "named_steps", {}).get("preprocessor")
        if pre is not None and hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)
    return []


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = _feature_names()
    if not columns:
        return pd.DataFrame(records)
    missing = [
        [c for c in columns if c not in rec] for rec in records
    ]
    first_missing = next((m for m in missing if m), None)
    if first_missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "missing_features",
                "missing": first_missing,
                "expected_features": columns,
            },
        )
    return pd.DataFrame([{col: rec[col] for col in columns} for rec in records])


def _frame_to_predictions(model, df: pd.DataFrame) -> list[PredictionResponse]:
    preds = model.predict(df)
    return [PredictionResponse(predicted_value=float(p)) for p in preds]


@app.get("/", include_in_schema=False)
def root():
    settings = get_settings()
    return {
        "service": "california-housing-regressor-api",
        "version": app.version,
        "model_name": settings.model_name,
        "model_alias": settings.model_alias,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
    }


@app.get("/health", response_model=HealthResponse)
def health(store: ModelStore = Depends(_get_store)):
    if not store.loaded:
        return HealthResponse(status="degraded", model_loaded=False)
    source = store.model.metadata.get("source")
    return HealthResponse(status="ok", model_loaded=True, model_source=source)


@app.post("/admin/reload", status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
def reload_model(store: ModelStore = Depends(_get_store)):
    ok = store.reload()
    if not ok:
        raise HTTPException(status_code=503, detail="Model reload failed.")
    return {"reloaded": True, "metadata": store.model.metadata}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(store: ModelStore = Depends(_get_store)):
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    feature_names = _feature_names()
    meta = _read_metadata()
    target_stats = meta.target_stats if meta else None
    target = meta.target if meta else get_config()["dataset"]["target_column"]
    return ModelInfoResponse(
        metadata=store.model.metadata,
        target=target,
        feature_names=feature_names,
        n_features=len(feature_names),
        target_stats=target_stats or None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, store: ModelStore = Depends(_get_store)):
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    df = _records_to_frame([payload.features])
    try:
        return _frame_to_predictions(store.model, df)[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, store: ModelStore = Depends(_get_store)):
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    df = _records_to_frame(payload.records)
    try:
        predictions = _frame_to_predictions(store.model, df)
        return BatchPredictionResponse(predictions=predictions)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
