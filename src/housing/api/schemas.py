"""Pydantic request / response schemas for the FastAPI service (regression).

The concrete column layout of the dataset is only known at training time, so
the request payload is intentionally kept as a permissive ``dict[str, Any]``.
At runtime the API still performs two-layer validation:

1. Pydantic enforces the *envelope* (single record or batch, basic types).
2. The model loader re-indexes the incoming dict against the list of feature
   names persisted alongside the champion (``feature_metadata.json``) and
   raises a clear HTTP 422 if a required feature is missing.

This keeps the API usable regardless of the dataset's column names while
still giving callers a meaningful error when they forget a feature.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """A single observation to score.

    ``features`` is a free-form dict mapping ``column_name -> value``. The
    exact set of keys is discovered at train time and surfaced via the
    ``/model/info`` endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Mapping of feature name to value. Keys must match the columns "
            "the model was trained on (see /model/info for the exact list)."
        ),
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    predicted_value: float = Field(
        ...,
        description=(
            "Model prediction for the target variable. Units match the "
            "dataset's target column (see /model/info.target)."
        ),
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    # `model_*` fields collide with pydantic's protected namespace; relax it.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_source: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    metadata: dict
    task: Literal["regression"] = "regression"
    target: str | None = None
    feature_names: list[str] = Field(default_factory=list)
    n_features: int
    target_stats: dict[str, float] | None = None
