"""Validate that the flexible Pydantic schemas accept / reject as expected."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from housing.api.schemas import (
    BatchPredictionRequest,
    PredictionRequest,
)


def test_prediction_request_accepts_arbitrary_features() -> None:
    req = PredictionRequest(features={"age": 17, "gender": "Female", "flag": 1})
    assert req.features["age"] == 17


def test_prediction_request_requires_features_key() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest()  # type: ignore[call-arg]


def test_prediction_request_forbids_extra_top_level_keys() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(features={"age": 17}, extra_top_level=1)  # type: ignore[call-arg]


def test_batch_request_min_length() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionRequest(records=[])


def test_batch_request_accepts_records() -> None:
    req = BatchPredictionRequest(records=[{"a": 1}, {"a": 2}])
    assert len(req.records) == 2
