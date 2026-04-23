"""Smoke tests for the FastAPI application.

We intercept the model loading so the tests don't need a running MLflow
server. A real sklearn Pipeline is fitted on the synthetic dataframe and
injected into the ModelStore before making requests.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from housing.api import main as api_main
from housing.features.preprocess import (
    build_preprocessor,
    infer_feature_groups,
)
from housing.models.predict import ChampionModel

DUMMY_CONFIG = {"dataset": {"target_column": "target_value", "drop_columns": []}}


@pytest.fixture()
def client(monkeypatch, sample_dataframe: pd.DataFrame) -> TestClient:
    groups = infer_feature_groups(sample_dataframe, DUMMY_CONFIG)
    X = sample_dataframe[groups.all].copy()
    y = sample_dataframe["target_value"].astype(float)

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(groups)),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)

    stub = ChampionModel(
        inner=pipeline, metadata={"source": "stub", "name": "unit-test"}
    )
    api_main.store._model = stub

    # Bypass real model loading and feature_metadata.json lookup.
    monkeypatch.setattr(api_main.store, "load", lambda *a, **kw: True)
    monkeypatch.setattr(api_main, "_feature_names", lambda: groups.all)
    monkeypatch.setattr(api_main, "_read_metadata", lambda: None)

    with TestClient(api_main.app) as c:
        yield c


def _valid_record(sample_dataframe: pd.DataFrame) -> dict:
    return sample_dataframe.iloc[0].drop("target_value").to_dict()


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint(client: TestClient, sample_dataframe: pd.DataFrame) -> None:
    r = client.post("/predict", json={"features": _valid_record(sample_dataframe)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predicted_value" in body
    assert isinstance(body["predicted_value"], float)


def test_batch_predict(client: TestClient, sample_dataframe: pd.DataFrame) -> None:
    record = _valid_record(sample_dataframe)
    r = client.post("/predict/batch", json={"records": [record, record]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert all(isinstance(p["predicted_value"], float) for p in preds)


def test_missing_feature_returns_422(
    client: TestClient, sample_dataframe: pd.DataFrame
) -> None:
    record = _valid_record(sample_dataframe)
    record.pop(next(iter(record)))
    r = client.post("/predict", json={"features": record})
    assert r.status_code == 422
