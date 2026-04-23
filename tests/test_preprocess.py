"""Smoke tests for the schema-agnostic preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from housing.features.preprocess import (
    build_preprocessor,
    infer_feature_groups,
)


def test_feature_inference_classifies_columns(sample_dataframe: pd.DataFrame) -> None:
    groups = infer_feature_groups(
        sample_dataframe,
        config={"dataset": {"target_column": "target_value", "drop_columns": []}},
    )
    assert "anxiety_score" in groups.numeric
    assert "gender" in groups.categorical
    assert "therapy_history" in groups.binary
    assert "target_value" not in groups.all


def test_preprocessor_fits_and_transforms(sample_dataframe: pd.DataFrame) -> None:
    groups = infer_feature_groups(
        sample_dataframe,
        config={"dataset": {"target_column": "target_value", "drop_columns": []}},
    )
    X = sample_dataframe[groups.all].copy()
    pre = build_preprocessor(groups)
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(X)
    # Categorical is one-hot expanded, so the output width is >= input width.
    assert transformed.shape[1] >= X.shape[1]


def test_preprocessor_handles_unseen_categoricals(sample_dataframe: pd.DataFrame) -> None:
    groups = infer_feature_groups(
        sample_dataframe,
        config={"dataset": {"target_column": "target_value", "drop_columns": []}},
    )
    X = sample_dataframe[groups.all].copy()
    pre = build_preprocessor(groups)
    pre.fit(X)

    test_row = X.iloc[[0]].copy()
    test_row["gender"] = "Alien"  # unseen
    result = pre.transform(test_row)
    assert result.shape[0] == 1
