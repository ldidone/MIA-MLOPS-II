"""Read/write helpers for the feature metadata JSON file.

The file is written at the end of training and consumed at inference time by:

* The FastAPI service, to validate incoming JSON payloads against the exact
  column list the estimator was trained on.
* The Streamlit frontend, to render input widgets dynamically (so the UI keeps
  working even if the underlying dataset schema changes).

We keep a small per-column summary (kind, min/max, categories, default) so the
UI can render sensible widgets without having to re-read the raw dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from housing.features.preprocess import FeatureGroups

FeatureKind = str  # "numeric" | "categorical" | "binary"


@dataclass
class ColumnSpec:
    name: str
    kind: FeatureKind
    default: Any = None
    min: float | None = None
    max: float | None = None
    categories: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureMetadata:
    target: str
    feature_groups: FeatureGroups
    columns: list[ColumnSpec] = field(default_factory=list)
    target_stats: dict[str, float] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return self.feature_groups.all

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "feature_groups": self.feature_groups.to_dict(),
            "columns": [c.to_dict() for c in self.columns],
            "target_stats": {k: _jsonify(v) for k, v in self.target_stats.items()},
        }


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _describe_column(series: pd.Series, kind: FeatureKind) -> ColumnSpec:
    name = str(series.name)
    clean = series.dropna()
    if kind == "numeric":
        if len(clean):
            return ColumnSpec(
                name=name,
                kind=kind,
                default=float(clean.median()),
                min=float(clean.min()),
                max=float(clean.max()),
            )
        return ColumnSpec(name=name, kind=kind, default=0.0)
    if kind == "binary":
        return ColumnSpec(name=name, kind=kind, default=0, categories=[0, 1])
    # categorical
    cats = [
        _jsonify(v) for v in clean.astype(str).value_counts().head(50).index.tolist()
    ]
    return ColumnSpec(
        name=name,
        kind=kind,
        default=cats[0] if cats else "",
        categories=cats,
    )


def build_column_specs(
    df: pd.DataFrame, feature_groups: FeatureGroups
) -> list[ColumnSpec]:
    specs: list[ColumnSpec] = []
    for col in feature_groups.numeric:
        if col in df:
            specs.append(_describe_column(df[col], "numeric"))
    for col in feature_groups.binary:
        if col in df:
            specs.append(_describe_column(df[col], "binary"))
    for col in feature_groups.categorical:
        if col in df:
            specs.append(_describe_column(df[col], "categorical"))
    return specs


def write_feature_metadata(
    path: str | Path,
    feature_groups: FeatureGroups,
    target: str,
    target_stats: dict[str, float] | None = None,
    reference_df: pd.DataFrame | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns: list[ColumnSpec] = []
    if reference_df is not None:
        columns = build_column_specs(reference_df, feature_groups)
    payload = FeatureMetadata(
        target=target,
        feature_groups=feature_groups,
        columns=columns,
        target_stats=dict(target_stats or {}),
    ).to_dict()
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


def read_feature_metadata(path: str | Path) -> FeatureMetadata | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    groups = FeatureGroups(**data.get("feature_groups", {}))
    cols = [ColumnSpec(**c) for c in data.get("columns", [])]
    return FeatureMetadata(
        target=data.get("target", ""),
        feature_groups=groups,
        columns=cols,
        target_stats=data.get("target_stats", {}),
    )
