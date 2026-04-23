"""Lightweight, schema-agnostic data validation.

Runs a small set of checks that hold for any tabular regression dataset:

* Basic shape (non-empty, minimum number of rows).
* The configured target column is present, non-null and numeric.
* No column is 100% null (completely useless feature).
* Report summary statistics of the target so downstream tasks can sanity-check.

The report is returned as a dataclass so it can be both logged as an MLflow
artifact and used by the Airflow task to decide whether to short-circuit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from housing.config import get_config, get_settings
from housing.utils.logging import get_logger

logger = get_logger(__name__)

MIN_ROWS = 100


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    fully_null_columns: list[str] = field(default_factory=list)
    target_stats: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _load(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> ValidationReport:
    settings = get_settings()
    cfg = get_config()
    target = cfg["dataset"].get("target_column", settings.target_column)

    report = ValidationReport(
        n_rows=len(df), n_cols=df.shape[1], columns=list(df.columns)
    )

    if len(df) < MIN_ROWS:
        report.errors.append(f"Dataset has only {len(df)} rows (<{MIN_ROWS})")
        report.passed = False

    if target not in df.columns:
        report.errors.append(
            f"Target column '{target}' missing. Columns: {list(df.columns)}"
        )
        report.passed = False

    null_counts = df.isna().sum()
    report.null_counts = {
        col: int(null_counts[col]) for col in df.columns if null_counts[col] > 0
    }
    report.fully_null_columns = [
        col for col in df.columns if null_counts[col] == len(df)
    ]
    if report.fully_null_columns:
        report.errors.append(f"Fully-null columns: {report.fully_null_columns}")
        report.passed = False

    if target in df.columns:
        col = df[target]
        if col.isna().any():
            report.errors.append(f"Target '{target}' contains null values")
            report.passed = False
        elif not is_numeric_dtype(col):
            report.errors.append(
                f"Target '{target}' must be numeric for regression (dtype={col.dtype})"
            )
            report.passed = False
        else:
            report.target_stats = {
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std()),
                "median": float(col.median()),
            }
            if float(col.std()) == 0.0:
                report.errors.append(
                    f"Target '{target}' has zero variance (all values equal)"
                )
                report.passed = False

    logger.info(
        "Validation report: rows=%d cols=%d passed=%s errors=%d warnings=%d",
        report.n_rows,
        report.n_cols,
        report.passed,
        len(report.errors),
        len(report.warnings),
    )
    return report


def validate_file(path: str | Path) -> ValidationReport:
    df = _load(path)
    return validate(df)
