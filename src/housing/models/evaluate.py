"""Evaluation utilities: regression metrics + diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend, required inside containers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return the canonical set of regression metrics we track.

    * ``r2``      — coefficient of determination, higher is better (max 1.0).
    * ``mae``     — mean absolute error, in target units.
    * ``rmse``    — root mean squared error, penalises big misses harder.
    * ``mape``    — mean absolute percentage error (fraction, not %).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mape": float(mean_absolute_percentage_error(y_true_arr, y_pred_arr)),
    }


def regression_report_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Return a JSON-friendly diagnostic report for regression predictions."""
    metrics = compute_metrics(y_true, y_pred)
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        **metrics,
        "residuals": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "q25": float(np.quantile(residuals, 0.25)),
            "q50": float(np.quantile(residuals, 0.50)),
            "q75": float(np.quantile(residuals, 0.75)),
        },
        "n_samples": int(len(y_true)),
    }


def plot_predicted_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_path: Path | str,
    title: str = "Predicted vs actual",
) -> Path:
    """Scatter plot of predictions vs ground truth, with the y=x reference."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true_arr, y_pred_arr, alpha=0.25, s=12, color="steelblue")
    lo = float(min(y_true_arr.min(), y_pred_arr.min()))
    hi = float(max(y_true_arr.max(), y_pred_arr.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_path: Path | str,
    title: str = "Residuals vs predicted",
) -> Path:
    """Residual plot (y_true - y_pred vs y_pred)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred_arr, residuals, alpha=0.25, s=12, color="seagreen")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
