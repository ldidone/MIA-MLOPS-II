"""Training entry point (regression).

For each model declared in ``conf/config.yaml`` we:

1. Build a ``Pipeline(preprocessor, estimator)``.
2. Run K-fold cross-validation on train+val to estimate generalisation.
3. Refit on the concatenation of train+val and evaluate on the held-out test.
4. Log parameters, metrics and the fitted pipeline to MLflow as a nested run.

The function returns the MLflow parent run ID plus a summary of all child runs,
which the Airflow downstream task uses to pick a champion.

Metric conventions
------------------
The primary CV score is **R²** (configured via ``training.scoring: r2``), which
is what we use to rank champions. We additionally log MAE, RMSE, MAPE and a
diagnostic residual summary as test metrics.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from housing.config import get_config, get_settings
from housing.features.preprocess import (
    FeatureGroups,
    Splits,
    build_preprocessor,
    load_and_split,
)
from housing.models.evaluate import (
    compute_metrics,
    plot_predicted_vs_actual,
    plot_residuals,
    regression_report_dict,
)
from housing.utils.feature_metadata import write_feature_metadata
from housing.utils.logging import get_logger
from housing.utils.mlflow_utils import configure_mlflow, ensure_experiment

logger = get_logger(__name__)


ESTIMATOR_FACTORIES: dict[str, Any] = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
}


def _build_pipeline(
    name: str,
    params: dict[str, Any],
    feature_groups: FeatureGroups,
) -> Pipeline:
    factory = ESTIMATOR_FACTORIES[name]
    estimator = factory(**dict(params))
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_groups)),
            ("regressor", estimator),
        ]
    )


def _log_dict_artifact(obj: dict, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(json.dumps(obj, indent=2, default=str))
        mlflow.log_artifact(str(path))


def _train_single(
    name: str,
    model_cfg: dict[str, Any],
    splits: Splits,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train one model as a nested MLflow run and return a summary."""
    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tag("model_family", name)
        mlflow.set_tag("task", "regression")
        mlflow.log_params({f"model.{k}": v for k, v in model_cfg["params"].items()})
        mlflow.log_param("cv_folds", config["training"]["cv_folds"])
        mlflow.log_param("cv_scoring", config["training"]["scoring"])

        feature_groups = splits.feature_groups or FeatureGroups(
            numeric=list(splits.X_train.columns)
        )
        pipeline = _build_pipeline(name, model_cfg["params"], feature_groups)

        X_fit = pd.concat([splits.X_train, splits.X_val], ignore_index=True)
        y_fit = pd.concat([splits.y_train, splits.y_val], ignore_index=True)

        pipeline.fit(X_fit, y_fit)
        preds = pipeline.predict(splits.X_test)

        cv = KFold(
            n_splits=config["training"]["cv_folds"], shuffle=True, random_state=42
        )
        cv_scores = cross_val_score(
            pipeline,
            X_fit,
            y_fit,
            cv=cv,
            scoring=config["training"]["scoring"],
            n_jobs=-1,
        )

        test_metrics = compute_metrics(splits.y_test, preds)
        metrics = {
            "cv_score_mean": float(cv_scores.mean()),
            "cv_score_std": float(cv_scores.std()),
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        mlflow.log_metrics(metrics)

        report = regression_report_dict(splits.y_test, preds)
        _log_dict_artifact(report, "regression_report.json")

        with tempfile.TemporaryDirectory() as tmp:
            pa_path = Path(tmp) / "predicted_vs_actual.png"
            plot_predicted_vs_actual(
                splits.y_test, preds, pa_path, title=f"Predicted vs actual — {name}"
            )
            res_path = Path(tmp) / "residuals.png"
            plot_residuals(
                splits.y_test, preds, res_path, title=f"Residuals — {name}"
            )
            mlflow.log_artifact(str(pa_path))
            mlflow.log_artifact(str(res_path))

        signature = infer_signature(splits.X_train, pipeline.predict(splits.X_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=splits.X_train.head(3),
        )

        logger.info("Trained %s: metrics=%s", name, metrics)
        return {
            "run_id": run.info.run_id,
            "model_name": name,
            "metrics": metrics,
        }


def train_all(
    splits: Splits | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train every model defined in ``training.models`` and return a summary."""
    configure_mlflow()
    cfg = config or get_config()
    settings = get_settings()
    ensure_experiment(settings.mlflow_experiment_name)

    if splits is None:
        splits = load_and_split(config=cfg)

    # Persist feature metadata alongside the model so the API can build a
    # DataFrame with the exact columns the estimator was trained on.
    feature_groups = splits.feature_groups
    all_y = pd.concat([splits.y_train, splits.y_val, splits.y_test])
    target_stats = {
        "min": float(all_y.min()),
        "max": float(all_y.max()),
        "mean": float(all_y.mean()),
        "std": float(all_y.std()),
        "median": float(all_y.median()),
    }
    metadata_path = settings.project_root / cfg["paths"]["feature_metadata"]
    if feature_groups is not None:
        write_feature_metadata(
            path=metadata_path,
            feature_groups=feature_groups,
            target=cfg["dataset"]["target_column"],
            target_stats=target_stats,
            reference_df=splits.X_train,
        )

    results: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="training_pipeline") as parent:
        mlflow.set_tag("pipeline", "train_all")
        mlflow.set_tag("task", "regression")
        mlflow.log_param("n_models", len(cfg["training"]["models"]))
        mlflow.log_param("train_rows", len(splits.X_train))
        mlflow.log_param("val_rows", len(splits.X_val))
        mlflow.log_param("test_rows", len(splits.X_test))
        mlflow.log_param("target_column", cfg["dataset"]["target_column"])

        for name, model_cfg in cfg["training"]["models"].items():
            results.append(_train_single(name, model_cfg, splits, cfg))

        # Pick champion by CV score (configured scorer — r2 by default).
        best = max(results, key=lambda r: r["metrics"]["cv_score_mean"])
        mlflow.log_metric("best_cv_score", best["metrics"]["cv_score_mean"])
        mlflow.set_tag("best_model_run_id", best["run_id"])
        mlflow.set_tag("best_model_name", best["model_name"])

        if metadata_path.exists():
            mlflow.log_artifact(str(metadata_path))

        summary = {
            "parent_run_id": parent.info.run_id,
            "runs": results,
            "best_run_id": best["run_id"],
            "best_model_name": best["model_name"],
        }

    return summary
