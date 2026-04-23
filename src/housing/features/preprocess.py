"""Feature engineering and dataset splitting.

The preprocessing is expressed as a scikit-learn :class:`ColumnTransformer` so
it can be embedded directly inside every trained estimator ``Pipeline``. This
guarantees bit-for-bit consistency between training and inference: the same
``ColumnTransformer`` instance is shipped in the MLflow artifact and applied at
serving time by FastAPI.

Feature groups (numeric / categorical / binary) can be either configured
explicitly in ``conf/config.yaml`` **or** auto-inferred from the dataframe.
When the lists in the config are empty we fall back to inference, which keeps
the pipeline working even if the upstream dataset schema changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from housing.config import get_config, get_settings
from housing.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureGroups:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return list(self.numeric) + list(self.categorical) + list(self.binary)

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_groups: FeatureGroups | None = None

    def __iter__(self):
        return iter(
            (self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test)
        )


def _columns_to_exclude(config: dict[str, Any]) -> set[str]:
    target = config["dataset"]["target_column"]
    drops = set(config["dataset"].get("drop_columns") or [])
    return drops | {target}


def infer_feature_groups(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> FeatureGroups:
    """Classify each column into numeric / categorical / binary.

    Rules:

    * Drop columns listed in ``dataset.drop_columns`` and the target.
    * Binary:       numeric/bool column with exactly 2 unique non-null values,
                    both in ``{0, 1, True, False}``.
    * Numeric:      any other numeric (int or float) column.
    * Categorical:  everything else (object, string, category).
    """
    cfg = config or get_config()
    excluded = _columns_to_exclude(cfg)

    numeric: list[str] = []
    categorical: list[str] = []
    binary: list[str] = []

    for col in df.columns:
        if col in excluded:
            continue
        series = df[col]
        if is_bool_dtype(series):
            binary.append(col)
            continue
        if is_numeric_dtype(series):
            uniq = set(pd.unique(series.dropna()))
            if uniq.issubset({0, 1}) and len(uniq) <= 2:
                binary.append(col)
            else:
                numeric.append(col)
        else:
            categorical.append(col)

    groups = FeatureGroups(numeric=numeric, categorical=categorical, binary=binary)
    logger.info(
        "Inferred feature groups: numeric=%d categorical=%d binary=%d",
        len(groups.numeric),
        len(groups.categorical),
        len(groups.binary),
    )
    return groups


def resolve_feature_groups(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> FeatureGroups:
    """Return the feature groups to use, honouring overrides from the config."""
    cfg = config or get_config()
    feats = cfg.get("features") or {}
    cfg_num = list(feats.get("numeric") or [])
    cfg_cat = list(feats.get("categorical") or [])
    cfg_bin = list(feats.get("binary") or [])
    if cfg_num or cfg_cat or cfg_bin:
        groups = FeatureGroups(numeric=cfg_num, categorical=cfg_cat, binary=cfg_bin)
    else:
        groups = infer_feature_groups(df, cfg)
    # Make sure every inferred column actually exists in the dataframe.
    present = set(df.columns)
    groups.numeric = [c for c in groups.numeric if c in present]
    groups.categorical = [c for c in groups.categorical if c in present]
    groups.binary = [c for c in groups.binary if c in present]
    return groups


def build_preprocessor(feature_groups: FeatureGroups) -> ColumnTransformer:
    """Return the :class:`ColumnTransformer` used by every model Pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    transformers = []
    if feature_groups.numeric:
        transformers.append(("num", numeric_pipeline, feature_groups.numeric))
    if feature_groups.categorical:
        transformers.append(("cat", categorical_pipeline, feature_groups.categorical))
    if feature_groups.binary:
        transformers.append(("bin", binary_pipeline, feature_groups.binary))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def _drop_leakage(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    drop = [c for c in (config["dataset"].get("drop_columns") or []) if c in df.columns]
    if drop:
        logger.info("Dropping leakage columns: %s", drop)
        return df.drop(columns=drop)
    return df


def load_and_split(
    csv_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Splits:
    """Load the CSV and produce random train/val/test splits for regression.

    The target is kept as a float; no stratification is applied because the
    target is continuous.
    """
    settings = get_settings()
    cfg = config or get_config()

    path = Path(csv_path) if csv_path else (settings.project_root / cfg["paths"]["raw_csv"])
    df = pd.read_csv(path)
    df = _drop_leakage(df, cfg)

    target = cfg["dataset"]["target_column"]
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)}"
        )

    groups = resolve_feature_groups(df, cfg)
    feature_cols = groups.all
    if not feature_cols:
        raise ValueError("No feature columns were resolved; check the config.")

    X = df[feature_cols].copy()
    y = df[target].astype(float)

    split_cfg = cfg["split"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    val_relative = split_cfg["val_size"] / (1.0 - split_cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=split_cfg["random_state"],
    )

    logger.info(
        "Splits: train=%d val=%d test=%d (target range=[%.3f, %.3f])",
        len(X_train),
        len(X_val),
        len(X_test),
        float(y.min()),
        float(y.max()),
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_groups=groups)


def persist_splits(splits: Splits, out_dir: Path | str) -> dict[str, str]:
    """Write splits to parquet so Airflow tasks can communicate via the FS."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, obj in {
        "X_train": splits.X_train,
        "X_val": splits.X_val,
        "X_test": splits.X_test,
        "y_train": splits.y_train.to_frame(),
        "y_val": splits.y_val.to_frame(),
        "y_test": splits.y_test.to_frame(),
    }.items():
        p = out / f"{name}.parquet"
        obj.to_parquet(p, index=False)
        paths[name] = str(p)
    # Persist the resolved feature groups so downstream tasks can recreate
    # the exact same preprocessor.
    if splits.feature_groups is not None:
        (out / "feature_groups.json").write_text(
            json.dumps(splits.feature_groups.to_dict(), indent=2)
        )
        paths["feature_groups"] = str(out / "feature_groups.json")
    logger.info("Persisted splits to %s", out)
    return paths


def load_splits(paths: dict[str, str], target_col: str) -> Splits:
    """Inverse of :func:`persist_splits`."""
    X_train = pd.read_parquet(paths["X_train"])
    X_val = pd.read_parquet(paths["X_val"])
    X_test = pd.read_parquet(paths["X_test"])
    y_train = pd.read_parquet(paths["y_train"])[target_col]
    y_val = pd.read_parquet(paths["y_val"])[target_col]
    y_test = pd.read_parquet(paths["y_test"])[target_col]
    groups: FeatureGroups | None = None
    fg_path = paths.get("feature_groups")
    if fg_path and Path(fg_path).exists():
        data = json.loads(Path(fg_path).read_text())
        groups = FeatureGroups(**data)
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_groups=groups)
