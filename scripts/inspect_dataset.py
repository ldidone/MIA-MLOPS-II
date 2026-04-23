"""Quickly inspect the ingested dataset.

Runs ingestion (idempotent) and prints the schema so you can verify that the
pipeline is pointing at the right dataset and confirm the target column is
present and well-behaved for regression.

Usage::

    docker compose exec api python -m scripts.inspect_dataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing.config import get_config, get_settings
from housing.data.ingest import ingest_dataset
from housing.features.preprocess import infer_feature_groups
from housing.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = get_config()
    settings = get_settings()

    raw_csv = Path(settings.project_root) / cfg["paths"]["raw_csv"]
    if not raw_csv.exists():
        logger.info("Raw CSV missing, running ingestion…")
        ingest_dataset(upload_to_s3=False)

    df = pd.read_csv(raw_csv)
    print()
    print("=" * 72)
    print(f"Dataset: {raw_csv}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print("=" * 72)
    print()

    dtype_table = (
        pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "n_unique": df.nunique(dropna=True),
                "n_null": df.isna().sum(),
                "sample": [df[c].dropna().head(3).tolist() for c in df.columns],
            }
        )
        .reset_index()
        .rename(columns={"index": "column"})
    )
    print(dtype_table.to_string(index=False))

    print()
    target = cfg["dataset"]["target_column"]
    print(f"Target column in config: '{target}'")
    if target not in df.columns:
        print(f"  !! '{target}' is NOT in the dataset columns.")
        print("     Pick another target and update conf/config.yaml.")
    else:
        ts = df[target].describe()
        print("  -> Target summary statistics:")
        print(ts.to_string())
        print()
        print("  -> Pearson correlation of every numeric feature with the target:")
        numeric = df.select_dtypes(include="number")
        if target in numeric.columns:
            corr = numeric.corr()[target].drop(labels=[target]).sort_values(
                ascending=False, key=lambda s: s.abs()
            )
            print(corr.round(3).to_string())

    groups = infer_feature_groups(df, cfg)
    print()
    print("Inferred feature groups:")
    print(f"  numeric     ({len(groups.numeric):2d}): {groups.numeric}")
    print(f"  categorical ({len(groups.categorical):2d}): {groups.categorical}")
    print(f"  binary      ({len(groups.binary):2d}): {groups.binary}")


if __name__ == "__main__":
    main()
