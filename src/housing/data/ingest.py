"""Dataset ingestion.

The California Housing dataset ships with scikit-learn; we fetch it via
``sklearn.datasets.fetch_california_housing`` (which caches under
``~/scikit_learn_data``) and persist two copies:

* A local CSV under ``data/raw/california_housing.csv`` for quick iteration
  and tests.
* The same CSV uploaded to the S3-compatible data lake (MinIO) so that the
  rest of the pipeline can treat the data lake as the source of truth.

No external credentials are required.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pandas as pd
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from sklearn.datasets import fetch_california_housing

from housing.config import get_config, get_settings
from housing.utils.logging import get_logger

logger = get_logger(__name__)


def _s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_default_region,
        config=BotoConfig(signature_version="s3v4"),
    )


def ensure_bucket(bucket: str) -> None:
    client = _s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        logger.info("Creating bucket %s", bucket)
        client.create_bucket(Bucket=bucket)


def upload_file(local_path: Path, bucket: str, key: str) -> str:
    ensure_bucket(bucket)
    client = _s3_client()
    client.upload_file(str(local_path), bucket, key)
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded %s -> %s", local_path, uri)
    return uri


def _fetch_california_housing_df(target_column: str) -> pd.DataFrame:
    """Fetch the California Housing dataset and return a single DataFrame.

    scikit-learn exposes the features as ``frame`` when ``as_frame=True``; we
    just add the configured target column name so downstream code stays
    dataset-agnostic.
    """
    logger.info("Fetching California Housing dataset from scikit-learn")
    bundle = fetch_california_housing(as_frame=True)
    df: pd.DataFrame = bundle.frame.copy()
    # sklearn's default target column name is "MedHouseVal"; rename if the
    # user configured something else.
    if "MedHouseVal" in df.columns and target_column != "MedHouseVal":
        df = df.rename(columns={"MedHouseVal": target_column})
    return df


def ingest_dataset(upload_to_s3: bool = True) -> dict[str, str]:
    """Fetch the dataset, persist locally, and optionally mirror to MinIO."""
    settings = get_settings()
    config = get_config()

    target_column = config["dataset"]["target_column"]
    local_csv = (settings.project_root / config["paths"]["raw_csv"]).resolve()
    local_csv.parent.mkdir(parents=True, exist_ok=True)

    df = _fetch_california_housing_df(target_column=target_column)
    df.to_csv(local_csv, index=False)
    logger.info(
        "Dataset loaded: rows=%d cols=%d columns=%s",
        len(df),
        df.shape[1],
        list(df.columns),
    )

    result: dict[str, str] = {"local_path": str(local_csv)}

    if upload_to_s3:
        try:
            result["s3_uri"] = upload_file(
                local_path=local_csv,
                bucket=settings.minio_data_bucket,
                key="raw/california_housing.csv",
            )
        except (BotoCoreError, ClientError, OSError) as exc:
            logger.warning("S3 upload skipped (%s). Local file still available.", exc)

    return result


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    ingest_dataset()
