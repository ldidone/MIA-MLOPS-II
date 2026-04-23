"""Create the MinIO buckets expected by the rest of the stack.

The ``minio-init`` compose service already does this via ``mc``; this script
is kept as a language-agnostic alternative for evaluators who may want to run
it outside of docker-compose.
"""

from __future__ import annotations

from housing.config import get_settings
from housing.data.ingest import ensure_bucket
from housing.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    for bucket in (settings.minio_data_bucket, settings.minio_mlflow_bucket):
        ensure_bucket(bucket)
        logger.info("Bucket ready: %s", bucket)


if __name__ == "__main__":
    main()
