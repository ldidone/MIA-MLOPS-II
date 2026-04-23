"""Register the best model in the MLflow Model Registry.

We use the modern *alias* API (``champion``) rather than the deprecated
``Stages``. The FastAPI service loads ``models:/<name>@champion`` and therefore
picks up new versions automatically once the alias is moved.
"""

from __future__ import annotations

import time

from mlflow.exceptions import MlflowException

from housing.config import get_settings
from housing.utils.logging import get_logger
from housing.utils.mlflow_utils import configure_mlflow, get_client

logger = get_logger(__name__)


def register_model(run_id: str, artifact_path: str = "model") -> str:
    """Register ``runs:/<run_id>/<artifact_path>`` and tag it as ``champion``.

    Returns the newly created model version.
    """
    configure_mlflow()
    settings = get_settings()
    client = get_client()

    # Ensure the registered model exists.
    try:
        client.create_registered_model(settings.model_name)
    except MlflowException as exc:
        if "RESOURCE_ALREADY_EXISTS" not in str(exc):
            raise

    source = f"runs:/{run_id}/{artifact_path}"
    mv = client.create_model_version(
        name=settings.model_name,
        source=source,
        run_id=run_id,
    )

    # Wait for the new version to leave the PENDING_REGISTRATION state.
    for _ in range(30):
        current = client.get_model_version(settings.model_name, mv.version)
        if current.status == "READY":
            break
        time.sleep(1)

    client.set_registered_model_alias(
        name=settings.model_name,
        alias=settings.model_alias,
        version=mv.version,
    )
    logger.info(
        "Registered %s version=%s alias=%s", settings.model_name, mv.version, settings.model_alias
    )
    return mv.version
