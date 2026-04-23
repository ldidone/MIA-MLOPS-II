"""Model loading with retries and graceful degradation.

The API is expected to start before training has finished (first boot of
docker-compose). This module keeps retrying the MLflow Registry, then falls
back to a local pickle if present, and finally exposes a ``loaded`` property
so the ``/health`` endpoint can reflect reality instead of crashing on import.
"""

from __future__ import annotations

import time
from threading import Lock

from housing.models.predict import ChampionModel, load_from_disk, load_from_registry
from housing.utils.logging import get_logger

logger = get_logger(__name__)


class ModelStore:
    """Thread-safe lazy holder for the champion model."""

    def __init__(self) -> None:
        self._model: ChampionModel | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> ChampionModel:
        if self._model is None:
            raise RuntimeError("Model is not loaded yet.")
        return self._model

    def load(self, retries: int = 3, delay_seconds: int = 5) -> bool:
        """Attempt to populate the store; returns whether it succeeded."""
        with self._lock:
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    self._model = load_from_registry()
                    logger.info("Model loaded from MLflow (attempt %d)", attempt)
                    return True
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "MLflow load attempt %d/%d failed: %s", attempt, retries, exc
                    )
                    time.sleep(delay_seconds)

            try:
                self._model = load_from_disk()
                logger.info("Model loaded from local pickle")
                return True
            except Exception as exc:
                logger.error(
                    "Could not load model. Last MLflow error: %s. Disk error: %s",
                    last_exc,
                    exc,
                )
                return False

    def reload(self) -> bool:
        self._model = None
        return self.load()
