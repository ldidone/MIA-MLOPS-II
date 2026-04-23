"""Uniform logging configuration for the whole project."""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

from housing.config import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _running_under_airflow() -> bool:
    # Airflow exports these env vars while a task is executing, and the
    # webserver/scheduler also set AIRFLOW_HOME. We must not attach a stdout
    # handler in that case: Airflow replaces ``sys.stdout`` with a writer that
    # pipes back into ``airflow.task``, which then propagates to the root
    # logger and re-feeds the same record → unbounded recursion.
    return any(
        os.environ.get(var)
        for var in ("AIRFLOW_CTX_DAG_ID", "AIRFLOW_HOME", "AIRFLOW__CORE__EXECUTOR")
    )


@lru_cache(maxsize=1)
def _configure_root() -> None:
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    if _running_under_airflow():
        # Let Airflow own logging configuration entirely; our records still
        # flow through ``airflow.task`` via standard propagation.
        return
    # Skip if anything (Uvicorn, pytest, manual setup) already configured
    # the root logger so we never double-emit.
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the shared formatter configured."""
    _configure_root()
    return logging.getLogger(name)
