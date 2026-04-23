"""Streamlit frontend for the California Housing regressor.

Two operating modes controlled by ``APP_MODE``:

* ``api`` (default, used inside docker-compose): sends JSON payloads to the
  FastAPI service via :class:`APIClient` and reads the feature schema from
  ``/model/info``.
* ``embedded`` (used when deploying to Hugging Face Spaces as a single Docker
  image): loads the champion model and the feature metadata JSON directly.

The form is generated dynamically from the feature metadata so the UI keeps
working regardless of which columns the dataset ships with.

Run locally with::

    streamlit run src/housing/streamlit_app/app.py
"""

from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd
import streamlit as st

from housing.config import get_config, get_settings
from housing.streamlit_app.api_client import APIClient

st.set_page_config(
    page_title="California Housing — Regressor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def _get_model():
    from housing.models.predict import load_model

    return load_model()


@st.cache_resource
def _get_client(api_url: str) -> APIClient:
    return APIClient(api_url)


def _current_mode() -> str:
    return os.getenv("APP_MODE", get_settings().app_mode).lower()


def _read_local_metadata_full():
    from housing.utils.feature_metadata import read_feature_metadata

    settings = get_settings()
    cfg = get_config()
    meta_path = settings.project_root / cfg["paths"]["feature_metadata"]
    return read_feature_metadata(meta_path)


def _read_local_metadata() -> list[dict[str, Any]]:
    """Read per-column metadata from disk (embedded mode fallback)."""
    meta = _read_local_metadata_full()
    if meta is None:
        return []
    return [c.to_dict() for c in meta.columns]


def _fetch_schema() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch the feature schema.

    Returns ``(columns, info)`` where ``columns`` is a list of dicts describing
    each feature (name, kind, default, min, max, categories) and ``info``
    is the raw ``/model/info`` payload (includes ``target`` and ``target_stats``).
    """
    mode = _current_mode()
    if mode == "embedded":
        cols = _read_local_metadata()
        meta = _read_local_metadata_full()
        info = {
            "target": meta.target if meta else get_config()["dataset"]["target_column"],
            "target_stats": meta.target_stats if meta else {},
            "feature_names": [c["name"] for c in cols],
        }
        return cols, info

    client = _get_client(os.getenv("API_URL", get_settings().api_url))
    info = client.model_info()
    cols: list[dict[str, Any]] = []
    for name in info.get("feature_names") or []:
        cols.append({"name": name, "kind": "numeric", "default": 0.0})
    # The /model/info endpoint returns only feature names. Enrich with local
    # per-column metadata (min/max/categories) when available via the shared
    # models/ volume in docker-compose.
    local_specs = {c["name"]: c for c in _read_local_metadata()}
    cols = [local_specs.get(c["name"], c) for c in cols]
    return cols, info


def _widget_for(col: dict[str, Any]) -> Any:
    kind = col.get("kind", "numeric")
    name = col["name"]
    label = name.replace("_", " ")
    default = col.get("default")

    if kind == "binary":
        return int(st.checkbox(label, value=bool(default or 0)))
    if kind == "categorical":
        cats = col.get("categories") or []
        if cats:
            idx = cats.index(default) if default in cats else 0
            return st.selectbox(label, cats, index=idx)
        return st.text_input(label, value=str(default or ""))
    # numeric
    lo = col.get("min")
    hi = col.get("max")
    val = float(default) if default is not None else 0.0
    if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and hi > lo:
        step = (hi - lo) / 100 if hi - lo > 0 else 1.0
        return float(st.slider(label, float(lo), float(hi), value=val, step=step))
    return float(st.number_input(label, value=val))


def _input_form(columns: list[dict[str, Any]]) -> dict[str, Any]:
    if not columns:
        st.warning(
            "No feature metadata available yet. Train a model first (see the "
            "README) and then reload this page."
        )
        return {}

    st.subheader("Input features")
    n_cols = 2
    grid = st.columns(n_cols)
    payload: dict[str, Any] = {}
    for i, col in enumerate(columns):
        with grid[i % n_cols]:
            payload[col["name"]] = _widget_for(col)
    return payload


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    mode = _current_mode()
    if mode == "embedded":
        model = _get_model()
        cols = [c["name"] for c in _read_local_metadata()]
        df = pd.DataFrame([{c: payload.get(c) for c in cols}]) if cols else pd.DataFrame([payload])
        preds = model.predict(df)
        value = preds[0]
        return {"predicted_value": float(value.item() if hasattr(value, "item") else value)}

    client = _get_client(os.getenv("API_URL", get_settings().api_url))
    return client.predict(payload)


def _format_value(raw: float, target: str) -> tuple[str, str | None]:
    """Return ``(primary, helper)`` strings for the metric widget.

    California Housing's target is median house value expressed in units of
    $100,000. We format it as dollars when the target column matches the
    sklearn default; otherwise show the raw number.
    """
    if target == "MedHouseVal":
        dollars = raw * 100_000
        return f"${dollars:,.0f}", f"{raw:.3f} × $100k"
    return f"{raw:.4f}", None


def _show_result(result: dict[str, Any], info: dict[str, Any]) -> None:
    st.subheader("Prediction")
    value = float(result.get("predicted_value", 0.0))
    target = info.get("target") or "target"
    primary, helper = _format_value(value, target)
    st.metric(f"Predicted {target}", primary, help=helper)

    stats = info.get("target_stats") or {}
    if stats:
        cols = st.columns(4)
        fields = [("min", "Min"), ("mean", "Mean"), ("median", "Median"), ("max", "Max")]
        for (k, label), c in zip(fields, cols, strict=False):
            if k in stats:
                ref, _ = _format_value(float(stats[k]), target)
                c.metric(f"Train {label}", ref)


def main() -> None:
    st.title("🏠 California Housing Regressor")
    st.caption(
        "Graduate MLOps II project — end-to-end regression pipeline served via "
        "FastAPI and the MLflow Model Registry. Dataset: scikit-learn's "
        "`fetch_california_housing` (median house value in units of $100,000)."
    )
    mode = _current_mode()
    st.sidebar.markdown(f"**Mode:** `{mode}`")
    if mode == "api":
        st.sidebar.markdown(f"**API URL:** `{os.getenv('API_URL', get_settings().api_url)}`")
        try:
            client = _get_client(os.getenv("API_URL", get_settings().api_url))
            health = client.health()
            st.sidebar.success(f"API health: {health['status']}")
        except Exception as exc:  # pragma: no cover - UI feedback
            st.sidebar.error(f"API unreachable: {exc}")

    columns, info = _fetch_schema()
    if info.get("target"):
        st.sidebar.markdown(f"**Target:** `{info['target']}`")

    payload = _input_form(columns)

    if payload and st.button("Predict", type="primary"):
        with st.spinner("Scoring…"):
            try:
                result = _predict(payload)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return
        _show_result(result, info)

    with st.expander("Raw request payload"):
        st.json(payload)


if __name__ == "__main__":
    main()
