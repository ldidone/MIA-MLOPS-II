"""Thin HTTP client used by the Streamlit app when ``APP_MODE=api``."""

from __future__ import annotations

from typing import Any

import httpx


class APIClient:
    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> dict[str, Any]:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def model_info(self) -> dict[str, Any]:
        r = self._client.get("/model/info")
        r.raise_for_status()
        return r.json()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/predict", json={"features": payload})
        r.raise_for_status()
        return r.json()
