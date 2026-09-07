"""Resolve remote JSON/CSV URLs without relying on host env (e.g. Reflex Cloud paywalled env).

Reads ``remote_data_urls.json`` next to this file. Environment variables still win
when set (useful for local .env).
"""

from __future__ import annotations

import json
import os
from typing import Any

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_APP_DIR, "remote_data_urls.json")


def _load_file() -> dict[str, Any]:
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def predictions_cache_url() -> str:
    u = (
        os.getenv("PREDICTIONS_CACHE_URL")
        or os.getenv("AUGO_PREDICTIONS_CACHE_URL")
        or ""
    ).strip()
    if u:
        return u
    return str(_load_file().get("predictions_cache_url", "")).strip()


def results_csv_url() -> str:
    u = (os.getenv("RESULTS_URL") or os.getenv("AUGO_RESULTS_URL") or "").strip()
    if u:
        return u
    return str(_load_file().get("results_url", "")).strip()
