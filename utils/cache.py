"""
utils/cache.py

LLM response cache keyed by SHA-256 hash of (model + prompt).
Persists to .cache/llm_cache.json to avoid redundant API calls.
Supports full replay of past experiments using cached responses.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "llm_cache.json"


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def make_cache_key(model: str, messages: list, temperature: float) -> str:
    """Deterministic SHA-256 key from model + messages + temperature."""
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached(key: str) -> Optional[str]:
    """Return cached response string, or None if not present."""
    cache = _load_cache()
    return cache.get(key)


def set_cached(key: str, response: str) -> None:
    """Store a response in the persistent cache."""
    cache = _load_cache()
    cache[key] = response
    _save_cache(cache)


def cache_stats() -> dict:
    """Return basic cache statistics."""
    cache = _load_cache()
    return {"entries": len(cache), "file": str(CACHE_FILE)}
