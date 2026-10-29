"""Central catalog of benchmark download sources.

The catalog is intentionally declarative. Benchmark modules still own their
schema conversion logic, while source IDs, override env vars, and fallback file
lists live in one JSON file for easier review.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).with_name("source_catalog.json")


@lru_cache(maxsize=1)
def source_catalog() -> dict[str, dict[str, Any]]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def source_entry(benchmark: str) -> dict[str, Any]:
    try:
        return source_catalog()[benchmark]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark source catalog entry {benchmark!r}") from exc


def provider_env(benchmark: str, provider: str) -> str | None:
    value = source_entry(benchmark).get(provider, {}).get("env")
    return str(value) if value else None


def provider_ids(benchmark: str, provider: str) -> list[str]:
    section = source_entry(benchmark).get(provider, {})
    ids: list[str] = []
    env_name = section.get("env")
    if env_name:
        override = os.environ.get(str(env_name))
        if override:
            ids.append(override)
    ids.extend(str(item) for item in section.get("default_ids", []) if item)
    return ids


def source_backend_order(
    benchmark: str,
    source: str | None,
    *,
    providers: tuple[str, ...] = ("modelscope", "huggingface"),
) -> list[str]:
    source = (source or os.environ.get("LYCHEE_DATA_SOURCE") or "auto").lower()
    if source == "auto":
        return [provider for provider in providers if provider_ids(benchmark, provider)]
    if source in providers:
        return [source]
    valid = ", ".join(("auto", *providers))
    raise ValueError(f"unknown {benchmark} data source {source!r}; choose {valid}")


def other_defaults(benchmark: str) -> list[dict[str, Any]]:
    return list(source_entry(benchmark).get("other_defaults", []))


def fallback_specs(benchmark: str) -> list[dict[str, Any]]:
    return list(source_entry(benchmark).get("fallback_files", []))
