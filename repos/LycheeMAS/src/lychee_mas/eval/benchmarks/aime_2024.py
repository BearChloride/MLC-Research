"""AIME 2024 loader and data preparation."""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

from .common import (
    load_modelscope_dataset,
    load_parquet,
    log_download_source,
    parquet_ready,
    prepared_root,
    raw_source_dir,
)
from .source_catalog import provider_ids, source_backend_order


def _standardize_columns(dataset):
    columns = set(dataset.column_names)
    rename = {}
    if "problem" not in columns:
        for candidate in ("problem", "Problem", "question", "Question", "prompt", "Prompt"):
            if candidate in columns and candidate != "problem":
                rename[candidate] = "problem"
                break
    if "answer" not in columns:
        for candidate in (
            "answer",
            "Answer",
            "final_answer",
            "Final answer",
            "gold",
            "solution_answer",
        ):
            if candidate in columns and candidate != "answer":
                rename[candidate] = "answer"
                break
    for old, new in rename.items():
        dataset = dataset.rename_column(old, new)
    if "problem" not in dataset.column_names or "answer" not in dataset.column_names:
        raise ValueError(
            f"AIME dataset columns must include problem/answer; got {dataset.column_names}"
        )
    return dataset


def _download_modelscope() -> str:
    out_dir = os.path.join(prepared_root(), "aime_2024", "data")
    dataset_ids = provider_ids("aime_2024", "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            src = raw_source_dir("aime_2024", "modelscope", dataset_id)
            log_download_source("aime_2024", "modelscope", dataset_id, src)
            snapshot_download(dataset_id, repo_type="dataset", local_dir=str(src))
            dataset = load_modelscope_dataset(dataset_id, None, "train")
            dataset = _standardize_columns(dataset)
            os.makedirs(out_dir, exist_ok=True)
            dataset.to_parquet(os.path.join(out_dir, "train-00000-of-00001.parquet"))
            return out_dir
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope AIME 2024 download failed; " + " | ".join(errors))


def _download_huggingface() -> str:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    out_dir = os.path.join(prepared_root(), "aime_2024", "data")
    dataset_ids = provider_ids("aime_2024", "huggingface")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            src = raw_source_dir("aime_2024", "huggingface", dataset_id)
            log_download_source("aime_2024", "huggingface", dataset_id, src)
            snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=str(src))
            dataset = load_dataset(dataset_id, split="train")
            dataset = _standardize_columns(dataset)
            os.makedirs(out_dir, exist_ok=True)
            dataset.to_parquet(os.path.join(out_dir, "train-00000-of-00001.parquet"))
            return out_dir
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("HuggingFace AIME 2024 download failed; " + " | ".join(errors))


def prepare_aime_2024(force: bool = False, source: Optional[str] = None) -> str:
    out_dir = os.path.join(prepared_root(), "aime_2024", "data")
    if not force and parquet_ready(
        "aime_2024/data", "train", required_columns=("problem", "answer")
    ):
        return out_dir
    if force and os.path.isdir(os.path.join(prepared_root(), "aime_2024")):
        shutil.rmtree(os.path.join(prepared_root(), "aime_2024"))
    errors: list[str] = []
    for backend in source_backend_order("aime_2024", source):
        try:
            return _download_modelscope() if backend == "modelscope" else _download_huggingface()
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare AIME 2024; " + " | ".join(errors))


def load_aime_2024(n: Optional[int] = None) -> List[Dict]:
    prepare_aime_2024()
    out = []
    for it in load_parquet("aime_2024/data", "train"):
        out.append(
            {
                "task": "aime_2024",
                "kind": "aime",
                "question": it["problem"].strip() + "\nGive the final integer answer.",
                "gold": str(it["answer"]).strip(),
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    return out
