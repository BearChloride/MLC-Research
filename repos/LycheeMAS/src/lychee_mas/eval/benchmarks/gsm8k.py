"""GSM8K loader and data preparation.

三种数据来源，按优先级：

1. **本地 jsonl（推荐，免联网）**：``LYCHEE_GSM8K_DATA_DIR`` 指向含官方
   ``train.jsonl`` / ``test.jsonl`` 的目录，或用 ``LYCHEE_GSM8K_TRAIN_JSONL`` /
   ``LYCHEE_GSM8K_TEST_JSONL`` 分别指定文件；prepare 时转成 prepared parquet。
2. ModelScope 下载（``LYCHEE_DATA_SOURCE=modelscope``）。
3. HuggingFace 下载（``LYCHEE_DATA_SOURCE=huggingface``，国内常不可达）。

本地分支与 ``math500.py`` 的 ``LYCHEE_MATH500_JSONL`` 同款：只做格式转换，不联网。
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional

from .common import (
    load_parquet,
    parquet_ready,
    prepared_root,
    save_hf_dataset,
    save_modelscope_dataset,
)
from .source_catalog import provider_ids, source_backend_order

_LOCAL_DIR_ENV = "LYCHEE_GSM8K_DATA_DIR"
_LOCAL_SPLIT_ENVS = {"train": "LYCHEE_GSM8K_TRAIN_JSONL", "test": "LYCHEE_GSM8K_TEST_JSONL"}


def _local_split_path(split: str) -> Optional[str]:
    """解析某个 split 的本地 jsonl 路径；未配置返回 None，配错则显式报错。"""
    env_name = _LOCAL_SPLIT_ENVS[split]
    explicit = os.environ.get(env_name)
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"{env_name}={explicit!r} 指向的文件不存在")
        return explicit
    root = os.environ.get(_LOCAL_DIR_ENV)
    if not root:
        return None
    candidate = os.path.join(root, f"{split}.jsonl")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"{_LOCAL_DIR_ENV}={root!r} 下缺少 {split}.jsonl")
    return candidate


def _prepare_from_local() -> Optional[str]:
    """本地 jsonl（官方 train/test.jsonl，字段 question/answer）→ prepared parquet。"""
    paths = {split: _local_split_path(split) for split in ("train", "test")}
    if all(path is None for path in paths.values()):
        return None
    missing = [split for split, path in paths.items() if path is None]
    if missing:
        raise ValueError(f"本地 GSM8K 需同时提供 train 与 test 两个 split（缺 {missing}）")

    import pandas as pd  # 惰性导入：仅本地转换路径需要

    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    os.makedirs(out_dir, exist_ok=True)
    for split, path in paths.items():
        rows: List[Dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rows.append({"question": item["question"], "answer": item["answer"]})
        if not rows:
            raise ValueError(f"本地 GSM8K {split} 为空：{path}")
        pd.DataFrame(rows).to_parquet(os.path.join(out_dir, f"{split}-00000-of-00001.parquet"))
        print(f"[prepare] gsm8k: {split} {len(rows)} 题 <- {path}", flush=True)
    return out_dir


def _download_modelscope() -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    dataset_ids = provider_ids("gsm8k", "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            for split in ("train", "test"):
                if not _split_ready(split):
                    save_modelscope_dataset(dataset_id, "main", split, out_dir, benchmark="gsm8k")
            return out_dir
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope GSM8K download failed; " + " | ".join(errors))


def _download_huggingface() -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    dataset_ids = provider_ids("gsm8k", "huggingface")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            for split in ("train", "test"):
                if not _split_ready(split):
                    save_hf_dataset(dataset_id, "main", split, out_dir, benchmark="gsm8k")
            return out_dir
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("HuggingFace GSM8K download failed; " + " | ".join(errors))


def prepare_gsm8k(force: bool = False, source: Optional[str] = None) -> str:
    out_dir = os.path.join(prepared_root(), "gsm8k", "main")
    if not force and _has_ready_cache():
        return out_dir
    if force and os.path.isdir(os.path.join(prepared_root(), "gsm8k")):
        shutil.rmtree(os.path.join(prepared_root(), "gsm8k"))
    local = _prepare_from_local()  # 本地 jsonl 优先，命中即免联网
    if local is not None:
        return local
    errors: list[str] = []
    for backend in source_backend_order("gsm8k", source):
        try:
            return _download_modelscope() if backend == "modelscope" else _download_huggingface()
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare GSM8K; " + " | ".join(errors))


def _has_ready_cache() -> bool:
    return _split_ready("train") and _split_ready("test")


def _split_ready(split: str) -> bool:
    return parquet_ready("gsm8k/main", split, required_columns=("question", "answer"))


def load_gsm8k(n: Optional[int] = None) -> List[Dict]:
    prepare_gsm8k()
    out = []
    for it in load_parquet("gsm8k/main", "test"):
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            {
                "task": "gsm8k",
                "kind": "exact",
                "question": it["question"].strip() + "\nGive the final numeric answer.",
                "gold": gold,
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    return out
