"""Shared helpers for LycheeMAS benchmark loaders.

The individual benchmark modules own their dataset-specific choices. This
module only keeps the common mechanics: data roots, provider order, parquet
IO, ModelScope/HuggingFace conversion, and multiple-choice formatting.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

DATA_BACKENDS = ("modelscope", "huggingface")

DEFAULT_RAW_ROOT = os.path.join("data", "benchmarks", "raw")
DEFAULT_PREPARED_ROOT = os.path.join("data", "benchmarks", "prepared")
DEFAULT_RUNS_ROOT = os.path.join("runs", "benchmarks")


def raw_root() -> str:
    return (
        os.environ.get("LYCHEE_BENCHMARK_RAW_ROOT")
        or os.environ.get("CDM_DATA_ROOT")
        or DEFAULT_RAW_ROOT
    )


def prepared_root() -> str:
    return (
        os.environ.get("LYCHEE_BENCHMARK_PREPARED_ROOT")
        or os.environ.get("CDM_DATA_ROOT")
        or DEFAULT_PREPARED_ROOT
    )


def processed_root() -> str:
    """Backward-compatible alias for the prepared benchmark cache root."""
    return prepared_root()


def runs_root() -> str:
    return os.environ.get("LYCHEE_BENCHMARK_RUNS_ROOT", DEFAULT_RUNS_ROOT)


def safe_source_id(identifier: str) -> str:
    """Make a provider source id safe as one local directory name."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "--", str(identifier).strip()).strip("._-")
    return value or "source"


def raw_source_dir(benchmark: str, provider: str, identifier: str) -> Path:
    return Path(raw_root()) / benchmark / provider / safe_source_id(identifier)


def prepared_benchmark_dir(benchmark: str) -> Path:
    return Path(prepared_root()) / benchmark


def remove_path(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.is_file():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p)


def copy_raw_to_prepared(raw_dir: str | os.PathLike, prepared_dir: str | os.PathLike) -> Path:
    """Copy raw source data into prepared so loaders always read real files."""
    raw_path = Path(raw_dir).resolve()
    prepared_path = Path(prepared_dir)
    if prepared_path.exists():
        remove_path(prepared_path)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.is_dir():
        shutil.copytree(raw_path, prepared_path)
    else:
        shutil.copy2(raw_path, prepared_path)
    return prepared_path


def raw_source_candidates(
    benchmark: str,
    providers: Sequence[str] = DATA_BACKENDS,
) -> list[tuple[str, str, Path]]:
    """Return raw cache locations for configured provider ids in preferred order."""
    from .source_catalog import provider_ids

    candidates: list[tuple[str, str, Path]] = []
    for provider in providers:
        for identifier in provider_ids(benchmark, provider):
            candidates.append(
                (provider, identifier, raw_source_dir(benchmark, provider, identifier))
            )
    return candidates


def restore_prepared_from_raw(
    benchmark: str,
    prepared_path: str | os.PathLike,
    candidates: Sequence[tuple[str, str, Path]],
    *,
    ready: Callable[[Path], bool] | None = None,
) -> Path | None:
    """Restore prepared data from an existing raw source when possible.

    This intentionally copies data into prepared rather than linking it, so
    loaders always see a standalone, inspectable prepared tree.
    """
    for provider, identifier, raw_path in candidates:
        if not raw_path.exists():
            continue
        if ready is not None and not ready(raw_path):
            continue
        restored = copy_raw_to_prepared(raw_path, prepared_path)
        print(
            f"[prepare] {benchmark}: restored prepared from raw "
            f"provider={provider} id={identifier} -> {restored}",
            flush=True,
        )
        return restored
    return None


def parquet_files(
    subdir: str, split: Optional[str] = None, root: Optional[str] = None
) -> list[str]:
    base = os.path.join(root or prepared_root(), subdir)
    if split:
        return glob.glob(os.path.join(base, f"{split}-*.parquet"))
    return glob.glob(os.path.join(base, "*.parquet"))


def has_parquet(subdir: str, split: Optional[str] = None, root: Optional[str] = None) -> bool:
    return bool(parquet_files(subdir, split=split, root=root))


def parquet_ready(
    subdir: str,
    split: str,
    *,
    required_columns: Sequence[str] = (),
    min_rows: int = 1,
    root: Optional[str] = None,
) -> bool:
    """Return True only if cached parquet files are readable and minimally shaped."""
    files = parquet_files(subdir, split=split, root=root)
    if not files:
        return False
    try:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files={split: files}, split=split)
        if min_rows and len(dataset) < min_rows:
            return False
        missing = [name for name in required_columns if name not in dataset.column_names]
        return not missing
    except Exception:
        return False


def json_file_ready(path: str | os.PathLike, *, min_rows: int = 1) -> bool:
    """Return True when a cached JSON/JSONL file can be parsed and is non-empty."""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        stripped = text.lstrip()
        if not stripped:
            return False
        if stripped.startswith("["):
            data = json.loads(text)
            return isinstance(data, list) and len(data) >= min_rows
        count = 0
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
                count += 1
                if count >= min_rows:
                    return True
        return min_rows == 0
    except Exception:
        return False


def backend_order(
    source: Optional[str] = None,
    *,
    backends: Sequence[str] = DATA_BACKENDS,
    source_label: str = "data source",
) -> list[str]:
    source = (source or os.environ.get("LYCHEE_DATA_SOURCE") or "auto").lower()
    if source == "auto":
        return list(backends)
    if source not in backends:
        valid = ", ".join(("auto", *backends))
        raise ValueError(f"unknown {source_label} {source!r}; choose {valid}")
    return [source]


def log_download_source(
    benchmark: str,
    provider: str,
    identifier: str,
    destination: str | os.PathLike | None = None,
) -> None:
    """Print a stable, grep-friendly line showing which provider is being used."""
    msg = f"[download] {benchmark}: provider={provider} id={identifier}"
    if destination is not None:
        msg += f" -> {destination}"
    print(msg, flush=True)


def load_parquet(subdir: str, split: str):
    from datasets import load_dataset

    base = os.path.join(prepared_root(), subdir)
    files = parquet_files(subdir, split=split) or parquet_files(subdir)
    if not files:
        raise FileNotFoundError(f"no parquet under {base}")
    return load_dataset("parquet", data_files={split: files}, split=split)


def save_hf_dataset(
    repo_id: str, subset: Optional[str], split: str, out_dir: str, *, benchmark: str = "hf_dataset"
):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    src = raw_source_dir(benchmark, "huggingface", repo_id)
    log_download_source(benchmark, "huggingface", repo_id, src)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(src))
    dataset = (
        load_dataset(repo_id, subset, split=split) if subset else load_dataset(repo_id, split=split)
    )
    os.makedirs(out_dir, exist_ok=True)
    dataset.to_parquet(os.path.join(out_dir, f"{split}-00000-of-00001.parquet"))
    return dataset


def msdataset_to_hf_dataset(dataset):
    from datasets import Dataset

    if hasattr(dataset, "to_hf_dataset"):
        return dataset.to_hf_dataset()
    if hasattr(dataset, "hf_ds"):
        return dataset.hf_ds
    if hasattr(dataset, "to_dataset"):
        converted = dataset.to_dataset()
        if hasattr(converted, "to_hf_dataset"):
            return converted.to_hf_dataset()
        return converted
    if hasattr(dataset, "to_list"):
        return Dataset.from_list(dataset.to_list())
    return Dataset.from_list(list(dataset))


def load_modelscope_dataset(dataset_id: str, subset: Optional[str], split: str):
    from modelscope.msdatasets import MsDataset

    kwargs = {"split": split}
    if subset:
        kwargs["subset_name"] = subset
    dataset = MsDataset.load(dataset_id, **kwargs)
    return msdataset_to_hf_dataset(dataset)


def save_modelscope_dataset(
    dataset_id: str,
    subset: Optional[str],
    split: str,
    out_dir: str,
    *,
    benchmark: str = "modelscope_dataset",
):
    from modelscope.hub.snapshot_download import snapshot_download

    src = raw_source_dir(benchmark, "modelscope", dataset_id)
    log_download_source(benchmark, "modelscope", dataset_id, src)
    snapshot_download(dataset_id, repo_type="dataset", local_dir=str(src))
    dataset = load_modelscope_dataset(dataset_id, subset, split)
    os.makedirs(out_dir, exist_ok=True)
    dataset.to_parquet(os.path.join(out_dir, f"{split}-00000-of-00001.parquet"))
    return dataset


def hf_resolve_url(repo_id: str, path: str, revision: str = "main") -> str:
    quoted_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_path}"


def download_url(url: str, dest: str, *, timeout: int = 120) -> str:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    resume_from = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    try:
        request = urllib.request.Request(url)
        if resume_from:
            request.add_header("Range", f"bytes={resume_from}-")
        response = urllib.request.urlopen(request, timeout=timeout)
        mode = "ab" if resume_from and getattr(response, "status", None) == 206 else "wb"
        if mode == "wb":
            resume_from = 0
        with response, open(tmp, mode) as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        # Keep the .part file so the next attempt can resume if the server
        # supports HTTP Range requests.
        raise
    return dest


def download_hf_files(
    repo_id: str, files: Sequence[str], out_dir: str, *, revision: str = "main"
) -> list[str]:
    log_download_source("hf_files", "huggingface", repo_id, out_dir)
    downloaded: list[str] = []
    errors: list[str] = []
    for path in files:
        dest = os.path.join(out_dir, path)
        try:
            downloaded.append(download_url(hf_resolve_url(repo_id, path, revision=revision), dest))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError("failed to download HuggingFace files; " + " | ".join(errors))
    return downloaded


def list_hf_dataset_files(repo_id: str, *, revision: str = "main") -> list[str]:
    try:
        from huggingface_hub import HfApi

        return list(
            HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        )
    except Exception:
        pass
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}?recursive=1"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    files: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("type") == "file" and item.get("path"):
                files.append(str(item["path"]))
    return files


def format_choices(stem: str, texts, labels) -> str:
    lines = [f"{lab}. {txt}" for lab, txt in zip(labels, texts)]
    return stem.strip() + "\n" + "\n".join(lines) + "\nAnswer with the option letter."
