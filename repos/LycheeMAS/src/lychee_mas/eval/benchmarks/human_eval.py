"""HumanEval loader + official-style scoring.

The shape mirrors the existing gsm8k/aime loaders: load lazily, cache under
the benchmark raw root, and return records with
``{task, kind, question, gold, context}``.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .common import log_download_source, prepared_root, raw_source_dir, restore_prepared_from_raw
from .source_catalog import other_defaults, provider_ids

URL = other_defaults("human_eval")[0]["id"]
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("LYCHEEMAS_EVAL_TIMEOUT", "20"))

_RUNNER = r"""
from pathlib import Path
import json
import sys
import traceback

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

try:
    namespace = {}
    exec(payload["test"], namespace)
    exec(payload["answer"], namespace)
    check = namespace["check"]
    candidate = namespace[payload["entry_point"]]
    check(candidate)
    print(json.dumps({"success": True, "error": None}))
except Exception:
    print(json.dumps({"success": False, "error": traceback.format_exc()}))
    sys.exit(1)
"""


def source_archive(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(prepared_root()) / "human_eval"
    return base / "HumanEval.jsonl.gz"


def _raw_archive(provider: str, identifier: str) -> Path:
    return raw_source_dir("human_eval", provider, identifier) / "HumanEval.jsonl.gz"


def _publish_archive(src: Path, archive: Path) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, archive)
    return archive


def safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "_")


def _write_records_archive(archive: Path, records: list[dict[str, Any]]) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
    with gzip.open(archive, "wb") as fh:
        fh.write(payload.encode("utf-8"))
    return archive


def _archive_ready(archive: Path) -> bool:
    if not archive.is_file() or archive.stat().st_size <= 0:
        return False
    required = {"task_id", "prompt", "test", "entry_point"}
    try:
        with gzip.open(archive, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                return required.issubset(row)
    except Exception:
        return False
    return False


def _records_from_dataset(dataset) -> list[dict[str, Any]]:
    rows = [dict(row) for row in dataset]
    required = {"task_id", "prompt", "test", "entry_point"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"HumanEval dataset must contain {sorted(required)}; "
            f"got {list(rows[0]) if rows else []}"
        )
    return rows


def _download_from_huggingface(archive: Path) -> Path:
    from datasets import load_dataset

    dataset_id = provider_ids("human_eval", "huggingface")[0]
    raw_archive = _raw_archive("huggingface", dataset_id)
    log_download_source("human_eval", "huggingface", dataset_id, raw_archive)
    _write_records_archive(
        raw_archive, _records_from_dataset(load_dataset(dataset_id, split="test"))
    )
    return _publish_archive(raw_archive, archive)


def _download_from_modelscope(archive: Path) -> Path:
    from datasets import Dataset
    from modelscope.msdatasets import MsDataset

    dataset_ids = provider_ids("human_eval", "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            raw_archive = _raw_archive("modelscope", dataset_id)
            log_download_source("human_eval", "modelscope", dataset_id, raw_archive)
            ms_dataset = MsDataset.load(dataset_id, split="test")
            if hasattr(ms_dataset, "to_hf_dataset"):
                dataset = ms_dataset.to_hf_dataset()
            elif hasattr(ms_dataset, "to_list"):
                dataset = Dataset.from_list(ms_dataset.to_list())
            else:
                dataset = Dataset.from_list(list(ms_dataset))
            _write_records_archive(raw_archive, _records_from_dataset(dataset))
            return _publish_archive(raw_archive, archive)
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope HumanEval download failed; " + " | ".join(errors))


def _download_from_github(archive: Path) -> Path:
    raw_archive = _raw_archive("github_raw", URL)
    log_download_source("human_eval", "github", URL, raw_archive)
    raw_archive.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "LycheeMAS/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw_archive.write_bytes(response.read())
    return _publish_archive(raw_archive, archive)


def _backend_order(source: str | None) -> list[str]:
    source = (source or os.environ.get("LYCHEE_DATA_SOURCE") or "auto").lower()
    if source == "auto":
        return ["modelscope", "huggingface", "github"]
    if source == "huggingface":
        return ["huggingface", "github"]
    if source == "modelscope":
        return ["modelscope"]
    if source == "github":
        return ["github"]
    raise ValueError(f"unknown HumanEval source {source!r}")


def _raw_archive_candidates(source: str | None) -> list[tuple[str, str, Path]]:
    candidates: list[tuple[str, str, Path]] = []
    for backend in _backend_order(source):
        if backend == "modelscope":
            for dataset_id in provider_ids("human_eval", "modelscope"):
                candidates.append(
                    ("modelscope", dataset_id, _raw_archive("modelscope", dataset_id))
                )
        elif backend == "huggingface":
            for dataset_id in provider_ids("human_eval", "huggingface"):
                candidates.append(
                    ("huggingface", dataset_id, _raw_archive("huggingface", dataset_id))
                )
        elif backend == "github":
            candidates.append(("github_raw", URL, _raw_archive("github_raw", URL)))
    return candidates


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    archive = source_archive(root)
    if _archive_ready(archive) and not force_download:
        return archive
    if not force_download:
        restored = restore_prepared_from_raw(
            "human_eval",
            archive,
            _raw_archive_candidates(source),
            ready=_archive_ready,
        )
        if restored is not None:
            return restored
    if force_download and archive.exists():
        archive.unlink()
    errors: list[str] = []
    for backend in _backend_order(source):
        try:
            if backend == "modelscope":
                return _download_from_modelscope(archive)
            if backend == "huggingface":
                return _download_from_huggingface(archive)
            return _download_from_github(archive)
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare HumanEval; " + " | ".join(errors))


def load_source_records(
    root: str | Path | None = None,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    archive = ensure_source(root, force_download=force_download)
    records: list[dict[str, Any]] = []
    with gzip.GzipFile(fileobj=io.BytesIO(archive.read_bytes())) as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def load_human_eval(n: Optional[int] = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in load_source_records():
        prompt = task["prompt"].strip()
        out.append(
            {
                "task": "human_eval",
                "kind": "human_eval",
                "question": (
                    "Complete the following Python function. Return only the complete "
                    "function implementation.\n\n"
                    f"```python\n{prompt}\n```"
                ),
                "gold": {
                    "task_id": task["task_id"],
                    "entry_point": task["entry_point"],
                    "test": task["test"],
                },
                "context": None,
                "metadata": {
                    "task_id": task["task_id"],
                    "safe_task_id": safe_task_id(task["task_id"]),
                },
            }
        )
        if n and len(out) >= n:
            break
    return out


def extract_python_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else text.strip()


def evaluate_code(answer_code: str, test_code: str, entry_point: str) -> dict[str, Any]:
    try:
        namespace: dict[str, Any] = {}
        exec(test_code, namespace)
        exec(answer_code, namespace)
        check = namespace["check"]
        candidate = namespace[entry_point]
        check(candidate)
        return {"success": True, "error": None}
    except Exception:
        return {"success": False, "error": traceback.format_exc()}


def evaluate_answer(
    answer_text: str,
    gold: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload = {
        "answer": extract_python_code(answer_text),
        "test": gold["test"],
        "entry_point": gold["entry_point"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-c", _RUNNER, str(payload_path)],
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "success": False,
                "error": f"Evaluation timed out after {exc.timeout} seconds.",
                "returncode": None,
                "stderr": exc.stderr,
            }

    eval_result = parse_eval_output(result.stdout)
    eval_result["returncode"] = result.returncode
    eval_result["stderr"] = result.stderr
    return eval_result


def parse_eval_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "success" in value:
            return value
    return {"success": False, "error": "No JSON evaluation result was produced."}
