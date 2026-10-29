"""Multiple-choice benchmark loaders."""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from typing import Dict, List, Optional

from .common import (
    format_choices,
    json_file_ready,
    load_modelscope_dataset,
    load_parquet,
    log_download_source,
    parquet_ready,
    prepared_root,
    raw_source_dir,
    save_hf_dataset,
    save_modelscope_dataset,
)
from .source_catalog import provider_ids, source_backend_order


def _download_arc_modelscope() -> str:
    out_dir = os.path.join(prepared_root(), "arc_easy")
    dataset_ids = provider_ids("arc_easy", "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            if dataset_id == "OmniData/ARC":
                return _download_arc_omnidata(dataset_id, out_dir)
            save_modelscope_dataset(dataset_id, "ARC-Easy", "test", out_dir, benchmark="arc_easy")
            return out_dir
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope ARC-Easy download failed; " + " | ".join(errors))


def _download_arc_omnidata(dataset_id: str, out_dir: str) -> str:
    from datasets import Dataset
    from modelscope.hub.snapshot_download import snapshot_download

    src = raw_source_dir("arc_easy", "modelscope", dataset_id)
    zip_path = src / "raw" / "ARC-V1-Feb2018.zip"
    if not zip_path.is_file() or zip_path.stat().st_size <= 0:
        log_download_source("arc_easy", "modelscope", dataset_id, src)
        snapshot_download(
            dataset_id,
            repo_type="dataset",
            local_dir=str(src),
            allow_patterns=["raw/ARC-V1-Feb2018.zip"],
        )
    else:
        print(
            f"[prepare] arc_easy: rebuilding prepared from raw provider=modelscope "
            f"id={dataset_id} -> {out_dir}",
            flush=True,
        )
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        jsonl_name = next(
            (
                name
                for name in archive.namelist()
                if name.endswith("ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl")
            ),
            None,
        )
        if not jsonl_name:
            raise FileNotFoundError("ARC-Easy-Test.jsonl not found in ARC-V1-Feb2018.zip")
        with archive.open(jsonl_name) as fh:
            for raw in fh:
                item = json.loads(raw.decode("utf-8"))
                choices = item["question"]["choices"]
                rows.append(
                    {
                        "id": item.get("id"),
                        "question": item["question"]["stem"],
                        "choices": {
                            "text": [str(choice["text"]) for choice in choices],
                            "label": [str(choice["label"]) for choice in choices],
                        },
                        "answerKey": str(item["answerKey"]),
                    }
                )
    if not rows:
        raise ValueError("ARC-Easy test split is empty after extracting OmniData/ARC")
    os.makedirs(out_dir, exist_ok=True)
    Dataset.from_list(rows).to_parquet(os.path.join(out_dir, "test-00000-of-00001.parquet"))
    return out_dir


def _download_arc_huggingface() -> str:
    out_dir = os.path.join(prepared_root(), "arc_easy")
    dataset_ids = provider_ids("arc_easy", "huggingface")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            save_hf_dataset(dataset_id, "ARC-Easy", "test", out_dir, benchmark="arc_easy")
            return out_dir
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("HuggingFace ARC-Easy download failed; " + " | ".join(errors))


def prepare_arc_easy(force: bool = False, source: Optional[str] = None) -> str:
    out_dir = os.path.join(prepared_root(), "arc_easy")
    if not force and parquet_ready(
        "arc_easy",
        "test",
        required_columns=("question", "choices", "answerKey"),
    ):
        return out_dir
    if force and os.path.isdir(os.path.join(prepared_root(), "arc_easy")):
        shutil.rmtree(os.path.join(prepared_root(), "arc_easy"))
    errors: list[str] = []
    for backend in source_backend_order("arc_easy", source):
        try:
            return (
                _download_arc_modelscope()
                if backend == "modelscope"
                else _download_arc_huggingface()
            )
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare ARC-Easy; " + " | ".join(errors))


def _download_openbookqa_modelscope() -> str:
    out_dir = os.path.join(prepared_root(), "openbookqa")
    dataset_ids = provider_ids("openbookqa", "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            save_modelscope_dataset(dataset_id, "main", "test", out_dir, benchmark="openbookqa")
            return out_dir
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope OpenBookQA download failed; " + " | ".join(errors))


def _download_openbookqa_huggingface() -> str:
    out_dir = os.path.join(prepared_root(), "openbookqa")
    dataset_ids = provider_ids("openbookqa", "huggingface")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            save_hf_dataset(dataset_id, "main", "test", out_dir, benchmark="openbookqa")
            return out_dir
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("HuggingFace OpenBookQA download failed; " + " | ".join(errors))


def prepare_openbookqa(force: bool = False, source: Optional[str] = None) -> str:
    out_dir = os.path.join(prepared_root(), "openbookqa")
    if not force and parquet_ready(
        "openbookqa",
        "test",
        required_columns=("question_stem", "choices", "answerKey"),
    ):
        return out_dir
    if force and os.path.isdir(os.path.join(prepared_root(), "openbookqa")):
        shutil.rmtree(os.path.join(prepared_root(), "openbookqa"))
    errors: list[str] = []
    for backend in source_backend_order("openbookqa", source):
        try:
            return (
                _download_openbookqa_modelscope()
                if backend == "modelscope"
                else _download_openbookqa_huggingface()
            )
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare OpenBookQA; " + " | ".join(errors))


def _split_medqa_options(options) -> tuple[list[str], list[str]]:
    if isinstance(options, dict):
        labels = [str(label) for label in sorted(options)]
        texts = [str(options[label]) for label in labels]
        return labels, texts
    labels: list[str] = []
    texts: list[str] = []
    for idx, option in enumerate(options or []):
        text = str(option)
        match = re.match(r"\s*([A-Z])[\).\s]+(.*)", text)
        if match:
            labels.append(match.group(1))
            texts.append(match.group(2).strip())
        else:
            labels.append(chr(ord("A") + idx))
            texts.append(text.strip())
    return labels, texts


def _standardize_medqa_row(row: dict) -> dict:
    source = dict(row)
    if "instruction" in source and "output" in source:
        instruction = str(source.get("instruction") or "")
        stem, _, option_block = instruction.partition("Options:")
        options: dict[str, str] = {}
        for line in option_block.splitlines():
            match = re.match(r"\s*([A-Z])[\).\s]+(.+?)\s*$", line)
            if match:
                options[match.group(1)] = match.group(2)
        source = {
            "question": stem.strip(),
            "options": options,
            "answer": source.get("output"),
            "answer_idx": (source.get("info") or {}).get("answer_idx"),
        }
    question = str(
        source.get("question") or source.get("sent1") or source.get("prompt") or ""
    ).strip()
    labels, texts = _split_medqa_options(source.get("options") or source.get("choices"))
    answer = str(source.get("answer") or source.get("gold") or "").strip()
    answer_idx = source.get("answer_idx", source.get("answerKey", source.get("label")))
    answer_letter = None
    if answer in labels:
        answer_letter = answer
        answer = dict(zip(labels, texts)).get(answer_letter, answer)
    if isinstance(answer_idx, int) and 0 <= answer_idx < len(labels):
        answer_letter = labels[answer_idx]
    elif answer_idx is not None:
        candidate = str(answer_idx).strip()
        if candidate in labels:
            answer_letter = candidate
        elif candidate.isdigit() and 0 <= int(candidate) < len(labels):
            answer_letter = labels[int(candidate)]
    if not answer and answer_letter:
        answer = dict(zip(labels, texts)).get(answer_letter, "")
    if not question or not labels or not texts or not answer:
        raise ValueError(f"unsupported MedQA row schema: {sorted(row)}")
    return {
        "question": question,
        "options": [f"{label}. {text}" for label, text in zip(labels, texts)],
        "answer": answer,
    }


def _save_medqa_dataset(dataset, out_file: str) -> str:
    rows = []
    for row in dataset:
        rows.append(_standardize_medqa_row(dict(row)))
    if not rows:
        raise ValueError("MedQA dataset is empty after standardization")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    return out_file


def _download_medqa_modelscope() -> str:
    out_file = os.path.join(prepared_root(), "medqa", "medqa.json")
    dataset_ids = provider_ids("medqa", "modelscope")
    if not any(dataset_ids):
        raise RuntimeError(
            "no known ModelScope mirror for MedQA; set LYCHEE_MEDQA_MODELSCOPE_ID "
            "or use source=huggingface"
        )
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        if dataset_id == "AI-ModelScope/med_qa":
            try:
                return _download_medqa_modelscope_zip(
                    dataset_id,
                    "data_clean.zip",
                    ["data_clean/questions/US/test.jsonl"],
                    out_file,
                )
            except Exception as exc:  # pragma: no cover - provider/network dependent
                errors.append(f"{dataset_id}/zip: {exc}")
                continue
        if dataset_id == "cloakone/MedQA":
            try:
                return _download_medqa_modelscope_zip(
                    dataset_id,
                    "MedQA.zip",
                    ["processed-v1/test.json"],
                    out_file,
                )
            except Exception as exc:  # pragma: no cover - provider/network dependent
                errors.append(f"{dataset_id}/zip: {exc}")
                continue
        for split in ("test", "validation", "train"):
            try:
                from modelscope.hub.snapshot_download import snapshot_download

                src = raw_source_dir("medqa", "modelscope", dataset_id)
                log_download_source("medqa", "modelscope", dataset_id, src)
                snapshot_download(dataset_id, repo_type="dataset", local_dir=str(src))
                dataset = load_modelscope_dataset(dataset_id, None, split)
                return _save_medqa_dataset(dataset, out_file)
            except Exception as exc:  # pragma: no cover - provider/network dependent
                errors.append(f"{dataset_id}/{split}: {exc}")
    raise RuntimeError("ModelScope MedQA download failed; " + " | ".join(errors))


def _download_medqa_modelscope_zip(
    dataset_id: str,
    zip_name: str,
    candidate_paths: list[str],
    out_file: str,
) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    src = raw_source_dir("medqa", "modelscope", dataset_id)
    zip_path = src / zip_name
    if not zip_path.is_file() or zip_path.stat().st_size <= 0:
        log_download_source("medqa", "modelscope", dataset_id, src)
        snapshot_download(
            dataset_id,
            repo_type="dataset",
            local_dir=str(src),
            allow_patterns=[zip_name],
        )
    else:
        print(
            f"[prepare] medqa: rebuilding prepared from raw provider=modelscope "
            f"id={dataset_id} -> {out_file}",
            flush=True,
        )
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        data_name = next((name for name in candidate_paths if name in archive.namelist()), None)
        if not data_name:
            raise FileNotFoundError(f"none of {candidate_paths} found in {zip_name}")
        with archive.open(data_name) as fh:
            raw = fh.read().decode("utf-8")
        stripped = raw.lstrip()
        if stripped.startswith("["):
            rows = json.loads(raw)
        else:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return _save_medqa_dataset(rows, out_file)


def _download_medqa_huggingface() -> str:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    out_file = os.path.join(prepared_root(), "medqa", "medqa.json")
    dataset_ids = provider_ids("medqa", "huggingface")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        for split in ("test", "validation", "train"):
            try:
                src = raw_source_dir("medqa", "huggingface", dataset_id)
                log_download_source("medqa", "huggingface", dataset_id, src)
                snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=str(src))
                dataset = load_dataset(dataset_id, split=split)
                return _save_medqa_dataset(dataset, out_file)
            except Exception as exc:  # pragma: no cover - provider/network dependent
                errors.append(f"{dataset_id}/{split}: {exc}")
    raise RuntimeError("HuggingFace MedQA download failed; " + " | ".join(errors))


def prepare_medqa(force: bool = False, source: Optional[str] = None) -> str:
    out_file = os.path.join(prepared_root(), "medqa", "medqa.json")
    if not force and json_file_ready(out_file):
        return out_file
    if force and os.path.isfile(out_file):
        os.remove(out_file)
    errors: list[str] = []
    for backend in _medqa_backend_order(source):
        try:
            return (
                _download_medqa_modelscope()
                if backend == "modelscope"
                else _download_medqa_huggingface()
            )
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare MedQA; " + " | ".join(errors))


def _medqa_backend_order(source: Optional[str]) -> list[str]:
    return source_backend_order("medqa", source)


def _load_arc(subset: str, task: str, n: Optional[int]) -> List[Dict]:
    out = []
    for it in load_parquet("arc_easy", "test"):
        ch = it["choices"]
        labels = [str(label) for label in ch["label"]]
        gold_letter = str(it["answerKey"])
        gold_text = dict(zip(labels, ch["text"])).get(gold_letter, "")
        out.append(
            {
                "task": task,
                "kind": "mc",
                "question": format_choices(it["question"], ch["text"], labels),
                "gold": [gold_text, gold_letter],
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    return out


def load_arc_easy(n: Optional[int] = None) -> List[Dict]:
    prepare_arc_easy()
    return _load_arc("ARC-Easy", "arc_easy", n)


def load_openbookqa(n: Optional[int] = None) -> List[Dict]:
    prepare_openbookqa()
    out = []
    for it in load_parquet("openbookqa", "test"):
        ch = it["choices"]
        labels = [str(label) for label in ch["label"]]
        gold_letter = str(it["answerKey"])
        gold_text = dict(zip(labels, ch["text"])).get(gold_letter, "")
        out.append(
            {
                "task": "openbookqa",
                "kind": "mc",
                "question": format_choices(it["question_stem"], ch["text"], labels),
                "gold": [gold_text, gold_letter],
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    return out


def load_medqa(n: Optional[int] = None) -> List[Dict]:
    prepare_medqa()
    with open(os.path.join(prepared_root(), "medqa", "medqa.json")) as f:
        data = json.load(f)
    out = []
    for it in data:
        labels, texts = _split_medqa_options(it["options"])
        ans = it["answer"].strip()
        letter = None
        for label, text in zip(labels, texts):
            if text.strip() == ans:
                letter = label
                break
        out.append(
            {
                "task": "medqa",
                "kind": "mc",
                "question": format_choices(it["question"], texts, labels),
                "gold": [ans, letter],
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    return out
