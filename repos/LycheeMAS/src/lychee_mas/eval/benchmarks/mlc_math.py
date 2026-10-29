"""MLC（ACL 2025）复现所需的数学推理数据集：AddSub / SingleEQ / MultiArith / ASDiv / SVAMP / MATH。

论文：*Advancing Collaborative Debates with Role Differentiation through Multi-Agent
Reinforcement Learning*（ACL 2025 Long Papers, ``2025.acl-long.1105``）在 7 个数学
推理数据集上评测。其中 GSM8K 已由 ``gsm8k.py`` 承载，其余 6 个由本模块承载。

数据来源是**本地已下载的原始文件**，本模块不联网；原始文件位置按以下顺序解析：

1. 各数据集专属环境变量（如 ``LYCHEE_ADDSUB_JSONL``）；
2. ``LYCHEE_MLC_DATA_ROOT`` 下的固定相对路径（推荐：指向存有全部数据集的目录）；
3. 仓库内 ``data/benchmarks/raw/mlc/``（默认位置，已被 .gitignore 忽略）。

``prepare_*`` 把原始文件规范化成 ``<prepared_root>/<task>/<task>.jsonl``（字段
``{question, gold}``）；``load_*`` 只读该缓存并输出框架统一记录
``{task, kind, question, gold, context}``。**只有 prepare 需要原始文件**，
之后 load 在任何工作目录都能跑，也不依赖下载。

gold 口径（三档，实验记录里要写清楚）：

* **数值型**（AddSub / SingleEQ / MultiArith / SVAMP）规整成最短数值字符串：
  ``43.0 -> "43"``、``51.5 -> "51.5"``。这一步是必须的——``metrics.score_exact``
  先取两边最后一个数字再**逐字比较**，若 gold 留成 ``"43.0"`` 而模型答 ``"43"``
  会被判错。
* **ASDiv** 的 ``<Answer>`` 形如 ``"9 (apples)"``：剥掉单位只留数字；少量非数值答案
  （Yes/No、时刻、人名，实测 146/2305）保留原文，靠 ``score_exact`` 的规整子串分支
  比较。prepare 会把两类的计数打印出来，便于判断评测覆盖。
* **MATH** 的答案在 ``solution`` 的 ``\\boxed{}`` 内，**保留 LaTeX 原样**，走
  ``kind="aime"``（数值 + sympy 符号等价）。实测 test 5000 条全部含 ``\\boxed{}``
  （缺一条即报错）；train 7500 条里有 4 条缺 ``\\boxed{}``，只作训练素材，
  prepare 时跳过并计数打印。

题面保持原始文本，不追加任何指令行（``gsm8k.py`` 会追加 "Give the final numeric
answer."，跨数据集比较时需注意这一处口径差异，可在实验脚本里统一处理）。
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from .common import prepared_root, raw_root

# 数据集 -> (专属环境变量, 相对 LYCHEE_MLC_DATA_ROOT 的候选路径，按优先级排列)
_SOURCE_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "addsub": ("LYCHEE_ADDSUB_JSONL", ("mawps/addsub.jsonl",)),
    "singleeq": ("LYCHEE_SINGLEEQ_JSONL", ("mawps/singleeq.jsonl",)),
    "multiarith": ("LYCHEE_MULTIARITH_JSONL", ("mawps/multiarith.jsonl",)),
    "asdiv": ("LYCHEE_ASDIV_XML", ("ASDiv_ASDiv.xml/dataset/ASDiv.xml",)),
    "svamp": ("LYCHEE_SVAMP_JSON", ("SVAMP_SVAMP.json/SVAMP.json",)),
    "math": ("LYCHEE_MATH_DIR", ("Math",)),
}
_DATA_ROOT_ENV = "LYCHEE_MLC_DATA_ROOT"

# MATH 的 7 个学科子目录（每个学科各一份 train / test parquet）
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

# kind 口径：数值答案走 exact，LaTeX 答案走 aime（数值 + sympy 符号等价）
_KINDS = {
    "addsub": "exact",
    "singleeq": "exact",
    "multiarith": "exact",
    "asdiv": "exact",
    "svamp": "exact",
    "math": "aime",
}

_STRICT_NUMBER = re.compile(r"^\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:\(.*\))?\s*$")


# ====================== gold 规整 ======================
def numeric_gold(value: Any) -> str:
    """数值字段（mawps ``target`` / SVAMP ``Answer``）→ 最短数值字符串。

    ``43.0 -> "43"``、``51.5 -> "51.5"``。非数值输入原样返回文本，不做静默兜底。
    """
    if isinstance(value, bool):
        raise TypeError("布尔值不是合法的数学答案")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        match = _STRICT_NUMBER.match(text)
        if match is None:
            return text
        number = float(match.group(1).replace(",", ""))
    if number == int(number):
        return str(int(number))
    return f"{number:g}"


def asdiv_gold(text: str) -> str:
    """ASDiv ``<Answer>``：``"9 (apples)" -> "9"``；非数值答案原样保留（去空白）。"""
    match = _STRICT_NUMBER.match(text or "")
    if match is None:
        return (text or "").strip()
    return numeric_gold(match.group(1))


def is_numeric_gold(gold: str) -> bool:
    """该 gold 是否已规整成纯数值（供 prepare 统计可打分比例）。"""
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", gold or ""))


def extract_boxed(solution: str) -> str:
    r"""从 MATH 的 ``solution`` 里取 ``\boxed{...}`` 内容。

    花括号配平，支持嵌套（如 ``\boxed{\frac{1}{2}}``）。
    """
    marker = solution.find(r"\boxed")
    if marker < 0:
        return ""
    start = solution.find("{", marker)
    if start < 0:
        return ""
    depth = 0
    for idx in range(start, len(solution)):
        char = solution[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1 : idx].strip()
    return ""


# ====================== 原始文件定位 ======================
def _source_candidates(task: str) -> List[Path]:
    env_name, relatives = _SOURCE_SPECS[task]
    candidates: List[Path] = []
    override = os.environ.get(env_name)
    if override:
        candidates.append(Path(override).expanduser())
    root = os.environ.get(_DATA_ROOT_ENV)
    if root:
        candidates.extend(Path(root).expanduser() / rel for rel in relatives)
    candidates.extend(Path(raw_root()) / "mlc" / rel for rel in relatives)
    return candidates


def _resolve_source(task: str, source: Optional[str]) -> Path:
    candidates = ([Path(source).expanduser()] if source else []) + _source_candidates(task)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    env_name, _ = _SOURCE_SPECS[task]
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"{task}: 原始数据不存在（本地优先，不联网）。已尝试：\n  {tried}\n"
        f"请设置 {env_name}=<文件路径>，或设置 {_DATA_ROOT_ENV}=<数据集总目录>。"
    )


# ====================== 规范化缓存读写 ======================
def _prepared_file(task: str, name: Optional[str] = None) -> Path:
    return Path(prepared_root()) / task / f"{name or task}.jsonl"


def _cache_ready(task: str, name: Optional[str] = None) -> bool:
    path = _prepared_file(task, name)
    return path.is_file() and path.stat().st_size > 0


def _write_records(task: str, records: List[dict], name: Optional[str] = None) -> Path:
    path = _prepared_file(task, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def _read_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _prepare_simple(
    task: str,
    parser: Callable[[Path], List[dict]],
    force: bool,
    source: Optional[str],
) -> str:
    path = _prepared_file(task)
    if not force and _cache_ready(task):
        return str(path.parent)
    origin = _resolve_source(task, source)
    records = parser(origin)
    if not records:
        raise ValueError(f"{task}: 从 {origin} 解析出 0 条记录")
    path = _write_records(task, records)
    print(f"[prepare] {task}: {len(records)} 题 <- {origin} -> {path}", flush=True)
    return str(path.parent)


# ====================== 各原始格式的解析 ======================
def _parse_mawps(path: Path) -> List[dict]:
    records: List[dict] = []
    for item in _read_records(path):
        question = str(item["input"]).strip()
        if not question:
            raise ValueError(f"mawps 记录缺 question: {path}")
        records.append({"question": question, "gold": numeric_gold(item["target"])})
    return records


def _xml_tag(block: str, name: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", block, re.S)
    return html.unescape(match.group(1)).strip() if match else ""


def _parse_asdiv(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    problems = re.findall(r"<Problem\b[^>]*>(.*?)</Problem>", text, re.S)
    if not problems:
        raise ValueError(f"ASDiv 未解析出任何 <Problem>：{path}")
    records: List[dict] = []
    numeric = 0
    for block in problems:
        parts = (_xml_tag(block, "Body"), _xml_tag(block, "Question"))
        question = " ".join(part for part in parts if part)
        if not question:
            continue
        gold = asdiv_gold(_xml_tag(block, "Answer"))
        numeric += 1 if is_numeric_gold(gold) else 0
        records.append({"question": question.strip(), "gold": gold})
    print(
        f"[prepare] asdiv: 数值 gold {numeric}/{len(records)}，其余为非数值答案（原样保留）",
        flush=True,
    )
    return records


def _parse_svamp(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[dict] = []
    for item in data:
        question = f"{str(item['Body']).strip()} {str(item['Question']).strip()}".strip()
        out.append({"question": question, "gold": numeric_gold(item["Answer"])})
    return out


def _parse_math(root: Path) -> tuple[List[dict], List[dict]]:
    """按 7 个学科目录读取 train / test parquet，返回 (test, train)。

    test 分片要求每题都能抽出 ``\\boxed{}``（实测 5000/5000 满足），缺一条即显式报错；
    train 分片只作训练素材，缺 ``\\boxed{}`` 的极少数样本（实测 4/7500）跳过并计数打印，
    不做数字兜底。
    """
    import pandas as pd  # 惰性导入：heavy dep 只在实际解析数据时触达

    splits: dict[str, List[dict]] = {"test": [], "train": []}
    skipped = 0
    for subject in MATH_SUBJECTS:
        subject_dir = root / subject
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"MATH 学科目录缺失：{subject_dir}")
        for split in ("test", "train"):
            files = sorted(subject_dir.glob(f"{split}-*.parquet"))
            if not files:
                raise FileNotFoundError(f"MATH 缺少 {split} 分片：{subject_dir}")
            for file in files:
                frame = pd.read_parquet(file)
                for row in frame.itertuples(index=False):
                    gold = extract_boxed(str(row.solution))
                    if not gold:
                        if split == "test":
                            raise ValueError(f"MATH test 题面缺少 \\boxed{{}} 答案：{file}")
                        skipped += 1
                        continue
                    splits[split].append(
                        {"question": str(row.problem).strip(), "gold": gold, "subject": subject}
                    )
    if skipped:
        note = "（仅作训练素材，不做数字兜底）"
        print(
            f"[prepare] math: train 跳过 {skipped} 条缺 \\boxed{{}} 的样本{note}",
            flush=True,
        )
    return splits["test"], splits["train"]


# ====================== prepare（写入 prepared 缓存） ======================
def prepare_addsub(force: bool = False, source: Optional[str] = None) -> str:
    return _prepare_simple("addsub", _parse_mawps, force, source)


def prepare_singleeq(force: bool = False, source: Optional[str] = None) -> str:
    return _prepare_simple("singleeq", _parse_mawps, force, source)


def prepare_multiarith(force: bool = False, source: Optional[str] = None) -> str:
    return _prepare_simple("multiarith", _parse_mawps, force, source)


def prepare_asdiv(force: bool = False, source: Optional[str] = None) -> str:
    return _prepare_simple("asdiv", _parse_asdiv, force, source)


def prepare_svamp(force: bool = False, source: Optional[str] = None) -> str:
    return _prepare_simple("svamp", _parse_svamp, force, source)


def prepare_math(force: bool = False, source: Optional[str] = None) -> str:
    """MATH 全量：test 5000 题写 ``math.jsonl``，train 7500 题写 ``math_train.jsonl``。"""
    if not force and _cache_ready("math") and _cache_ready("math", "math_train"):
        return str(_prepared_file("math").parent)
    root = _resolve_source("math", source)
    test, train = _parse_math(root if root.is_dir() else root.parent)
    test_path = _write_records("math", test)
    train_path = _write_records("math", train, name="math_train")
    print(
        f"[prepare] math: test {len(test)} 题 / train {len(train)} 题 <- {root}\n"
        f"          test -> {test_path}\n          train -> {train_path}",
        flush=True,
    )
    return str(test_path.parent)


# ====================== load（读 prepared 缓存） ======================
def _load_records(task: str, n: Optional[int] = None, name: Optional[str] = None) -> List[dict]:
    path = _prepared_file(task, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"{task} 尚未 prepare：{path}。请先运行 "
            f"`python scripts/prepare_mlc_math.py --tasks {task}` "
            f"或调用 benchmarks.prepare({task!r})。"
        )
    kind = _KINDS[task]
    out: List[dict] = []
    for record in _read_records(path):
        out.append(
            {
                "task": task,
                "kind": kind,
                "question": record["question"],
                "gold": record["gold"],
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    if not out:
        raise RuntimeError(f"{task} 加载为空：{path}")
    return out


def load_addsub(n: Optional[int] = None) -> List[dict]:
    return _load_records("addsub", n)


def load_singleeq(n: Optional[int] = None) -> List[dict]:
    return _load_records("singleeq", n)


def load_multiarith(n: Optional[int] = None) -> List[dict]:
    return _load_records("multiarith", n)


def load_asdiv(n: Optional[int] = None) -> List[dict]:
    return _load_records("asdiv", n)


def load_svamp(n: Optional[int] = None) -> List[dict]:
    return _load_records("svamp", n)


def load_math(n: Optional[int] = None) -> List[dict]:
    return _load_records("math", n)


def load_math_train(n: Optional[int] = None) -> List[dict]:
    """MATH train 分片（7500 题）——供 MLC / MASPO 这类需要训练素材的方法取用。"""
    return _load_records("math", n, name="math_train")
