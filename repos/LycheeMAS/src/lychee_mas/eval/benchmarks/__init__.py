"""Benchmark registry for LycheeMAS evaluation.

Each loader returns records shaped as ``{task, kind, question, gold, context}``.
The entry points stay intentionally small: benchmark-specific loading and data
preparation live in sibling modules, while this file only exposes public
``load``/``prepare`` registries and registers benchmark classes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ...core.registry import REGISTRY
from .aftraj import ensure_source as ensure_aftraj_source
from .aftraj import load_aftraj_audit, load_aftraj_audit_test
from .agent_collab import ensure_source as ensure_agent_collab_source
from .agent_collab import (
    load_agent_collab_clc,
    load_agent_collab_cpr,
    load_agent_collab_idr,
    load_agent_collab_rtd,
)
from .aime_2024 import load_aime_2024, prepare_aime_2024
from .choice_qa import (
    load_arc_easy,
    load_medqa,
    load_openbookqa,
    prepare_arc_easy,
    prepare_medqa,
    prepare_openbookqa,
)
from .common import DATA_BACKENDS, prepared_root, processed_root, raw_root
from .gaia import (
    ensure_full_source as ensure_gaia_full_source,
)
from .gaia import (
    ensure_validation_source as ensure_gaia_validation_source,
)
from .gaia import (
    load_gaia_validation,
    load_gaia_validation_level_1,
    load_gaia_validation_level_2,
    load_gaia_validation_level_3,
)
from .gsm8k import load_gsm8k, prepare_gsm8k
from .human_eval import ensure_source as ensure_human_eval_source
from .human_eval import load_human_eval
from .locomo10 import load_locomo10, prepare_locomo10
from .mast_data import ensure_source as ensure_mast_source
from .mast_data import load_mast_failure
from .math500 import load_math500, prepare_math500
from .mlc_math import (
    load_addsub,
    load_asdiv,
    load_math,
    load_multiarith,
    load_singleeq,
    load_svamp,
    prepare_addsub,
    prepare_asdiv,
    prepare_math,
    prepare_multiarith,
    prepare_singleeq,
    prepare_svamp,
)
from .open_agent_traces import ensure_source as ensure_open_agent_traces_source
from .open_agent_traces import load_open_agent_traces

RAW = raw_root()
PREPARED = prepared_root()
PROCESSED = processed_root()

REASONING_POLE = ("gsm8k", "aime_2024", "math500")
# MLC（ACL 2025, 2025.acl-long.1105）复现用的数学推理数据集族：GSM8K 走 gsm8k.py，
# 其余 6 个走 mlc_math.py（本地数据，不联网；先把原始文件落到 raw/mlc 或设 LYCHEE_MLC_DATA_ROOT）。
MLC_MATH_TASKS = ("addsub", "singleeq", "multiarith", "asdiv", "svamp", "math")
FACT_POLE = ("medqa", "openbookqa", "arc_easy")
MEMORY_TASKS = ("locomo10",)
CODE_TASKS = ("human_eval",)
TOOL_TASKS = (
    "gaia_validation",
    "gaia_validation_level_1",
    "gaia_validation_level_2",
    "gaia_validation_level_3",
)
MAS_DIAGNOSTIC_TASKS = (
    "aftraj_audit",
    "aftraj_audit_test",
    "mast_failure",
    "open_agent_traces",
)
MAS_COLLAB_TASKS = (
    "agent_collab_idr",
    "agent_collab_rtd",
    "agent_collab_cpr",
    "agent_collab_clc",
)

FULL_PREPARE_TARGETS = (
    "gsm8k",
    "aime_2024",
    "math500",
    "arc_easy",
    "openbookqa",
    "medqa",
    "locomo10",
    "human_eval",
    "gaia",
    "aftraj",
    "agent_collab",
    "mast_data",
    "open_agent_traces",
)

BENCHMARK_STRUCTURE = [
    {
        "benchmark_source": "GSM8K",
        "full_prepare_target": "gsm8k",
        "prepare_targets": ["gsm8k"],
        "runnable_tasks": ["gsm8k"],
        "kinds": ["exact"],
    },
    {
        "benchmark_source": "AIME 2024",
        "full_prepare_target": "aime_2024",
        "prepare_targets": ["aime_2024"],
        "runnable_tasks": ["aime_2024"],
        "kinds": ["aime"],
    },
    {
        "benchmark_source": "MATH-500",
        "full_prepare_target": "math500",
        "prepare_targets": ["math500"],
        "runnable_tasks": ["math500"],
        "kinds": ["aime"],
    },
    {
        "benchmark_source": "AddSub (MAWPS)",
        "full_prepare_target": "addsub",
        "prepare_targets": ["addsub"],
        "runnable_tasks": ["addsub"],
        "kinds": ["exact"],
    },
    {
        "benchmark_source": "SingleEQ (MAWPS)",
        "full_prepare_target": "singleeq",
        "prepare_targets": ["singleeq"],
        "runnable_tasks": ["singleeq"],
        "kinds": ["exact"],
    },
    {
        "benchmark_source": "MultiArith (MAWPS)",
        "full_prepare_target": "multiarith",
        "prepare_targets": ["multiarith"],
        "runnable_tasks": ["multiarith"],
        "kinds": ["exact"],
    },
    {
        "benchmark_source": "ASDiv",
        "full_prepare_target": "asdiv",
        "prepare_targets": ["asdiv"],
        "runnable_tasks": ["asdiv"],
        "kinds": ["exact"],
    },
    {
        "benchmark_source": "SVAMP",
        "full_prepare_target": "svamp",
        "prepare_targets": ["svamp"],
        "runnable_tasks": ["svamp"],
        "kinds": ["exact"],
    },
    {
        "benchmark_source": "MATH",
        "full_prepare_target": "math",
        "prepare_targets": ["math"],
        "runnable_tasks": ["math"],
        "kinds": ["aime"],
    },
    {
        "benchmark_source": "ARC-Easy",
        "full_prepare_target": "arc_easy",
        "prepare_targets": ["arc_easy"],
        "runnable_tasks": ["arc_easy"],
        "kinds": ["mc"],
    },
    {
        "benchmark_source": "OpenBookQA",
        "full_prepare_target": "openbookqa",
        "prepare_targets": ["openbookqa"],
        "runnable_tasks": ["openbookqa"],
        "kinds": ["mc"],
    },
    {
        "benchmark_source": "MedQA",
        "full_prepare_target": "medqa",
        "prepare_targets": ["medqa"],
        "runnable_tasks": ["medqa"],
        "kinds": ["mc"],
    },
    {
        "benchmark_source": "LoCoMo10",
        "full_prepare_target": "locomo10",
        "prepare_targets": ["locomo10"],
        "runnable_tasks": ["locomo10"],
        "kinds": ["f1"],
    },
    {
        "benchmark_source": "HumanEval",
        "full_prepare_target": "human_eval",
        "prepare_targets": ["human_eval"],
        "runnable_tasks": ["human_eval"],
        "kinds": ["human_eval"],
    },
    {
        "benchmark_source": "GAIA",
        "full_prepare_target": "gaia",
        "prepare_targets": [
            "gaia",
            "gaia_validation",
            "gaia_validation_level_1",
            "gaia_validation_level_2",
            "gaia_validation_level_3",
        ],
        "runnable_tasks": [
            "gaia_validation",
            "gaia_validation_level_1",
            "gaia_validation_level_2",
            "gaia_validation_level_3",
        ],
        "kinds": ["gaia"],
    },
    {
        "benchmark_source": "AFTraj-2K",
        "full_prepare_target": "aftraj",
        "prepare_targets": ["aftraj", "aftraj_audit", "aftraj_audit_test"],
        "runnable_tasks": ["aftraj_audit", "aftraj_audit_test"],
        "kinds": ["mas_audit"],
    },
    {
        "benchmark_source": "AgentCollabBench",
        "full_prepare_target": "agent_collab",
        "prepare_targets": [
            "agent_collab",
            "agent_collab_idr",
            "agent_collab_rtd",
            "agent_collab_cpr",
            "agent_collab_clc",
        ],
        "runnable_tasks": [
            "agent_collab_idr",
            "agent_collab_rtd",
            "agent_collab_cpr",
            "agent_collab_clc",
        ],
        "kinds": [
            "mas_instruction_decay",
            "mas_tracer_durability",
            "mas_consensus_pollution",
            "mas_context_leakage",
        ],
    },
    {
        "benchmark_source": "MAST-Data",
        "full_prepare_target": "mast_data",
        "prepare_targets": ["mast_data", "mast_failure"],
        "runnable_tasks": ["mast_failure"],
        "kinds": ["mas_failure_taxonomy"],
    },
    {
        "benchmark_source": "Open Agent Traces",
        "full_prepare_target": "open_agent_traces",
        "prepare_targets": ["open_agent_traces"],
        "runnable_tasks": ["open_agent_traces"],
        "kinds": ["mas_deviation"],
    },
]


BENCHMARK_MAPPINGS = {
    "GSM8K": [
        {"prepare_target": "gsm8k", "runnable_tasks": ["gsm8k"], "kinds": ["exact"]},
    ],
    "AIME 2024": [
        {"prepare_target": "aime_2024", "runnable_tasks": ["aime_2024"], "kinds": ["aime"]},
    ],
    "MATH-500": [
        {"prepare_target": "math500", "runnable_tasks": ["math500"], "kinds": ["aime"]},
    ],
    "ARC-Easy": [
        {"prepare_target": "arc_easy", "runnable_tasks": ["arc_easy"], "kinds": ["mc"]},
    ],
    "OpenBookQA": [
        {"prepare_target": "openbookqa", "runnable_tasks": ["openbookqa"], "kinds": ["mc"]},
    ],
    "MedQA": [
        {"prepare_target": "medqa", "runnable_tasks": ["medqa"], "kinds": ["mc"]},
    ],
    "LoCoMo10": [
        {"prepare_target": "locomo10", "runnable_tasks": ["locomo10"], "kinds": ["f1"]},
    ],
    "HumanEval": [
        {"prepare_target": "human_eval", "runnable_tasks": ["human_eval"], "kinds": ["human_eval"]},
    ],
    "GAIA": [
        {
            "prepare_target": "gaia",
            "runnable_tasks": [],
            "kinds": [],
            "note": "full dataset prepare only",
        },
        {
            "prepare_target": "gaia_validation",
            "runnable_tasks": ["gaia_validation"],
            "kinds": ["gaia"],
        },
        {
            "prepare_target": "gaia_validation_level_1",
            "runnable_tasks": ["gaia_validation_level_1"],
            "kinds": ["gaia"],
        },
        {
            "prepare_target": "gaia_validation_level_2",
            "runnable_tasks": ["gaia_validation_level_2"],
            "kinds": ["gaia"],
        },
        {
            "prepare_target": "gaia_validation_level_3",
            "runnable_tasks": ["gaia_validation_level_3"],
            "kinds": ["gaia"],
        },
    ],
    "AFTraj-2K": [
        {
            "prepare_target": "aftraj",
            "runnable_tasks": ["aftraj_audit", "aftraj_audit_test"],
            "kinds": ["mas_audit"],
        },
        {
            "prepare_target": "aftraj_audit",
            "runnable_tasks": ["aftraj_audit"],
            "kinds": ["mas_audit"],
        },
        {
            "prepare_target": "aftraj_audit_test",
            "runnable_tasks": ["aftraj_audit_test"],
            "kinds": ["mas_audit"],
        },
    ],
    "AgentCollabBench": [
        {
            "prepare_target": "agent_collab",
            "runnable_tasks": [
                "agent_collab_idr",
                "agent_collab_rtd",
                "agent_collab_cpr",
                "agent_collab_clc",
            ],
            "kinds": [
                "mas_instruction_decay",
                "mas_tracer_durability",
                "mas_consensus_pollution",
                "mas_context_leakage",
            ],
        },
        {
            "prepare_target": "agent_collab_idr",
            "runnable_tasks": ["agent_collab_idr"],
            "kinds": ["mas_instruction_decay"],
        },
        {
            "prepare_target": "agent_collab_rtd",
            "runnable_tasks": ["agent_collab_rtd"],
            "kinds": ["mas_tracer_durability"],
        },
        {
            "prepare_target": "agent_collab_cpr",
            "runnable_tasks": ["agent_collab_cpr"],
            "kinds": ["mas_consensus_pollution"],
        },
        {
            "prepare_target": "agent_collab_clc",
            "runnable_tasks": ["agent_collab_clc"],
            "kinds": ["mas_context_leakage"],
        },
    ],
    "MAST-Data": [
        {
            "prepare_target": "mast_data",
            "runnable_tasks": ["mast_failure"],
            "kinds": ["mas_failure_taxonomy"],
        },
        {
            "prepare_target": "mast_failure",
            "runnable_tasks": ["mast_failure"],
            "kinds": ["mas_failure_taxonomy"],
        },
    ],
    "Open Agent Traces": [
        {
            "prepare_target": "open_agent_traces",
            "runnable_tasks": ["open_agent_traces"],
            "kinds": ["mas_deviation"],
        },
    ],
    "AddSub (MAWPS)": [
        {
            "prepare_target": "addsub",
            "runnable_tasks": ["addsub"],
            "kinds": ["exact"],
        },
    ],
    "SingleEQ (MAWPS)": [
        {
            "prepare_target": "singleeq",
            "runnable_tasks": ["singleeq"],
            "kinds": ["exact"],
        },
    ],
    "MultiArith (MAWPS)": [
        {
            "prepare_target": "multiarith",
            "runnable_tasks": ["multiarith"],
            "kinds": ["exact"],
        },
    ],
    "ASDiv": [
        {
            "prepare_target": "asdiv",
            "runnable_tasks": ["asdiv"],
            "kinds": ["exact"],
        },
    ],
    "SVAMP": [
        {
            "prepare_target": "svamp",
            "runnable_tasks": ["svamp"],
            "kinds": ["exact"],
        },
    ],
    "MATH": [
        {
            "prepare_target": "math",
            "runnable_tasks": ["math"],
            "kinds": ["aime"],
        },
    ],
}


for _row in BENCHMARK_STRUCTURE:
    _row["mappings"] = BENCHMARK_MAPPINGS[_row["benchmark_source"]]


def prepare_human_eval(force: bool = False, source: Optional[str] = None) -> str:
    return str(ensure_human_eval_source(force_download=force, source=source))


def prepare_gaia(force: bool = False, source: Optional[str] = None) -> str:
    return str(ensure_gaia_full_source(force_download=force, source=source))


def prepare_gaia_validation(force: bool = False, source: Optional[str] = None) -> str:
    return str(ensure_gaia_validation_source(force_download=force, source=source))


def prepare_aftraj(force: bool = False, source: Optional[str] = None) -> str:
    return str(ensure_aftraj_source(force_download=force, source=source))


def prepare_agent_collab(force: bool = False, source: Optional[str] = None) -> str:
    return str(ensure_agent_collab_source(force_download=force, source=source))


def prepare_mast_data(force: bool = False, source: Optional[str] = None) -> str:
    return str(ensure_mast_source(force_download=force, source=source))


def prepare_open_agent_traces(force: bool = False, source: Optional[str] = None) -> str:
    return str(ensure_open_agent_traces_source(force_download=force, source=source))


SOURCE_PREPARERS: dict[str, Callable[[bool, Optional[str]], str]] = {
    "gsm8k": prepare_gsm8k,
    "aime_2024": prepare_aime_2024,
    "math500": prepare_math500,
    "arc_easy": prepare_arc_easy,
    "openbookqa": prepare_openbookqa,
    "medqa": prepare_medqa,
    "locomo10": prepare_locomo10,
    "human_eval": prepare_human_eval,
    "gaia": prepare_gaia,
    "gaia_validation": prepare_gaia_validation,
    "aftraj": prepare_aftraj,
    "agent_collab": prepare_agent_collab,
    "mast_data": prepare_mast_data,
    "open_agent_traces": prepare_open_agent_traces,
    # MLC（ACL 2025）数学数据集族：本地文件优先，prepare 时需原始数据在位
    "addsub": prepare_addsub,
    "singleeq": prepare_singleeq,
    "multiarith": prepare_multiarith,
    "asdiv": prepare_asdiv,
    "svamp": prepare_svamp,
    "math": prepare_math,
}

PREPARE_ALIASES: dict[str, str] = {
    "gaia_validation_level_1": "gaia_validation",
    "gaia_validation_level_2": "gaia_validation",
    "gaia_validation_level_3": "gaia_validation",
    "aftraj_audit": "aftraj",
    "aftraj_audit_test": "aftraj",
    "agent_collab_idr": "agent_collab",
    "agent_collab_rtd": "agent_collab",
    "agent_collab_cpr": "agent_collab",
    "agent_collab_clc": "agent_collab",
    "mast_failure": "mast_data",
}

PREPARERS: dict[str, Callable[[bool, Optional[str]], str]] = {
    **SOURCE_PREPARERS,
    **{alias: SOURCE_PREPARERS[source] for alias, source in PREPARE_ALIASES.items()},
}


def prepare(task: str, force: bool = False, source: Optional[str] = None) -> str:
    try:
        return PREPARERS[task](force, source)
    except KeyError as exc:
        available = ", ".join(sorted(PREPARERS))
        raise KeyError(f"No preparer for benchmark {task!r}; available: {available}") from exc


LOADERS = {
    "gsm8k": load_gsm8k,
    "aime_2024": load_aime_2024,
    "math500": load_math500,
    "arc_easy": load_arc_easy,
    "openbookqa": load_openbookqa,
    "medqa": load_medqa,
    "locomo10": load_locomo10,
    "human_eval": load_human_eval,
    "gaia_validation": load_gaia_validation,
    "gaia_validation_level_1": load_gaia_validation_level_1,
    "gaia_validation_level_2": load_gaia_validation_level_2,
    "gaia_validation_level_3": load_gaia_validation_level_3,
    "aftraj_audit": load_aftraj_audit,
    "aftraj_audit_test": load_aftraj_audit_test,
    "agent_collab_idr": load_agent_collab_idr,
    "agent_collab_rtd": load_agent_collab_rtd,
    "agent_collab_cpr": load_agent_collab_cpr,
    "agent_collab_clc": load_agent_collab_clc,
    "mast_failure": load_mast_failure,
    "open_agent_traces": load_open_agent_traces,
    # MLC（ACL 2025）数学数据集族（read prepared 缓存；gsm8k 见上）
    "addsub": load_addsub,
    "singleeq": load_singleeq,
    "multiarith": load_multiarith,
    "asdiv": load_asdiv,
    "svamp": load_svamp,
    "math": load_math,
}


def load(task: str, n: Optional[int] = None) -> List[Dict]:
    return LOADERS[task](n=n)


class _Benchmark:
    """Uniform registry adapter: construction is lazy and does not touch data."""

    task: str = ""

    def __init__(self, n: Optional[int] = None):
        self.n = n

    def load(self, n: Optional[int] = None) -> List[Dict]:
        return load(self.task, n=self.n if n is None else n)


def _register_benchmarks() -> None:
    for _task in LOADERS:
        cls = type(f"Benchmark_{_task}", (_Benchmark,), {"task": _task, "name": _task})
        REGISTRY.register("benchmark", _task)(cls)


_register_benchmarks()

__all__ = [
    "load",
    "prepare",
    "PREPARERS",
    "SOURCE_PREPARERS",
    "PREPARE_ALIASES",
    "LOADERS",
    "FULL_PREPARE_TARGETS",
    "BENCHMARK_STRUCTURE",
    "BENCHMARK_MAPPINGS",
    "DATA_BACKENDS",
    "RAW",
    "PREPARED",
    "PROCESSED",
    "REASONING_POLE",
    "MLC_MATH_TASKS",
    "FACT_POLE",
    "MEMORY_TASKS",
    "CODE_TASKS",
    "TOOL_TASKS",
    "MAS_DIAGNOSTIC_TASKS",
    "MAS_COLLAB_TASKS",
]
