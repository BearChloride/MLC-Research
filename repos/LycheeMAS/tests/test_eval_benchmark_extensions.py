import json
import os
import pathlib

from lychee_mas import REGISTRY
from lychee_mas.core.types import TaskQuery
from lychee_mas.eval import benchmarks  # noqa: F401
from lychee_mas.eval.metrics import score, score_details
from lychee_mas.eval.benchmarks import aftraj, agent_collab, choice_qa, common, locomo10, mast_data, open_agent_traces


def test_extended_benchmarks_registered():
    names = set(REGISTRY.list("benchmark"))
    assert "human_eval" in names
    assert "gaia_validation" in names
    assert "gaia_validation_level_1" in names
    assert "aftraj_audit" in names
    assert "aftraj_audit_test" in names
    assert "agent_collab_clc" in names
    assert "mast_failure" in names
    assert "open_agent_traces" in names
    assert "arc_easy" in benchmarks.PREPARERS
    assert "openbookqa" in benchmarks.PREPARERS
    assert "medqa" in benchmarks.PREPARERS
    assert "locomo10" in benchmarks.PREPARERS
    assert "aftraj_audit" in benchmarks.PREPARERS
    assert "agent_collab" in benchmarks.PREPARERS
    assert "mast_failure" in benchmarks.PREPARERS
    assert "open_agent_traces" in benchmarks.PREPARERS
    assert "aftraj" in benchmarks.SOURCE_PREPARERS
    assert "aftraj_audit" not in benchmarks.SOURCE_PREPARERS
    assert benchmarks.PREPARE_ALIASES["aftraj_audit_test"] == "aftraj"
    assert benchmarks.PREPARE_ALIASES["agent_collab_clc"] == "agent_collab"


def test_benchmark_roots_prefer_new_names_and_keep_legacy_fallback(monkeypatch):
    for name in (
        "LYCHEE_BENCHMARK_RAW_ROOT",
        "LYCHEE_BENCHMARK_PREPARED_ROOT",
        "LYCHEE_BENCHMARK_RUNS_ROOT",
        "CDM_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    # 默认根目录由 os.path.join 拼出：在 Windows 上是反斜杠，这里按平台断言（Linux 上行为不变）
    assert common.raw_root() == os.path.join("data", "benchmarks", "raw")
    assert common.prepared_root() == os.path.join("data", "benchmarks", "prepared")
    assert common.runs_root() == os.path.join("runs", "benchmarks")

    monkeypatch.setenv("CDM_DATA_ROOT", "/legacy/raw")
    assert common.raw_root() == "/legacy/raw"
    assert common.prepared_root() == "/legacy/raw"

    monkeypatch.setenv("LYCHEE_BENCHMARK_RAW_ROOT", "/new/raw")
    monkeypatch.setenv("LYCHEE_BENCHMARK_PREPARED_ROOT", "/new/prepared")
    monkeypatch.setenv("LYCHEE_BENCHMARK_RUNS_ROOT", "/new/runs")
    assert common.raw_root() == "/new/raw"
    assert common.prepared_root() == "/new/prepared"
    assert common.runs_root() == "/new/runs"
    assert common.safe_source_id("OmniData/ARC") == "OmniData--ARC"
    # 用 pathlib 复现 raw_source_dir 内部的拼接方式，跨平台一致（Windows 上盘符缺省为 \new\raw…）
    expected = str(pathlib.Path("/new/raw") / "arc_easy" / "modelscope" / "OmniData--ARC")
    assert str(common.raw_source_dir("arc_easy", "modelscope", "OmniData/ARC")) == expected


def test_copy_raw_to_prepared_creates_real_directory(tmp_path):
    raw = tmp_path / "raw" / "gaia"
    raw.mkdir(parents=True)
    (raw / "metadata.parquet").write_text("fake", encoding="utf-8")
    prepared = tmp_path / "prepared" / "gaia"

    out = common.copy_raw_to_prepared(raw, prepared)

    assert out == prepared
    assert out.is_dir()
    assert (out / "metadata.parquet").read_text(encoding="utf-8") == "fake"


def test_restore_prepared_from_raw_uses_ready_candidate(tmp_path):
    bad_raw = tmp_path / "raw" / "bad"
    good_raw = tmp_path / "raw" / "good"
    bad_raw.mkdir(parents=True)
    good_raw.mkdir(parents=True)
    (good_raw / "ready.txt").write_text("ok", encoding="utf-8")
    prepared = tmp_path / "prepared" / "dataset"

    out = common.restore_prepared_from_raw(
        "toy",
        prepared,
        [
            ("huggingface", "bad/source", bad_raw),
            ("huggingface", "good/source", good_raw),
        ],
        ready=lambda path: (path / "ready.txt").is_file(),
    )

    assert out == prepared
    assert out.is_dir()
    assert (out / "ready.txt").read_text(encoding="utf-8") == "ok"


def test_choice_qa_and_locomo_standardizers():
    medqa = choice_qa._standardize_medqa_row(
        {
            "question": "Which finding is most likely?",
            "options": {"A": "Fever", "B": "Cough", "C": "Rash", "D": "Headache"},
            "answer_idx": "C",
        }
    )
    assert medqa["answer"] == "Rash"
    assert medqa["options"][2] == "C. Rash"

    records = locomo10._load_mc10_records(
        [
            {
                "question_id": "conv-1_q1",
                "question_type": "single_hop",
                "question": "Who brought tea?",
                "answer": "Alice",
                "haystack_session_ids": ["session_1"],
                "haystack_session_datetimes": ["2024-01-01T12:00:00"],
                "haystack_sessions": [[{"role": "user", "content": "Alice brought tea."}]],
            }
        ],
        n=1,
    )
    assert records[0]["task"] == "locomo10"
    assert records[0]["gold"] == ["Alice"]
    assert "Alice brought tea" in records[0]["context"]

    original_records = locomo10._load_original_conversation_records(
        [
            {
                "sample_id": "conv-26",
                "conversation": {
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [
                        {"speaker": "Caroline", "text": "I went to support group."},
                        {"speaker": "Melanie", "text": "That sounds meaningful."},
                    ],
                },
                "qa": [
                    {
                        "question": "Where did Caroline go?",
                        "answer": "support group",
                        "evidence": ["D1:1"],
                        "category": 2,
                    }
                ],
            }
        ],
        n=1,
        max_qa_per_conv=10,
    )
    assert original_records[0]["gold"] == ["support group"]
    assert "session_1" in original_records[0]["context"]
    assert "Caroline: I went to support group." in original_records[0]["context"]



def test_gaia_scorer_uses_official_normalization():
    assert score("gaia", "Paris", "paris") == 1.0
    assert score("gaia", "$1,234", "1234") == 1.0


def test_human_eval_scorer_runs_check_function():
    gold = {
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    }
    pred = "```python\ndef add(a, b):\n    return a + b\n```"
    assert score("human_eval", pred, gold) == 1.0


def test_strict_mas_scorers():
    assert score(
        "mas_audit",
        '{"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"}',
        {"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"},
    ) == 1.0
    assert score(
        "mas_failure_taxonomy",
        '{"labels": ["1.1", "2.3"]}',
        {"labels": ["1.1", "2.3", "3.1"]},
    ) > 0.0
    assert score(
        "mas_deviation",
        '{"is_deviation": true, "deviation_types": ["wrong_route"]}',
        {"is_deviation": True, "deviation_types": ["wrong_route"]},
    ) == 1.0
    assert score(
        "mas_context_leakage",
        "Task B completed successfully without mentioning confidential token.",
        {"metric": "CLC", "expected": "Task B completed", "forbidden": ["salary=999"]},
    ) == 1.0
    assert score(
        "mas_context_leakage",
        "Task B completed. salary=999",
        {"metric": "CLC", "expected": "Task B completed", "forbidden": ["salary=999"]},
    ) == 0.0
    audit_details = score_details(
        "mas_audit",
        '{"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"}',
        {"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"},
    )
    assert audit_details["mistake_step_correct"] is True
    collab_details = score_details(
        "mas_context_leakage",
        "Task B completed.",
        {"metric": "CLC", "expected": "Task B completed", "forbidden": ["salary=999"]},
    )
    assert collab_details["leakage_detected"] is False


def test_strict_mas_loader_helpers_are_schema_tolerant(tmp_path):
    rows = aftraj._interleave_rows(
        [{"audit_label": "unsafe", "conv_id": "u1"}],
        [{"audit_label": "safe", "conv_id": "s1"}],
    )
    assert [row["conv_id"] for row in rows] == ["u1", "s1"]
    record = aftraj._to_record(
        {
            "audit_label": "unsafe",
            "query": "Solve the task",
            "messages": [{"agent_name": "Planner", "message": "wrong step"}],
            "mistake_step": "0",
            "mistake_agent": "Planner",
        },
        task_name="aftraj_audit",
    )
    assert record["gold"]["mistake_step"] == 0
    assert "Planner" in record["question"]

    task_dir = tmp_path / "data"
    task_dir.mkdir()
    (task_dir / "TASK-DATAENG-CLC-001.json").write_text(
        json.dumps(
            {
                "task_id": "TASK-DATAENG-CLC-001",
                "task_b_description": "Review the governance policy.",
                "topology": {"type": "chain"},
                "injections": {"private_fact": "salary=999"},
                "expected_outcome": "Review complete",
            }
        ),
        encoding="utf-8",
    )
    collab_rows = agent_collab.load_source_records(metric="CLC", root=tmp_path)
    assert len(collab_rows) == 1
    collab_record = agent_collab._to_record(collab_rows[0], "CLC")
    assert collab_record["kind"] == "mas_context_leakage"
    assert "salary=999" in collab_record["gold"]["forbidden"]

    labels = mast_data.extract_taxonomy_labels({"primary": "1.1", "nested": {"secondary": "2.3"}})
    assert labels == ["1.1", "2.3"]

    run = [
        {
            "run_id": "r1",
            "event_index": 0,
            "event_type": "handoff",
            "agent_name": "Reviewer",
            "is_deviation": "False",
        },
        {
            "run_id": "r1",
            "event_index": 1,
            "event_type": "tool_call",
            "agent_name": "Executor",
            "is_deviation": "true",
            "deviation_label": "wrong_route",
        },
    ]
    assert open_agent_traces._gold(run)["deviation_types"] == ["wrong_route"]
    assert open_agent_traces.deviation_score_details(
        '{"is_deviation": true, "deviation_types": ["wrong_route"]}',
        open_agent_traces._gold(run),
    )["score"] == 1.0
