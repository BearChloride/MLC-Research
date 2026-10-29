"""mlc_math（MLC/ACL 2025 数学数据集族）的离线测试：不联网、不依赖真实数据集。

覆盖三件事：gold 规整口径、原始格式解析（jsonl / xml / json / parquet）、
以及 prepare→load 两段式与显式报错路径。
"""
import json

import pytest
from lychee_mas.eval import benchmarks  # noqa: F401  触发注册
from lychee_mas.eval.benchmarks import mlc_math
from lychee_mas.eval.task_config import TASK_CONFIG, extractor_for_task, team_name_for_task


def _use_prepared_root(monkeypatch, tmp_path):
    """把 prepared 缓存指向临时目录，避免污染仓库内 data/。"""
    prepared = tmp_path / "prepared"
    monkeypatch.setenv("LYCHEE_BENCHMARK_PREPARED_ROOT", str(prepared))
    return prepared


# ====================== gold 规整 ======================
def test_numeric_gold_normalizes_float_and_string_forms():
    assert mlc_math.numeric_gold(43.0) == "43"
    assert mlc_math.numeric_gold(39) == "39"
    assert mlc_math.numeric_gold(51.5) == "51.5"
    assert mlc_math.numeric_gold("-3.0") == "-3"
    assert mlc_math.numeric_gold("1,000") == "1000"


def test_numeric_gold_rejects_bool():
    with pytest.raises(TypeError, match="布尔值"):
        mlc_math.numeric_gold(True)


def test_asdiv_gold_strips_unit_and_keeps_non_numeric_text():
    assert mlc_math.asdiv_gold("9 (apples)") == "9"
    assert mlc_math.asdiv_gold("16 (oranges)") == "16"
    # 非数值答案原样保留，不做数字兜底
    assert mlc_math.asdiv_gold("Yes") == "Yes"
    assert mlc_math.asdiv_gold("3:30 p.m.") == "3:30 p.m."
    assert mlc_math.asdiv_gold("February 3rd") == "February 3rd"


def test_is_numeric_gold():
    assert mlc_math.is_numeric_gold("43")
    assert mlc_math.is_numeric_gold("-3.5")
    assert not mlc_math.is_numeric_gold("Yes")


def test_extract_boxed_handles_nested_braces():
    assert mlc_math.extract_boxed(r"therefore $\boxed{2}$ asymptotes") == "2"
    assert mlc_math.extract_boxed(r"answer is \boxed{\frac{14}{3}}") == r"\frac{14}{3}"
    assert mlc_math.extract_boxed(r"no box here") == ""
    assert mlc_math.extract_boxed(r"\boxed{") == ""


# ====================== 原始格式解析 + prepare/load ======================
def test_prepare_and_load_mawps(monkeypatch, tmp_path):
    _use_prepared_root(monkeypatch, tmp_path)
    mawps = tmp_path / "mawps"
    mawps.mkdir()
    (mawps / "addsub.jsonl").write_text(
        json.dumps({"input": "Joan found 70 seashells .", "target": 43.0}) + "\n"
        + json.dumps({"input": "Sam has 5 apples .", "target": 5}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LYCHEE_ADDSUB_JSONL", str(mawps / "addsub.jsonl"))

    assert mlc_math.prepare_addsub() == str(tmp_path / "prepared" / "addsub")
    records = benchmarks.load("addsub")

    assert [r["gold"] for r in records] == ["43", "5"]  # 43.0 -> "43"（否则 exact 会误判）
    assert records[0]["task"] == "addsub"
    assert records[0]["kind"] == "exact"
    assert records[0]["context"] is None


def test_prepare_and_load_asdiv(monkeypatch, tmp_path):
    _use_prepared_root(monkeypatch, tmp_path)
    xml = tmp_path / "ASDiv.xml"
    xml.write_text(
        "<Machine-Reading-Corpus-File><ProblemSet>"
        '<Problem ID="1" Grade="1"><Body>Seven apples.</Body>'
        "<Question>How many apples?</Question><Answer>7 (apples)</Answer></Problem>"
        '<Problem ID="2" Grade="1"><Body>Asked if it rained.</Body>'
        "<Question>Did it rain?</Question><Answer>Yes</Answer></Problem>"
        "</ProblemSet></Machine-Reading-Corpus-File>",
        encoding="utf-8",
    )
    monkeypatch.setenv("LYCHEE_ASDIV_XML", str(xml))

    mlc_math.prepare_asdiv(force=True)
    records = benchmarks.load("asdiv")

    assert len(records) == 2
    assert records[0]["gold"] == "7"
    assert "Seven apples." in records[0]["question"]
    assert "How many apples?" in records[0]["question"]
    assert records[1]["gold"] == "Yes"  # 非数值答案保留原文


def test_prepare_and_load_svamp(monkeypatch, tmp_path):
    _use_prepared_root(monkeypatch, tmp_path)
    svamp = tmp_path / "SVAMP.json"
    svamp.write_text(
        json.dumps(
            [
                {
                    "ID": "chal-1",
                    "Body": "Each pack of dvds costs 76 dollars.",
                    "Question": "How much do you pay?",
                    "Answer": 51.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LYCHEE_SVAMP_JSON", str(svamp))

    mlc_math.prepare_svamp()
    records = benchmarks.load("svamp")

    assert records[0]["gold"] == "51"
    assert records[0]["question"] == "Each pack of dvds costs 76 dollars. How much do you pay?"


def test_prepare_and_load_math(monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _use_prepared_root(monkeypatch, tmp_path)
    root = tmp_path / "Math"
    for subject in mlc_math.MATH_SUBJECTS:
        subject_dir = root / subject
        subject_dir.mkdir(parents=True)
        for split in ("test", "train"):
            frame = pd.DataFrame(
                [
                    {
                        "problem": f"{subject} {split} problem",
                        "level": "Level 1",
                        "type": subject,
                        "solution": r"work ... so the answer is $\boxed{\frac{14}{3}}$.",
                    }
                ]
            )
            frame.to_parquet(subject_dir / f"{split}-00000-of-00001.parquet")
    monkeypatch.setenv("LYCHEE_MATH_DIR", str(root))

    mlc_math.prepare_math()
    records = benchmarks.load("math")

    assert len(records) == len(mlc_math.MATH_SUBJECTS)  # 7 个学科各 1 题 test
    assert all(r["kind"] == "aime" for r in records)
    assert records[0]["gold"] == r"\frac{14}{3}"
    # train 分片另存一份，供需要训练素材的方法取用
    assert len(mlc_math.load_math_train()) == len(mlc_math.MATH_SUBJECTS)


# ====================== 显式报错路径 ======================
def test_load_before_prepare_raises(monkeypatch, tmp_path):
    _use_prepared_root(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="尚未 prepare"):
        benchmarks.load("svamp")


def test_missing_source_raises_with_hint(monkeypatch, tmp_path):
    _use_prepared_root(monkeypatch, tmp_path)
    monkeypatch.delenv("LYCHEE_MULTIARITH_JSONL", raising=False)
    monkeypatch.setenv("LYCHEE_MLC_DATA_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="LYCHEE_MULTIARITH_JSONL"):
        mlc_math.prepare_multiarith()


def test_unparsable_asdiv_raises(monkeypatch, tmp_path):
    _use_prepared_root(monkeypatch, tmp_path)
    xml = tmp_path / "empty.xml"
    xml.write_text("<Machine-Reading-Corpus-File></Machine-Reading-Corpus-File>", encoding="utf-8")
    monkeypatch.setenv("LYCHEE_ASDIV_XML", str(xml))
    with pytest.raises(ValueError, match="未解析出任何"):
        mlc_math.prepare_asdiv(force=True)


# ====================== 注册与任务配置 ======================
def test_mlc_tasks_registered():
    from lychee_mas import REGISTRY

    registered = set(REGISTRY.list("benchmark"))
    for task in benchmarks.MLC_MATH_TASKS:
        assert task in registered
        assert task in benchmarks.LOADERS
        assert task in benchmarks.PREPARERS
        assert task in TASK_CONFIG
    assert set(benchmarks.MLC_MATH_TASKS) == {
        "addsub",
        "singleeq",
        "multiarith",
        "asdiv",
        "svamp",
        "math",
    }
    # 结构表里每个新 benchmark_source 都要有 mappings（否则 import 时就会 KeyError）
    sources = {row["benchmark_source"] for row in benchmarks.BENCHMARK_STRUCTURE}
    expected_sources = (
        "AddSub (MAWPS)",
        "SingleEQ (MAWPS)",
        "MultiArith (MAWPS)",
        "ASDiv",
        "SVAMP",
        "MATH",
    )
    for source in expected_sources:
        assert source in sources


def test_mlc_task_config_uses_expected_team_and_extractor():
    assert team_name_for_task("math") == "aime"
    assert extractor_for_task("math").__name__ == "_extract_boxed"
    for task in ("addsub", "singleeq", "multiarith", "asdiv", "svamp"):
        assert team_name_for_task(task) == "reason"
        assert extractor_for_task(task).__name__ == "_extract_default"


def test_kind_mapping_matches_gold_shape():
    assert mlc_math._KINDS["math"] == "aime"  # LaTeX 答案需要符号等价比较
    for task in ("addsub", "singleeq", "multiarith", "asdiv", "svamp"):
        assert mlc_math._KINDS[task] == "exact"


# ====================== GSM8K 本地数据入口 ======================
def test_gsm8k_local_jsonl_hook(monkeypatch, tmp_path):
    """本地 train/test.jsonl 应能直接进 prepared 缓存，不需要联网下载。"""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    _use_prepared_root(monkeypatch, tmp_path)
    data_dir = tmp_path / "gsm8k_local"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        json.dumps({"question": "train q", "answer": "reasoning\n#### 1"}) + "\n",
        encoding="utf-8",
    )
    (data_dir / "test.jsonl").write_text(
        json.dumps({"question": "test q", "answer": "reasoning\n#### 2"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LYCHEE_GSM8K_DATA_DIR", str(data_dir))
    monkeypatch.delenv("LYCHEE_GSM8K_TRAIN_JSONL", raising=False)
    monkeypatch.delenv("LYCHEE_GSM8K_TEST_JSONL", raising=False)

    assert benchmarks.prepare("gsm8k") == str(tmp_path / "prepared" / "gsm8k" / "main")
    records = benchmarks.load("gsm8k")
    assert records[0]["gold"] == "2"
    assert records[0]["kind"] == "exact"


def test_gsm8k_local_dir_missing_split_raises(monkeypatch, tmp_path):
    _use_prepared_root(monkeypatch, tmp_path)
    data_dir = tmp_path / "gsm8k_partial"
    data_dir.mkdir()
    (data_dir / "test.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("LYCHEE_GSM8K_DATA_DIR", str(data_dir))
    monkeypatch.delenv("LYCHEE_GSM8K_TRAIN_JSONL", raising=False)
    monkeypatch.delenv("LYCHEE_GSM8K_TEST_JSONL", raising=False)

    with pytest.raises(FileNotFoundError, match="train.jsonl"):
        benchmarks.prepare("gsm8k")
