"""Focused tests for MLC runner diagnostics and answer parsing."""

import pytest

torch = pytest.importorskip("torch")

from scripts.run_mlc_gsm8k import (  # noqa: E402
    _answer,
    _output_distance,
    _paired_accuracy_test,
)


def test_answer_parser_treats_empty_approve_as_empty_answer() -> None:
    assert _answer("APPROVE:") == ""
    assert _answer("APPROVE: 42") == "42"


def test_output_distance_is_zero_for_identical_and_positive_for_distinct_text() -> None:
    assert _output_distance(["2 + 2 = 4", "2 + 2 = 4"]) == 0.0
    distance = _output_distance(["2 + 2 = 4", "3 * 5 = 15"])
    assert 0.0 < distance <= 1.0


def test_paired_accuracy_counts_discordant_cases_by_method() -> None:
    mlc = [
        {"case_id": "1", "score": 1, "prediction": "3", "gold": "3"},
        {"case_id": "2", "score": 0, "prediction": "4", "gold": "5"},
    ]
    baseline = [
        {"case_id": "1", "score": 0, "prediction": "2", "gold": "3"},
        {"case_id": "2", "score": 1, "prediction": "5", "gold": "5"},
    ]
    result = _paired_accuracy_test(mlc, baseline, strict=False)
    assert result["paired_n"] == 2
    assert result["mlc_only_correct"] == 1
    assert result["baseline_only_correct"] == 1
    assert result["mcnemar_exact_two_sided_p"] == 1.0
