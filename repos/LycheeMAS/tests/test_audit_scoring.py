"""审计口径（strict numeric）的离线测试。"""

from lychee_mas.eval.audit import strict_correct, strict_numeric_score


def test_strict_matches_when_numbers_equal():
    assert strict_numeric_score(r"$\boxed{64}$", "64") == 1.0
    assert strict_numeric_score(r"$\boxed{64}$", "64.0") == 1.0  # 数值相等，非字符串相等
    assert strict_numeric_score("答案是 1,000", "1000") == 1.0


def test_strict_catches_substring_false_positive():
    # 官方 score_exact 会因为 "64" 是 "10648" 的子串而判对；严格口径必须判错
    assert strict_numeric_score(r"$\boxed{10.648}$", "64") == 0.0
    assert strict_numeric_score(r"\boxed{3.25}**", "5") == 0.0


def test_strict_returns_none_when_gold_has_no_number():
    assert strict_numeric_score("Yes", "Yes") is None
    assert strict_numeric_score("Purple", "Purple") is None


def test_strict_uses_last_number_for_time_like_text():
    # "3:30 p.m." 里最后一个数字是 30：严格口径按"最后一个数字"取值，这是 metrics 既有的抽取约定
    assert strict_numeric_score("3:30 p.m.", "3:30 p.m.") == 1.0


def test_strict_zero_when_prediction_has_no_number():
    assert strict_numeric_score("I cannot solve it", "64") == 0.0


def test_strict_correct_boolean_form():
    assert strict_correct(r"$\boxed{7}$", "7") is True
    assert strict_correct(r"$\boxed{9}$", "7") is False
    assert strict_correct("Yes", "Yes") is None
