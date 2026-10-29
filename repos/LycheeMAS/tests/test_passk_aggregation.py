"""pass@K 聚合（metrics.aggregate_samples）离线测试 —— 零重依赖。

run_mas.py --samples K 会给每个 case 落 K 份 prediction（同 case_id、不同 k_index）；
打分侧 analyze_benchmark_run.py 逐份打分后交给 aggregate_samples，按 case 分组补充：
  pass@1 = 各 case 内 K 份得分均值，再对 case 求平均
  pass@K = 各 case best-of-K 得分，再对 case 求平均（二值打分即「任一正确率」）
仅当检测到「样本数 > 去重 case 数」时才写入 pass_at_k，K=1 运行完全不变。
"""
from lychee_mas.eval.metrics import aggregate_samples


def _s(case_id, score, k_index, num_samples, kind="exact"):
    return {
        "case_id": case_id, "score": score,
        "is_correct": score == 1.0, "scorer_kind": kind,
        "k_index": k_index, "num_samples": num_samples,
    }


def test_passk_binary_multi_sample():
    # case A: [1,0,0] -> mean=1/3, best=1 ; case B: [0,0,0] -> mean=0, best=0
    samples = [
        _s("A", 1.0, 0, 3), _s("A", 0.0, 1, 3), _s("A", 0.0, 2, 3),
        _s("B", 0.0, 0, 3), _s("B", 0.0, 1, 3), _s("B", 0.0, 2, 3),
    ]
    m = aggregate_samples(samples, run_info={"scorer_kind": "exact", "task": "aime_2024"})
    pk = m["pass_at_k"]
    assert pk["num_distinct_cases"] == 2
    assert pk["max_samples_per_case"] == 3
    assert pk["samples_per_case"] == 3.0
    assert pk["pass@1"] == round((1 / 3 + 0) / 2, 4)   # 0.1667
    assert pk["pass@3"] == 0.5                          # (1 + 0) / 2
    # pass_at_k 字段同时展开到顶层，便于旧读取器直接取
    assert m["pass@1"] == pk["pass@1"] and m["pass@3"] == pk["pass@3"]


def test_passk_absent_for_single_sample():
    # 每个 case 恰好 1 份 -> 不写 pass_at_k，其余指标与旧格式一致
    single = [
        {"case_id": "A", "score": 1.0, "is_correct": True, "scorer_kind": "exact"},
        {"case_id": "B", "score": 0.0, "is_correct": False, "scorer_kind": "exact"},
    ]
    m = aggregate_samples(single, run_info={"scorer_kind": "exact"})
    assert "pass_at_k" not in m
    assert "pass@1" not in m
    assert m["score_mean"] == 0.5
    assert m["num_cases"] == 2


def test_passk_continuous_scorer_best_of_k():
    # f1 等连续打分：pass@K 取 best-of-K 均值，pass@1 取均值的均值
    samples = [
        _s("A", 0.4, 0, 2, kind="f1"), _s("A", 0.9, 1, 2, kind="f1"),
        _s("B", 0.2, 0, 2, kind="f1"), _s("B", 0.0, 1, 2, kind="f1"),
    ]
    m = aggregate_samples(samples, run_info={"scorer_kind": "f1"})
    pk = m["pass_at_k"]
    assert pk["pass@1"] == round(((0.4 + 0.9) / 2 + (0.2 + 0.0) / 2) / 2, 4)  # 0.375
    assert pk["pass@2"] == round((0.9 + 0.2) / 2, 4)                          # 0.55


def test_passk_uneven_samples_per_case():
    # 断点续跑可能导致每个 case 采样数不齐：按各自 K 归一，max_samples 取最大
    samples = [
        _s("A", 1.0, 0, 3), _s("A", 0.0, 1, 3),          # A 只有 2 份
        _s("B", 0.0, 0, 3), _s("B", 0.0, 1, 3), _s("B", 1.0, 2, 3),
    ]
    m = aggregate_samples(samples, run_info={"scorer_kind": "exact"})
    pk = m["pass_at_k"]
    assert pk["num_distinct_cases"] == 2
    assert pk["max_samples_per_case"] == 3
    assert pk["pass@1"] == round((0.5 + 1 / 3) / 2, 4)   # (A:1/2, B:1/3) 均值
    assert pk["pass@3"] == 1.0                            # 两个 case 都有命中
