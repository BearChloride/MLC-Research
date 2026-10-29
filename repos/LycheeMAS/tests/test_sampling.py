"""固定种子评测子集抽样的离线测试。"""

import pytest
from lychee_mas.eval.sampling import sample_records


def test_same_seed_is_deterministic_and_indices_sorted():
    pool = list(range(500))
    first, meta_a = sample_records(pool, n=50, seed=0)
    second, meta_b = sample_records(pool, n=50, seed=0)

    assert first == second
    assert meta_a == meta_b
    assert meta_a["pool_size"] == 500
    assert meta_a["sampled"] == 50
    indices = meta_a["sampled_indices"]
    assert indices == sorted(indices)          # 顺序稳定
    assert len(set(indices)) == 50             # 无重复
    assert all(0 <= i < 500 for i in indices)
    assert first == [pool[i] for i in indices]


def test_different_seed_gives_different_subset():
    pool = list(range(1000))
    a, _ = sample_records(pool, n=50, seed=0)
    b, _ = sample_records(pool, n=50, seed=1)
    assert a != b


def test_n_not_smaller_than_pool_returns_full_pool():
    pool = list(range(10))
    out, meta = sample_records(pool, n=100, seed=0)
    assert out == pool
    assert meta["sampled"] == 10
    assert meta["sampled_indices"] is None


def test_none_returns_full_pool():
    pool = list(range(10))
    out, meta = sample_records(pool, n=None, seed=0)
    assert out == pool
    assert meta["pool_size"] == 10
    assert meta["sampled_indices"] is None


def test_empty_pool_raises():
    with pytest.raises(ValueError, match="输入为空"):
        sample_records([], n=5, seed=0)
