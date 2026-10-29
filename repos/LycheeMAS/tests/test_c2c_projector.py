"""C2CProjector 单元测试（latent 通道的 C2C 融合器）。

torch 不可用时整文件跳过，保证离线 [dev] 环境 `make test` 仍全绿（黄金法则 5）。
"""
import pytest

torch = pytest.importorskip("torch")

from lychee_mas.methods.memory.channels.c2c_projector import (  # noqa: E402
    C2CProjector,
    build_projector_stack,
)


def _kv(B, H, N, D):
    return torch.randn(B, H, N, D), torch.randn(B, H, N, D)


def test_forward_shapes_same_model():
    # 同模型场景：source/target 同维同头（Qwen3-4B：head_dim=128, kv_heads=8）
    B, H, N, D = 2, 8, 5, 128
    proj = C2CProjector(source_dim=D, target_dim=D, source_num_heads=H, target_num_heads=H,
                        hidden_dim=64, intermediate_dim=64, num_layers=3).eval()
    src, tgt = _kv(B, H, N, D), _kv(B, H, N, D)
    ok, ov = proj(src, tgt)
    assert ok.shape == (B, H, N, D) and ov.shape == (B, H, N, D)


def test_untrained_gate_is_identity():
    # gate_logit=0 + 推理模式 ⇒ gate=(logit>0)=0 ⇒ 输出==target（安全恒等初值）
    B, H, N, D = 1, 8, 4, 128
    proj = C2CProjector(D, D, H, H, hidden_dim=64, intermediate_dim=64, num_layers=3).eval()
    src, tgt = _kv(B, H, N, D), _kv(B, H, N, D)
    ok, ov = proj(src, tgt)
    assert torch.allclose(ok, tgt[0]) and torch.allclose(ov, tgt[1])


def test_open_gate_changes_output():
    # 打开门控后输出应偏离 target（投影路真的起作用）
    B, H, N, D = 1, 8, 4, 128
    proj = C2CProjector(D, D, H, H, hidden_dim=64, intermediate_dim=64, num_layers=3).eval()
    with torch.no_grad():
        proj.key_gate_logit.fill_(5.0)
        proj.value_gate_logit.fill_(5.0)
    src, tgt = _kv(B, H, N, D), _kv(B, H, N, D)
    ok, _ = proj(src, tgt)
    assert not torch.allclose(ok, tgt[0])


def test_cross_dim_projection_shape():
    # 跨模型场景：source/target 维度/头数不同，输出仍落在 target 空间
    B, N = 1, 6
    proj = C2CProjector(source_dim=64, target_dim=128, source_num_heads=4, target_num_heads=8,
                        hidden_dim=64, intermediate_dim=64, num_layers=3).eval()
    src = (torch.randn(B, 4, N, 64), torch.randn(B, 4, N, 64))
    tgt = (torch.randn(B, 8, N, 128), torch.randn(B, 8, N, 128))
    ok, ov = proj(src, tgt)
    assert ok.shape == (B, 8, N, 128) and ov.shape == (B, 8, N, 128)


def test_temperature_anneal_monotonic():
    proj = C2CProjector(128, 128, 8, 8, hidden_dim=64, intermediate_dim=64, num_layers=3,
                        initial_temperature=1.0, final_temperature=0.001, anneal_steps=100)
    proj.update_temperature(0)
    t0 = float(proj.gate_temperature)
    proj.update_temperature(50)
    t1 = float(proj.gate_temperature)
    proj.update_temperature(1000)
    t2 = float(proj.gate_temperature)
    assert t0 > t1 > t2 and abs(t2 - 0.001) < 1e-6


def test_build_projector_stack():
    stack = build_projector_stack(num_hidden_layers=4, head_dim=128, num_kv_heads=8,
                                  hidden_dim=64, intermediate_dim=64)
    assert len(stack) == 4 and all(isinstance(p, C2CProjector) for p in stack)
