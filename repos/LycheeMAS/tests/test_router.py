"""路由：static.decide + _enforce_availability（latent 不可用/异构对回退 nl/none）。"""
from __future__ import annotations

from lychee_mas.methods.memory.routing.base import RouterInputs
from lychee_mas.methods.memory.routing.static import StaticRouter, fixed_channel_router


def _inputs(**kw):
    base = dict(role="worker", task="gsm8k", turn=0)
    base.update(kw)
    return RouterInputs(**base)


def test_static_table_lookup():
    r = StaticRouter(table={("worker", "*"): "both"}, default="nl")
    d = r.decide(_inputs(role="worker"))
    assert d.channel == "both"


def test_static_default_when_unmatched():
    r = StaticRouter(table={("manager", "*"): "nl"}, default="none")
    d = r.decide(_inputs(role="verifier"))
    assert d.channel == "none"


def test_latent_falls_back_to_nl_when_unavailable():
    # latent 通道被标为不可用 -> latent 应降级为 nl
    r = fixed_channel_router("latent")
    d = r.decide(_inputs(availability={"latent": False}, same_model_pair=True))
    assert d.channel == "nl"
    assert "latent->nl" in d.reason


def test_latent_falls_back_on_heterogeneous_pair():
    # 收发为异构（非同模型）对 -> latent 应降级为 nl
    r = fixed_channel_router("latent")
    d = r.decide(_inputs(availability={"latent": True}, same_model_pair=False))
    assert d.channel == "nl"


def test_both_drops_latent_keeps_nl_when_unavailable():
    # both 在 latent 不可用时应保留 nl 分量（降级为 nl）
    r = fixed_channel_router("both")
    d = r.decide(_inputs(availability={"latent": False}))
    assert d.channel == "nl"


def test_latent_kept_when_available_and_same_model():
    r = fixed_channel_router("latent")
    d = r.decide(_inputs(availability={"latent": True}, same_model_pair=True))
    assert d.channel == "latent"
