"""C2C 投影器 —— latent 通道的「可学习压缩/融合器」（移植自 Cache-to-Cache, arXiv 2510.03215）。

背景：免训练 soft_token 前缀在真实长上下文上会塌成 OOD 噪声（见
[[latent-prefix-must-be-input-embedding-space]] / probe）。C2C 改走**逐层 KV-cache 融合**：
对每个注意力层，用一个 projector 把 source 的 (K,V) 投影并融合进 target 的 (K,V) 里，替换 target
注意力用的 K,V。只训 projector，两个 LM 全冻结，普通 CLM 损失。

本文件只实现**单层 projector 模块**（C2CProjector）+ 它依赖的 FFN 子块；
「在 target 前向里逐层注入融合 KV」的 backend 接线与训练在后续阶段实现（见 DOCUMENTS 路线图）。

同模型场景（CDM：单 Qwen3-4B 跨 agent）：source 与 target 同架构 ⇒
source_dim==target_dim、source_num_heads==target_num_heads（Qwen3-4B：head_dim=128, KV heads=8）。

⚠️ 本模块在 import 时即触发 torch（nn.Module 定义需要），故**只能被惰性 import**
（在 backend/训练脚本的方法内部 import），不得进入包 __init__ 链，否则破坏 `make selfcheck`。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class StandardFFNLayer(nn.Module):
    """Pre-RMSNorm + 经典 MLP + 残差：y = x + Dropout(W2(act(W1(RMSNorm(x)))))（C2C 原样）。"""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0,
                 dtype: torch.dtype = torch.float32, activation: str = "gelu"):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[activation.lower()]

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.act(self.w1(h))
        h = self.drop(self.w2(h))
        return x + h


class RegularMLP(nn.Module):
    """固定 hidden 尺寸上堆 num_layers 个 StandardFFNLayer（无输入/输出投影，由调用方负责）。"""

    def __init__(self, hidden_dim: int = 1024, intermediate_dim: int = 3072,
                 num_layers: int = 3, dropout: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.blocks = nn.ModuleList([
            StandardFFNLayer(hidden_size=hidden_dim, intermediate_size=intermediate_dim,
                             dropout=dropout, dtype=dtype)
            for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class C2CProjector(nn.Module):
    """单层 KV-cache 融合器（移植自 rosetta.model.projector.C2CProjector）。

    输入：source_kv=(K,V) 形状 (B,Hs,N,Ds)；target_kv=(K,V) 形状 (B,Ht,N,Dt)。
    输出：融合后的 (K,V)，形状 (B,Ht,N,Dt)，作为 target 注意力的新 K,V。

    结构（key/value 各一套，对称）：
      concat(source,target) -Linear-> hidden -RegularMLP(1)-> 中间表示，再分两路：
        · scalar 路：MLP+Linear -> 每头标量权重 (B,Ht,N,1)，sigmoid 归一
        · proj   路：MLP+Linear -> 投影到 target 空间 (B,Ht,N,Dt)
      门控 gate：标量 logit，训练用 Gumbel-sigmoid（温度退火），推理用硬阈值 (logit>0)
      融合（add_self=True）：out = target + gate * sigmoid(scalar) * projected
    未训练时 gate_logit=0 ⇒ 推理期 gate=0 ⇒ out==target（恒等，安全初值）。
    """

    def __init__(self, source_dim: int, target_dim: int,
                 source_num_heads: int = 1, target_num_heads: int = 1,
                 intermediate_dim: int = 1024, hidden_dim: int = 1024, num_layers: int = 3,
                 dropout: float = 0.1, initial_temperature: float = 1.0,
                 final_temperature: float = 0.001, anneal_steps: int = 1929,
                 dtype: torch.dtype = torch.float32, zero_init: bool = False):
        super().__init__()
        assert num_layers >= 3, "num_layers must be >= 3"
        self.source_dim, self.target_dim = source_dim, target_dim
        self.source_num_heads, self.target_num_heads = source_num_heads, target_num_heads
        in_dim = source_dim * source_num_heads
        out_dim = target_dim * target_num_heads

        # 1) concat(source,target) -> hidden
        self.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
        self.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
        # 2) 公共一层嵌入 MLP
        self.key_mlp1 = RegularMLP(hidden_dim, intermediate_dim, 1, dropout, dtype)
        self.value_mlp1 = RegularMLP(hidden_dim, intermediate_dim, 1, dropout, dtype)
        # 3a) scalar 权重路（每头一个标量）
        self.key_scalar_mlp2 = RegularMLP(hidden_dim, hidden_dim, 1, dropout, dtype)
        self.value_scalar_mlp2 = RegularMLP(hidden_dim, hidden_dim, 1, dropout, dtype)
        self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
        self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
        # 3b) 投影路（投到 target 空间）
        npl = num_layers - 2
        self.key_proj_mlp2 = RegularMLP(hidden_dim, intermediate_dim, npl, dropout, dtype)
        self.value_proj_mlp2 = RegularMLP(hidden_dim, intermediate_dim, npl, dropout, dtype)
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        if zero_init:  # 投影输出置零 ⇒ 初始恒等
            for lin in (self.key_proj_out, self.value_proj_out):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        # 门控 + 温度退火
        self.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.use_gumbel = True
        self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps
        self.scalar_temperature = 1.0
        # 推理门控：True=硬阈值 (logit>0)（C2C 默认，假设退火把 logit 推到 ±大）；
        # False=软门 sigmoid(logit)（当退火未把 logit 推开时更忠实地施加所学融合）。
        self.hard_gate = True

    def update_temperature(self, step: int) -> None:
        """指数退火门控温度（1.0 -> final，anneal_steps 步内）；训练每 optimizer step 调一次。"""
        ratio = min(step / self.anneal_steps, 1.0)
        rel = (self.final_temperature / self.initial_temperature) ** ratio
        self.gate_temperature.fill_(self.initial_temperature * rel)

    def _one_side(self, source_flat: Tensor, target_flat: Tensor, target_kv: Tensor,
                  lin_in, mlp1, scalar_mlp2, scalar_head, proj_mlp2, proj_out,
                  gate_logit) -> Tensor:
        """key/value 共用的单侧融合逻辑。target_kv: (B,Ht,N,Dt)。"""
        B, Ht, N, Dt = target_kv.shape
        h = mlp1(lin_in(torch.cat([source_flat, target_flat], dim=-1)))  # (B,N,hidden)
        # 投影路 -> (B,Ht,N,Dt)
        projected = proj_out(proj_mlp2(h)).view(B, N, Ht, Dt).transpose(1, 2)
        # scalar 路 -> (B,Ht,N,1)
        scalar = scalar_head(scalar_mlp2(h)).permute(0, 2, 1).unsqueeze(-1)
        # 门控：训练 Gumbel-sigmoid，推理硬阈值
        gl = gate_logit.view(1, 1, 1, 1)
        if self.training and self.use_gumbel:
            u = torch.rand(B, Ht, N, 1, device=gl.device, dtype=gl.dtype)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            gate = torch.sigmoid((gl + g) / self.gate_temperature)
        else:
            gate = (gl > 0).to(gl.dtype) if self.hard_gate else torch.sigmoid(gl)
        return target_kv + gate * torch.sigmoid(scalar) * projected  # add_self

    def forward(self, source_kv: Tuple[Tensor, Tensor], target_kv: Tuple[Tensor, Tensor],
                position_ids: Optional[Tensor] = None,
                max_pos: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        source_key, source_value = source_kv
        target_key, target_value = target_kv
        B, Hs, N, Ds = source_key.shape
        _, Ht, _, Dt = target_key.shape
        # 展平多头：(B,H,N,D) -> (B,N,H*D)
        sk = source_key.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        sv = source_value.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        tk = target_key.transpose(1, 2).contiguous().view(B, N, Ht * Dt)
        tv = target_value.transpose(1, 2).contiguous().view(B, N, Ht * Dt)
        out_key = self._one_side(sk, tk, target_key, self.key_in, self.key_mlp1,
                                 self.key_scalar_mlp2, self.key_scalar_head,
                                 self.key_proj_mlp2, self.key_proj_out, self.key_gate_logit)
        out_value = self._one_side(sv, tv, target_value, self.value_in, self.value_mlp1,
                                   self.value_scalar_mlp2, self.value_scalar_head,
                                   self.value_proj_mlp2, self.value_proj_out, self.value_gate_logit)
        return out_key, out_value


def build_projector_stack(num_hidden_layers: int, head_dim: int, num_kv_heads: int,
                          hidden_dim: int = 1024, intermediate_dim: int = 1024,
                          num_layers: int = 3, dtype: torch.dtype = torch.float32,
                          zero_init: bool = True, src_num_kv_heads=None,
                          src_head_dim=None) -> "nn.ModuleList":
    """每个 target 层建一个 C2CProjector。zero_init 让初始为恒等。

    同模型：src_* 留空 = 与 target 同维同头。跨模型：传 source 的 kv 头数/head_dim
    （C2C 原生支持 source≠target 维度；num_hidden_layers 始终是 *target* 层数）。
    """
    sh = src_head_dim or head_dim
    sk = src_num_kv_heads or num_kv_heads
    return nn.ModuleList([
        C2CProjector(source_dim=sh, target_dim=head_dim,
                     source_num_heads=sk, target_num_heads=num_kv_heads,
                     hidden_dim=hidden_dim, intermediate_dim=intermediate_dim,
                     num_layers=num_layers, dtype=dtype, zero_init=zero_init)
        for _ in range(num_hidden_layers)])


def map_source_to_target_layers(source_layers, n_target: int):
    """跨模型层映射：source 与 target 层数可能不同（如 28 vs 36）。
    返回长度 n_target 的列表，第 i 项 = source 第 round(i*(n_src-1)/(n_target-1)) 层的 (K,V)。
    层数相同则原样返回（恒等映射）。
    """
    n_src = len(source_layers)
    if n_src == n_target:
        return list(source_layers)
    denom = max(1, n_target - 1)
    return [source_layers[round(i * (n_src - 1) / denom)] for i in range(n_target)]
