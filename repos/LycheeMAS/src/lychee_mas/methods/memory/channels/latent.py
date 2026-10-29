"""LatentMemory —— 从源上下文产出一个 (1, P, hidden) 的 prefix（latent 通道）。

探针阶段对 latent 通道的操作化 = **免训练的自压缩**（既定方案）：把源文本喂进 Qwen3-4B，
取末层 hidden states (1, T, H)，压成 P 个位置，作为 soft prefix 注入。

P 是 latent 通道的成本旋钮：P 小 = 便宜但有损。扫 P 就能画出 latent 的 cost-quality 曲线，与 NL 对
照。

压缩策略（都免训练、确定性）：
  - "segment_mean": 把 T 个 token 切成 P 段连续区间，各段均值池化 -> (P,H)。保序、平滑，好默认。
  - "stride":       均匀取 P 个单点位置。保留尖锐的 token 状态，丢掉其余（更有损，作对照）。
  - "tail":         只保留最后 P 个 hidden 状态（recency）。
  - "soft_token":   把 hidden 投回「输入嵌入空间」再池化——这是当前正确做法（默认）。

⚠️ 关键坑（记忆 latent-prefix-must-be-input-embedding-space）：直接把末层 hidden 当 prefix 注入是
「表示空间不匹配」，表现像噪声（P 越大越差）。soft_token 修了这一点。

⚠️ 惰性导入：torch 只在函数/方法内部 import，保证本模块在无 torch 的环境也可 import（黄金法则 4）。

TODO（真实组件）：探针出现交叉后，换成可学习压缩器（gist tokens / cross-attn pooler）。接口
(text -> (1,P,H)) 保持不变。
"""
from __future__ import annotations

from typing import Any


def compress_hidden(hidden: Any, P: int, strategy: str = "segment_mean") -> Any:
    """hidden: (1, T, H) 末层状态 -> prefix (1, P, H)。naive 对照路径（非 soft_token）。"""
    import torch  # 惰性导入

    assert hidden.dim() == 3 and hidden.shape[0] == 1
    T = hidden.shape[1]
    P = max(1, min(P, T))  # P 夹到 [1, T]，避免越界
    if strategy == "tail":
        return hidden[:, T - P:, :].contiguous()  # 取末尾 P 个
    if strategy == "stride":
        idx = torch.linspace(0, T - 1, steps=P, device=hidden.device).round().long()
        return hidden[:, idx, :].contiguous()  # 均匀取 P 个单点
    if strategy == "segment_mean":
        return _segment_mean(hidden, P)  # 分 P 段均值池化
    raise ValueError(f"unknown strategy {strategy}")


def _segment_mean(x: Any, P: int) -> Any:
    """(1,T,D) -> (1,P,D)：切成 P 个近似等长的连续段，各段在序列维上取均值。"""
    import torch  # 惰性导入

    T = x.shape[1]
    P = max(1, min(P, T))
    bounds = torch.linspace(0, T, steps=P + 1, device=x.device).round().long()  # P+1 个分界点
    segs = []
    for i in range(P):
        a, b = int(bounds[i]), int(max(bounds[i] + 1, bounds[i + 1]))  # 保证每段至少 1 个位置
        segs.append(x[:, a:b, :].mean(dim=1, keepdim=True))
    return torch.cat(segs, dim=1).contiguous()


def soft_token_prefix(hidden: Any, embed_weight: Any, P: int,
                      tau: float = 1.0, chunk: int = 512) -> Any:
    """把末层 hidden 投回「输入嵌入空间」，再池化到 P。这是 latent 的正确构造方式。

    为什么：直接拿末层 hidden 当 inputs_embeds 的 prefix 是空间不匹配——hidden 活在输出/预测空间，
    不是 token 嵌入空间，注进去像噪声。Qwen 是 tied embeddings（unembed = embed^T），所以
        soft_embed_t = softmax(h_t @ E^T / tau) @ E
    就是 t 位置上模型「期望的 token 嵌入」——一个忠实、免训练、确实活在输入嵌入空间的 latent 表示。
    最后再 segment-mean 到 P 个位置。tau 越小越接近 argmax（更尖锐）；chunk 是为省显存的分块大小。
    """
    import torch  # 惰性导入

    h = hidden[0]  # (T, H) 去掉 batch 维
    E = embed_weight  # (V, H) 输入嵌入矩阵（与 unembed 共享）
    soft = []
    for s in range(0, h.shape[0], chunk):  # 分块算，避免 (T,V) 巨矩阵爆显存
        logits = (h[s:s + chunk] @ E.t()) / tau  # (c, V) 每位置对全词表的相似度
        probs = torch.softmax(logits.float(), dim=-1).to(E.dtype)  # (c, V) 软分布
        soft.append(probs @ E)  # (c, H) 软加权得到期望嵌入
    soft_embeds = torch.cat(soft, dim=0).unsqueeze(0)  # (1, T, H)
    return _segment_mean(soft_embeds, P)  # 再压到 P 个位置


class LatentMemory:
    """latent 通道的**统一接口**：把源上下文物化成注入所需的 latent 表示。

    两种策略（`strategy`）：
    - `soft_token`（默认，免训练）/ `segment_mean` / `stride` / `tail`：把源文本压成 (1,P,H) soft
      prefix（`build_prefix`）。
    - `c2c`：训练好的 Cache-to-Cache 融合器——**内部调** `c2c_channel`（懒加载 projector 栈），
      不物化 prefix；source 的组装/对齐由注入 client 用 ctx 完成。

    「其它方法」（`c2c_channel`/`c2c_projector`）都在本类内部按需调用；manager 只依赖 nl/latent
    两个主接口，**不直接 import** c2c。
    """

    # 走「KV-cache 融合」注入的 latent 策略（Latent_Channel=projector 栈，而非 (1,P,H) soft
    # prefix）。后续新增此类方法在此登记即可，无需改 materialize 的分支逻辑。
    FUSION_STRATEGIES = frozenset({"c2c"})

    def __init__(self, backend, strategy: str = "segment_mean", max_encode_tokens: int = 4096,
                 tau: float = 1.0, P: int = 16, c2c_ckpt: str | None = None,
                 c2c_gate: str = "soft"):
        self.backend = backend  # HF 后端（提供 encode_hidden 与 embed）
        self.strategy = strategy  # 压缩策略；DualChannel 默认传 "soft_token"，或 "c2c"
        self.max_encode_tokens = max_encode_tokens  # 编码时源文本最多取多少 token（防过长）
        self.tau = tau  # soft_token 的温度
        self.P = P  # prefix 长度（latent 成本旋钮）；原在 RouteDecision，现归 latent 通道自身
        self.c2c_ckpt = c2c_ckpt  # strategy=c2c 时的 projector ckpt 路径
        self.c2c_gate = c2c_gate  # c2c 门控 soft/hard
        self._c2c: Any = None  # 懒建的 C2CLatentChannel（内部调用）
        self._prefix_cache: dict = {}  # (len(source), P) -> soft prefix 缓存

    def reset(self) -> None:
        # source 变了 / 换样本时清 prefix 缓存（manager 在 reset/seed/observe 里调）
        self._prefix_cache = {}

    def materialize(self, bundle, source: str) -> None:
        """把 latent 通道物化进 bundle.Latent_Channel（+ Latent_strategy=本方法名）：
        融合类策略 ⇒ projector 栈；prefix 类 ⇒ (1,P,H) soft prefix（P=self.P，按 source 缓存）。"""
        bundle.Latent_strategy = self.strategy  # 注入 client 据此决定注入方式
        if self.strategy in self.FUSION_STRATEGIES:
            bundle.Latent_Channel = self._c2c_channel().get_projectors()
            bundle.meta["latent_kind"] = self.strategy
            return
        if source:
            key = (len(source), self.P)  # 同 source 同 P 复用缓存，省重复编码
            if key not in self._prefix_cache:
                self._prefix_cache[key] = self.build_prefix(source, self.P)
            bundle.Latent_Channel = self._prefix_cache[key]

    def _c2c_channel(self):
        # 懒加载 C2C 融合器；**在 latent 内部 import**，使 manager 无需 import c2c_channel
        if self._c2c is None:
            from .c2c_channel import C2CLatentChannel
            self._c2c = C2CLatentChannel(self.backend, self.c2c_ckpt, gate=self.c2c_gate)
        return self._c2c

    def build_prefix(self, source_text: str, P: int) -> Any:
        # 先把源文本编码成末层 hidden (1,T,H)
        hidden = self.backend.encode_hidden(source_text, max_tokens=self.max_encode_tokens)
        if self.strategy == "soft_token":
            # 正确路径：投回输入嵌入空间（需要嵌入矩阵 E = backend.embed.weight）
            return soft_token_prefix(hidden, self.backend.embed.weight, P, self.tau)
        # 对照路径：直接在末层 hidden 上压缩（naive，注入后偏噪声）
        return compress_hidden(hidden, P, self.strategy)
