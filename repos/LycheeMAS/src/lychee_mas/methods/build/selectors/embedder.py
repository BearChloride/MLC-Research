"""角色/query 文本嵌入器（generate 模式用；移植自官方 `AgentInit/agentinit/embedder.py`）。

HF `AutoModel` + mean-pooling + L2 归一化。模型路径走 env `LYCHEE_EMBED_MODEL`（一个句向量编码器，
与 CDM 生成模型 `LYCHEE_HF_MODEL` 分开），绝不硬编码（黄金法则 §7）。torch/transformers **惰性导入**
（本模块 import 期零重依赖；`make selfcheck` 保持 NONE）。

pool 模式不用它（那里用 `pool._hash_embed` 的确定性替身）；测试用 mock embed_fn 注入，不下载模型。
"""
from __future__ import annotations

import os


class HFEmbedder:
    """进程内单例缓存 model/tokenizer（同一模型只加载一次）。"""

    _model = None
    _tokenizer = None
    _device = None
    _loaded_path = None

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("LYCHEE_EMBED_MODEL")
        if not self.model_path:
            raise RuntimeError(
                "agentinit generate mode needs a text encoder: set env LYCHEE_EMBED_MODEL "
                "to an HF model path, or inject an embed_fn (tests use a fake one)."
            )

    def _ensure_loaded(self):
        import torch  # 惰性导入
        from transformers import AutoModel, AutoTokenizer

        if HFEmbedder._model is None or HFEmbedder._loaded_path != self.model_path:
            HFEmbedder._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            HFEmbedder._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            HFEmbedder._model = AutoModel.from_pretrained(self.model_path).to(HFEmbedder._device)
            HFEmbedder._model.eval()
            HFEmbedder._loaded_path = self.model_path

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        import torch

        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def embed(self, sentences, batch_size: int = 32):
        """(list[str]) -> np.ndarray (n, d)，L2 归一化。"""
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()
        if isinstance(sentences, str):
            sentences = [sentences]

        out = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = HFEmbedder._tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(HFEmbedder._device)
            with torch.no_grad():
                model_output = HFEmbedder._model(**enc)
            emb = self._mean_pooling(model_output, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())

        import numpy as np

        return np.concatenate(out, axis=0)
