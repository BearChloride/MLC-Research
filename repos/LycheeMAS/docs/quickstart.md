# 快速上手

## 离线端到端 demo（零重依赖）

```bash
make demo
# 等价：PYTHONPATH=src python examples/01_static_chain_e2e.py
```

五接缝离线端到端（LLM 为脚本化假后端，需 `.[langgraph]` extra）：`build_langgraph` 建契约图 → `optimize_langgraph` 挂载优化提示 → `compile` → `run_processed` 并发投票归约，打印最终答案与 `REGISTRY.snapshot()`。**无需 GPU / API key。**

## 看现有组件 / 校验零重依赖

```bash
make snapshot     # 打印 REGISTRY.snapshot()：每个类别下已注册的实现名
make selfcheck    # 应打印 HEAVY LOADED: NONE（注册路径不触发 torch/autogen）
make test         # 离线单元测试（LLM 全脚本化，无需 GPU/API）
```

## 五接缝挂载（换算法 = 换 `method`）

```python
from lychee_mas.plugins import build_langgraph, optimize_langgraph, run_processed

sg = build_langgraph(method="static", node_factory=..., state_schema=..., team="default")
sg = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file="p.json")
result = await run_processed(runner, method="parallel", k=8, aggregator="self_consistency")
```

真实实验：`scripts/run_maspo_langgraph.py`（MASPO × MATH-500）、`scripts/run_agentprune_gsm8k.py`（AgentPrune × GSM8K）、`scripts/run_mas.py`（记忆线 benchmark）。

## 带记忆通道的真实 AIME 实验（需 `.[all]` + GPU）

`scripts/run_mas.py` 把 backend + 记忆 manager + 固定通道路由 + 处理层接成端到端实验（`--runtime autogen|langgraph`）。四种记忆通道消融：`none | nl_only | latent_only | both`；latent 走 `memory.latent_strategy`（`soft_token` / `c2c`）；NL 走 `memory.nl_strategy`（`prev_output` / `simplemem`）。

```bash
export LYCHEE_HF_MODEL=/path/to/Qwen3-4B     # HF 后端模型路径（不写进代码）
CDM_DATA_ROOT=/path/to/Data/raw CUDA_VISIBLE_DEVICES=0 \
    python scripts/run_mas.py --config configs/aime_latent_c2c.yaml
```

**跑几次由处理层决定**：`run.samples=K`，K=1 用 `processor/serial`（跑 1 次），K>1 用 `processor/parallel`（并发跑 K 次 + `aggregator` 聚合）。pass@1 对每条轨迹单独评分。结果落 `eval.results_root`（metrics.json + outputs.jsonl + config 快照）。

## 下一步

- 想懂模块职责与接口契约 → **[架构设计](DESIGN.md)**
- 想加一个组件（六步配方）→ **[开发指南](contributing.md)**
- 想查某个类/函数 → **[API Reference](reference/lychee_mas/)**
