# `eval` — 评测（基准 + 指标 + task 默认配置）

## 功能

评测 harness，遵循**推理 / 打分分离**：推理脚本（`scripts/run_mas.py`）只落 `predictions.jsonl` + `spans.jsonl`；打分由 `scripts/analyze_benchmark_run.py --score-predictions` 事后完成（落 `outputs.jsonl` + `metrics.json`）。

## `benchmarks/` — 基准数据加载与注册

**每条记录统一形如 `{task, kind, question, gold, context}`**（`kind` 对应 `metrics.py` 的评分类型）。

```python
from lychee_mas.eval import benchmarks
benchmarks.load(task, n=None) -> list[dict]        # 统一加载入口
benchmarks.prepare(task, force=False, source=None) # 数据准备（下载/转换）
bench = REGISTRY.create("benchmark", "gsm8k", n=100); bench.load()   # 注册组件形式（构造不碰数据）
```

已注册 **19 个**：

- 文本推理/知识：`gsm8k`、`aime_2024`、`medqa`、`arc_easy`、`openbookqa`、`locomo10`
- 代码/通用助理：`human_eval`、`gaia_validation`（+ `_level_1..3`）
- MAS 轨迹分析：`aftraj_audit`（+ `_test`）、`agent_collab_{idr,rtd,cpr,clc}`、`mast_failure`、`open_agent_traces`

组成：`__init__.py`（LOADERS/PREPARERS 登记 + 动态注册 `benchmark/<task>`）、`common.py`（数据根目录 `raw_root/prepared_root/processed_root/runs_root`、parquet IO、多 provider 下载）、`source_catalog.py`（数据源目录）、其余每基准一个 loader 模块。数据根目录走环境变量（`CDM_DATA_ROOT` 等）。

## `metrics.py` — 评分 + 聚合 + 落盘

```python
score(kind, pred, gold) -> float          # 统一打分入口；score_details 附细节
aggregate_samples(samples, run_info)      # accuracy/token/latency 汇总；多采样含 pass_at_k
result_dir / write_outputs / write_metrics / write_config / write_results
```

评分类型（`kind`）：`mc` / `exact` / `aime`（数值+sympy 符号等价）/ `f1` / `human_eval`（代码单测）/ `gaia` / `mas_audit`、`mas_failure_taxonomy`、`mas_deviation`、`mas_instruction_decay`、`mas_tracer_durability`、`mas_consensus_pollution`、`mas_context_leakage`。

pass@K 口径（`--samples K` 多采样时）：pass@1 = 各 case 内 K 份得分均值再对 case 平均；pass@K = 各 case best-of-K 再平均。

## `task_config.py` — task 级默认配置

```python
team_name_for_task(task) -> str           # 该 task 默认队伍 profile
extractor_for_task(task) -> Callable      # 答案提取策略（default / boxed；兼容对象与 dict 消息）
```

## 约定

- import 本包触发 benchmark 注册，但**不读盘、不导入 datasets/sympy/yaml**（惰性）；数据准备重依赖走 `[benchmark]` extra。
- 新增基准：写 loader/preparer 模块 → `__init__.py` 的 LOADERS/PREPARERS 登记（自动注册）→ `task_config.py` 与 `metrics.py` 补默认队伍/评分类型。
- 评测须同时报告 accuracy / token / latency。
