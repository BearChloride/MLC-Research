<h1 align="center">LycheeMAS</h1>

<div align="center">
    <img src="images/logo.png" width=250></img>
</div>

<p align="center"><strong>运行时可插拔的五层多智能体系统（MAS）研究框架</strong></p>
<p align="center"><em>一张图 G=(V,E,W,T,M)，五层变换 + 优化插件，组件可插拔、可消融、可复现。</em></p>

<p align="center">
  <a href="https://github.com/ecoli-hit/LycheeMAS"><img alt="Release v0.2" src="https://img.shields.io/badge/release-v0.2-111827?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square">
  <img alt="核心零依赖" src="https://img.shields.io/badge/core%20deps-zero-0EA5E9?style=flat-square">
</p>

<p align="center">
  <a href="https://github.com/ecoli-hit/LycheeMAS">GitHub</a> ·
  <a href="docs/DESIGN.md">架构设计</a> ·
  <a href="CLAUDE.md">开发规范</a> ·
  <a href="docs/ONBOARDING.md">团队入门</a> ·
  <a href="docs/GIT_GUIDE.md">Git 规范</a> ·
  <a href="docs/example-maspo.md">MASPO 实例</a> ·
  <a href="configs/">配置</a> ·
  <a href="examples/">示例</a>
</p>

> **把整个 MAS 统一表示为一张带时序与记忆状态的有向图 G=(V,E,W,T,M)，每一层都是对 G（或其执行轨迹 τ）的一次变换。**

> **版本范围：** 本 README 对应 **LycheeMAS v0.3**。

[框架总览](#框架总览) · [安装](#安装) · [跑 demo](#跑-demo离线零重依赖) · [跑测试](#跑测试) · [支持的 Benchmarks](#支持的-benchmarks) · [目录](#目录) · [团队文档](#团队文档)

---

## 框架总览

五模块多智能体系统（MAS）研究框架。把整个 MAS 统一表示为一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（载体 = LangGraph `StateGraph` + AgentSpec 节点契约）；**五个模块 = 五个挂载式接缝**，一切算法经 REGISTRY 按 `method` 名挂载。完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)：

| 接缝 | 统一入口（`plugins/`） | 实现（`methods/`） | 职责 |
| --- | --- | --- | --- |
| 构建 | `build_langgraph(method, ...)` | `static` 模板 / **AgentInit** 选队 | 产出契约 StateGraph |
| 运行前 | `optimize_langgraph(sg, method)` | **MASPO**（提示优化）/ **AgentPrune**（剪枝）/ **AgentDropout**（节点淘汰）/ GEPA | 执行前改写图；optimize 离线产物化 + apply 即插即用 |
| 记忆 | `attach_memory(sg, method)` | channels / managers（cdm）/ routing | 注入六步包裹进 agent 节点（实现中） |
| 处理 | `run_processed(runner, method)` | serial / parallel + self_consistency | 跑几次 + 归约（pass@K 承载点） |
| 归因训练 | `analyze_run(...)` + `train_from_runs(...)` | attributor / credit / trainer（桩） | 读侧归因信用 + 写侧离线训练 |

两层结构：**`plugins/` 定义接缝（薄），`methods/` 存方法（厚）**，按接缝镜像。设计四原则：**可插拔可消融**（registry + method 按名挂载）、**接口/实现/生成原语三层隔离**（plugins / methods / backends）、**性能-成本联合度量**、**可复现**。

---

## 安装

`src-layout`，包名 `lychee_mas`，发行名 `lychee-mas`。核心骨架**零运行依赖**：纯标准库即可 import 并跑通离线 mock 示例。

**环境约定**：conda 环境 `LycheeMAS`（系统 / CUDA 工具链）+ uv 管理的 `.venv`（torch / transformers / vLLM 等重依赖已就位）。包管理统一用 `uv`。

```bash
# 日常：进入环境
conda activate LycheeMAS
source .venv/bin/activate          # 首次创建见下方「初始化 .venv」

# 安装可编辑包（在已激活的 .venv 内执行）
uv pip install -e ".[dev]"         # 仅骨架 + 开发工具：离线示例即可跑通（无需 torch/API）
uv pip install -e ".[all]"         # 全量：autogen + 真实推理/评测依赖（torch/transformers/vLLM 已在 .venv 内）
```

> `.venv` 内已是 CUDA 版 torch / transformers / vLLM；默认 `uv pip install` 不会改动已满足约束的包，**请勿加 `--upgrade`**，以免把它们换成无 CUDA 的 PyPI 轮子。

**初始化 `.venv`（首次 / 换机）**：

```bash
conda create -n LycheeMAS python=3.12 -y && conda activate LycheeMAS   # 首次；已建好直接 conda activate LycheeMAS
uv venv .venv --python 3.12        # 用 uv 托管的 CPython 3.12 建 venv（等价 `make venv`）
source .venv/bin/activate
uv pip install -e ".[all]"
```

重依赖（torch / transformers / autogen / numpy / yaml / sympy …）一律惰性导入：缺这些库时 `import lychee_mas` 与 `REGISTRY.snapshot()` 仍可成功（纯离线开发只需 `.[dev]`）。

---

## 跑 demo（离线，零 GPU / 零 API）

```bash
make demo
# 等价于：
PYTHONPATH=src python examples/01_five_seams_demo.py
```

五接缝离线端到端（LLM 为脚本化假后端，需 `.[langgraph]` extra）：`build_langgraph`（构建契约图）→ `optimize_langgraph`（prerun apply 挂载优化提示）→ `compile` → `run_processed`（并发 3 次 + self_consistency 投票），打印最终答案与 `REGISTRY.snapshot()`。

统一挂载用法（换算法 = 换 `method` 字符串）：

```python
from lychee_mas.plugins import build_langgraph, optimize_langgraph, run_processed

sg = build_langgraph(method="static", node_factory=..., state_schema=..., team="default")
sg = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file="p.json")   # 或 method="agentprune"
result = await run_processed(runner, method="parallel", k=8, aggregator="self_consistency")
```

真实实验入口见 `scripts/run_maspo_langgraph.py`（MASPO × MATH-500 复现：baseline / optimize / eval 三条腿）与 `scripts/run_agentprune_gsm8k.py`（AgentPrune × GSM8K）。

---

## 跑测试

```bash
make test     # PYTHONPATH=src pytest -q
make lint     # ruff check src
```

全部离线（LLM 脚本化），无需 API key。

---

## 支持的 Benchmarks

内置 **20 个**基准（`eval/benchmarks/`，均注册为 `benchmark/<name>`，数据加载惰性；数据准备与重依赖走 `[benchmark]` extra）：

**文本推理 / 知识问答（7）**

| 注册名 | 任务 | 评分 |
| --- | --- | --- |
| `gsm8k` | 小学数学应用题 | exact |
| `aime_2024` | AIME 竞赛数学 | 数值 + 符号等价 |
| `math500` | MATH-500 竞赛数学（MASPO 主实验数据集） | 数值 + 符号等价 |
| `medqa` | 医学选择题 | mc |
| `arc_easy` | 科学常识选择题 | mc |
| `openbookqa` | 开放课本科学选择题 | mc |
| `locomo10` | 长时对话记忆问答 | token-F1 |

**代码 / 通用助理（5）**

| 注册名 | 任务 | 评分 |
| --- | --- | --- |
| `human_eval` | Python 代码生成 | 单测通过 |
| `gaia_validation` | GAIA 通用助理任务（validation 全集） | 短答案规整 |
| `gaia_validation_level_1` / `_2` / `_3` | GAIA 按难度分级子集 | 短答案规整 |

**MAS 轨迹分析（8）**

| 注册名 | 任务 | 评分 |
| --- | --- | --- |
| `aftraj_audit` / `aftraj_audit_test` | AFTraj 轨迹安全审计（safe/unsafe + 定位关键步/agent） | mas_audit |
| `agent_collab_idr` | AgentCollabBench：指令衰减（Instruction Decay） | mas_instruction_decay |
| `agent_collab_rtd` | AgentCollabBench：追踪耐久（Tracer Durability） | mas_tracer_durability |
| `agent_collab_cpr` | AgentCollabBench：共识污染（Consensus Pollution） | mas_consensus_pollution |
| `agent_collab_clc` | AgentCollabBench：上下文泄漏（Context Leakage） | mas_context_leakage |
| `mast_failure` | MAST 失败模式分类 | taxonomy-F1 |
| `open_agent_traces` | Open Agent Traces 轨迹偏差检测 | mas_deviation |

统一记录格式 `{task, kind, question, gold, context}`；加载入口 `benchmarks.load(task, n)`，数据准备 `benchmarks.prepare(task)`；推理/打分分离与 pass@K 口径见 `docs/DESIGN.md` §6。

---

## 目录

```
src/lychee_mas/
├── plugins/     接口层（薄）：五接缝统一入口 + 协议 + 节点契约
│   ├── build/ │ prerun/ │ memory/ │ processing/ │ postrun/
├── methods/     实现层（厚）：论文复现/训练循环，按接缝镜像
│   ├── build/（static, agentinit）  prerun/（agentprune, maspo/, gepa/）
│   ├── memory/（channels/managers/routing）  processing/（serial/parallel）
│   └── postrun/（attributors, credit, TraceStore）
├── eval/        benchmarks（20 个）+ metrics（评分/落盘/pass@K）+ task_config
├── core/        公共类型（types：AgentSpec 等）+ 组件注册表（registry）
└── backends/    生成原语（hf / openai_api / spans 落盘）
configs/         YAML 配置（按接缝分组：build / prerun / memory / processing / benchmarks）
examples/        01_five_seams_demo.py（make demo：五接缝离线端到端）
scripts/         实验入口（run_maspo_langgraph / run_agentprune_gsm8k / analyze_benchmark_run）
tests/           pytest（离线、LLM 全脚本化）
docs/            DESIGN.md（唯一架构设计文档）+ ONBOARDING / GIT_GUIDE / example-maspo（团队文档）+ plans/ + 文档站页面
```

> **架构设计**（接缝职责、节点契约、组件全景、评测体系）见 `docs/DESIGN.md`；**开发规范**（环境、命令、黄金法则、六步配方、检查清单）见 `CLAUDE.md`。

## 团队文档

新成员按顺序读这三份（均随仓库版本管理）：

| 文档 | 读什么 |
| --- | --- |
| [`docs/ONBOARDING.md`](docs/ONBOARDING.md) | 团队入门与开发指南：项目目标与总览（框架主图）、项目结构、各模块解读、模块开发与评测实例（六步配方） |
| [`docs/GIT_GUIDE.md`](docs/GIT_GUIDE.md) | Git 项目管理规范：fork + `<module>_<方法名>` 分支 + PR、push 前清单、五条红线、事故复盘 |
| [`docs/example-maspo.md`](docs/example-maspo.md) | MASPO 开发实例走读：一个 prerun 方法从协议、注册、测试到实验落盘的完整范例 |

---

<p align="center"><strong>把每一个 MAS 研究问题，变成同一张图上一次可消融的变换。</strong></p>
