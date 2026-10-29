# 五模块全插件化重构方案

> 状态：**已执行至 P1+P2+P4 主体**（2026-09-03）：methods/backends 两层立层、五接缝接口就位、
> 旧接缝组（Orchestrator/adapters/mock/autogen 线）已删除、docs/README 全面改写。
> 追加决策（2026-09-03）：runtime/ 兼容层**整体删除**（含 MASGraph/injection/langgraph 后端与
> run_mas 线脚本配置），P3 记忆挂载按新架构全新实现（旧注入引擎见 git 历史）。
> 剩余：P3 memory 挂载全新实现（attach_memory 显式桩）、
> GEPA graph-native 适配、run_mas 的 lg 原生重写、脚本三合一收尾。
> 用户决策：①五个模块全部收敛为挂载式插件；②`lg_prerun` 改名 `prerun`；③plugins/ 只做
> 接口层，复杂实现放同级 `methods/`（按接缝镜像分组）；④训练不立第六接缝——写侧入口并入
> postrun（`train_from_runs`），产物经 prerun apply 挂载；⑤`eval/` 维持顶层独立。
> 基准形态 = `plugins/prerun`：统一入口 + method 按名分发 + 图进图出/包裹执行
> + optimize/apply 两段式 + AgentSpec 节点契约。

---

## 1. 目标形态：五个模块 = 五个挂载式接缝

全部 LangGraph 原生；每个接缝一个统一入口、一个 REGISTRY 类别、换算法 = 换 `method`：

两层结构：**plugins/ 定义接缝（薄），methods/ 存方法（厚）**，按接缝互相镜像。

```
src/lychee_mas/
├── plugins/            接口层（薄）：协议、统一入口、method 分发、校验、薄适配、graphview 契约
│   ├── build.py        构建   build_langgraph(method, query, **kw) -> StateGraph   [graph_builder]
│   ├── prerun/         运行前 optimize_langgraph(sg, method, **kw) -> sg  ✅ 范式样板
│   │                   [pre_run_optimizer]（base + graphview + 薄适配 agentprune_lg）
│   ├── memory.py       运行时 attach_memory(sg, method, backend, **kw) -> sg
│   │                   [memory_manager + memory_router] 注入六步包裹进 agent 节点
│   ├── processing.py   执行   run_processed(sg, method, k, aggregator, **kw) -> 结果
│   │                   [processor + aggregator]  serial / parallel×K + 归约（pass@K 承载点）
│   └── postrun.py      运行后 analyze_run(result, method) 读侧归因
│                       + train_from_runs(method) 写侧离线训练  [post_run_optimizer + trainer]
├── methods/            实现层（厚）：算法本体/论文复现/重机器，按接缝镜像分组
│   ├── build/          static 模板、agentinit（Pareto 选队）
│   ├── prerun/         agentprune.py、maspo/（executor/prompts/textops/optimizer）、gepa/
│   ├── memory/         channels/ managers/ routing/ store/ context
│   ├── processing/     serial/parallel、self_consistency 等归约器
│   └── postrun/        attributors、credit_assigners、（未来 RL trainer）
├── eval/               ✅ 顶层独立（benchmarks + metrics + task_config）
├── core/               types + registry（地基）
└── backends/           生成原语（hf / openai_api / spans）—— P4 由 runtime/ 瘦身改名
```

约束：plugins/ 内任何文件不含论文级算法逻辑（目标 <300 行/文件）；带引用、训练循环、
缓存执行器的代码一律在 methods/。训练不立第六接缝：离线产物化走 prerun 的
mode="optimize"（MASPO/AgentPrune/GEPA 已然如此），RL 权重级训练消费 postrun 信号、
产物仍经 prerun apply 挂载。

生命周期：`build → prerun(可选) → memory(可选) → compile → processing 包裹执行 → postrun(可选)`。
每一环不配置时行为与不挂载完全一致（零回归原则沿用）。

## 2. 现存模块处置表

| 现位置 | 处置 | 去向/理由 |
|---|---|---|
| `plugins/prerun/` | **拆分 P1** | 接口件（base/graphview/agentprune_lg）留 plugins/prerun；maspo/ 实现体迁 methods/prerun/maspo |
| `layers/construct/` | **迁移 P1** | 入口 → `plugins/build.py`；实现（templates/agentinit）→ `methods/build/`；四个脚本重复建图代码归此 |
| `layers/prune/` | **迁移 P1** | agentprune 算法体 → `methods/prerun/`；桩随迁占名；layers/ 目录随 P2 清空退役 |
| `memory/` + `runtime/injection.py` | **迁移 P3** | memory/ 算法库 → `methods/memory/`；注入六步 → `plugins/memory.py` 节点包裹挂载 |
| `layers/processing/` + aggregator | **迁移 P2** | 入口 → `plugins/processing.py`；实现 → `methods/processing/` |
| `trace/` + `train/` | **迁移 P2** | 协议 → `plugins/postrun.py`（analyze_run + train_from_runs 双入口）；桩 → `methods/postrun/`；train/ 目录删除 |
| `plugins/gepa/ + program.py` | **迁移候选（P4 定）** | 以 `method="gepa"` 并入 prerun，或维持独立 optimizer 接缝 |
| `eval/`、`core/` | **保留** | 两线共用地基，不属五模块 |
| `runtime/backends/{hf,openai_api}` + spans | **迁移 P4** | → 顶层 `backends/`（runtime/ 其余退役后瘦身改名） |
| `runtime/backends/langgraph_runtime.py` | **P3 吸收** | 其 MASState/终止语义并入 build+memory 挂载后退役 |
| **旧接缝组**：`pipeline.py`、`plugins/adapters.py`、`plugins/base.py` 三协议、`mock_runtime`、`run_experiment.py`、`examples/01`、`configs/config.yaml`、`configs/plugins/`、旧接缝测试×3 | **删除 P4** | 被五接缝完全取代；替代物（build/processing/postrun + lg 原生 demo/测试）就位后一次性下线 |
| **AutoGen 执行线**：`autogen_runtime.py`、`autogen_injection_client.py`、`configs/runtime/autogen.yaml` | **删除 P4** | 新指令下不再等双后端对拍门槛；随旧接缝退役 |
| `run_mas.py` + `default.yaml` + `aime_*.yaml` | **迁移 P3** | 重写为 lg 原生（build+memory+processing 挂载版）；配置随之收编 |
| `run_langgraph_{baseline,construct,preplug}.py` | **合并 P1** | 三合一 `run_langgraph.py`（`--pre-optimize` 走统一接口）；preplug 的旧接缝演示退场 |
| `scripts/test_generate_team.py` | **迁移 P1** | 改名 `smoke_agentinit_generate.py`（在线冒烟，非单测） |
| `src/autogen/`（1831 文件/56MB vendored） | **删除 P0 ✅** | 零引用死重 |
| 本地垃圾（DOCUMENTS/HybridMem/site/空 config 目录/.DS_Store/egg-info） | **删除 P0 ✅** | 未跟踪杂物 |
| `requirements/benchmark.txt`、`docker/` | **保留** | 评测环境基建 |

## 3. 分阶段执行

- **P0 ✅（本次）**：死重与垃圾清理；pyproject 陈旧排除项清理；本方案文档。
- **P1 build 接缝 + methods/ 立层**：`plugins/build.py` + `methods/{build,prerun}/`（construct/agentprune/maspo 实现体迁入）+ 四脚本建图去重 + 三合一运行脚本 + preplug 退场。（prerun 改名已先行完成 ✅）
- **P2 processing / postrun 接缝**（小）：两个包裹接缝 + pass@K 承接 + 桩迁移。
- **P3 memory 挂载**（最大）：`attach_memory` 节点包裹语义 + run_mas 的 lg 原生重写 + 双实现对拍回归。
- **P4 旧接缝退役**：删除表中"删除 P4"整组；CLAUDE.md / DESIGN.md / README / make demo 全面改写为五接缝口径。

每阶段收尾：三件套全绿 + 小样本回归对拍（P3 另需与 run_mas 旧实现对拍 predictions 一致）。

## 4. 待签字的接缝设计决策

1. **memory 挂载语义**：推荐 `attach_memory` 重包节点 runnable（对建图方零侵入，与 prerun 同风格）；备选：建图方显式接受 memory 回调（侵入建图契约）。
2. **processing 入口**：推荐吃未编译 `sg`（内部按 K 编译/并发实例化）；备选：吃编译后 app。
3. **GEPA 去向**：并入 prerun（`method="gepa"`，MASProgram 收敛到 AgentSpec 提示视图）或保持独立 optimizer 接缝。
4. **AutoGen 退役提前**（P4 随旧接缝下线，不再等双后端对拍门槛）——请确认。
