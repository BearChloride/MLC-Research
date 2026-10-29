# LycheeMAS

**多智能体系统（MAS）研究框架。** 核心思想：把整个 MAS 统一表示成一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V=智能体节点、E=通信边、W=边权、T=多轮时序、M=记忆状态），载体是 LangGraph `StateGraph` + AgentSpec 节点契约；**五个研究模块 = 五个挂载式接缝**，一切算法经 REGISTRY 按 `method` 名挂载，**可插拔、可消融**。

!!! tip "本站是什么"
    这是 LycheeMAS 的开发文档站：叙述性**指南**（安装 / 快速上手 / 架构 / 核心概念 / 组件开发）+ 从代码 docstring 自动生成的 **[API Reference](reference/lychee_mas/)**（每个模块一页，始终与源码同步）。

# 整体架构：五接缝

| 接缝 | 统一入口（`plugins/`） | 实现（`methods/`） | 状态 |
| --- | --- | --- | --- |
| 构建 build | `build_langgraph(method, ...)` | `static` 模板 / **AgentInit** 选队（EMNLP'25） | 已实现 |
| 运行前 prerun | `optimize_langgraph(sg, method)` | **MASPO** 提示联合优化（ICML 2026）/ **AgentPrune** 剪枝（ICLR 2025）/ **AgentDropout** 节点淘汰（ACL 2025）/ GEPA | 已实现（GEPA 图原生适配待接） |
| 记忆 memory | `attach_memory(sg, method)` | channels（NL/隐空间/C2C）+ managers（**cdm**）+ routing | 算法库已实现；挂载语义 P3 全新实现（接缝已立，显式桩） |
| 执行 processing | `run_processed(runner, method)` | serial / parallel×K + self_consistency 归约 | 已实现（pass@K 承载点） |
| 归因训练 postrun | `analyze_run(...)` / `train_from_runs(...)` | attributor / credit_assigner / trainer | 接缝已立，方法为桩（占名待接） |

两层结构：**`plugins/` 定义接缝（薄：协议 + 统一入口 + method 分发），`methods/` 存方法（厚：论文复现 / 训练循环）**，按接缝互相镜像。生命周期：

```text
build → prerun(可选) → memory(可选) → compile → processing 包裹执行 → postrun(可选)
```

## 源码结构（`src/lychee_mas/`）

> 每个模块的类/函数详情见 **[API Reference](reference/lychee_mas/)**。

```text
src/lychee_mas/
├── plugins/                          # ★ 接口层（薄）
│   ├── build/                        #   build_langgraph + AgentSelector/GraphBuilder 协议
│   ├── prerun/                       #   optimize_langgraph + graphview 节点契约 + 薄适配
│   ├── memory/                       #   attach_memory（P3 实现中，显式桩）
│   ├── processing/                   #   run_processed + Processor/Aggregator 协议
│   └── postrun/                      #   analyze_run + optimize_postrun + train_from_runs
├── methods/                          # ★ 实现层（厚，按接缝镜像）
│   ├── build/                        #   static 队伍模板 + agentinit 选队（Pareto 多样性×相关性）
│   ├── prerun/                       #   agentprune / agentdropout（两阶段淘汰）/ maspo/ / gepa/
│   ├── memory/                       #   channels{nl,latent,c2c} + managers{cdm,外部桩} + routing + store/context
│   ├── processing/                   #   serial / parallel + self_consistency / dynamicagg
│   └── postrun/                      #   attributor·credit_assigner 桩 + TraceStore
├── eval/                             # 评测（顶层独立）
│   ├── benchmarks/                   #   20 个基准（gsm8k/aime_2024/math500/…，惰性加载）
│   ├── metrics.py                    #   score（exact/aime/mc/f1/…）+ 落盘 + pass@K 聚合
│   └── task_config.py                #   每个 task 的默认队伍 + 答案提取策略
├── core/                             # 公共基座（零重依赖）
│   ├── types.py                      #   AgentSpec（节点契约载体）/Message/Trajectory/TaskQuery/…
│   └── registry.py                   #   REGISTRY（@register / create / snapshot，类别按接缝分组）
└── backends/                         # 生成原语（怎么调一个 LLM）
    ├── hf_backend.py                 #   本地 HF（generate_chat / encode_hidden / KV 原语）
    ├── openai_api_backend.py         #   OpenAI 兼容 API
    └── spans.py                      #   JsonlSpanLogger 运行事件落盘

configs/    按接缝分组 YAML（build / prerun / memory / processing / benchmarks）
examples/   01_five_seams_demo.py（make demo：五接缝离线端到端）
scripts/    实验入口（run_maspo_langgraph / run_agentprune_gsm8k / analyze_benchmark_run）
tests/      pytest（离线、LLM 全脚本化）
docs/       MkDocs 文档站（本站）+ DESIGN.md（唯一架构设计文档）+ plans/
```

## 四个设计原则

- **可插拔可消融**：每个算法 = 注册一个类（`@REGISTRY.register(category, name)`）+ `method` 按名挂载，换单一方法名即一组对照实验，**不改任何接缝文件**。
- **接口/实现/原语三层隔离**：plugins（接缝）/ methods（方法）/ backends（LLM 原语）；langgraph 与模型库一律惰性导入，`import lychee_mas` 零重依赖。
- **性能-成本联合度量**：评测同时报 accuracy / token / latency。
- **可复现**：固定随机种子；落 config 快照 + git SHA 到 `runs/`。

## 从这里开始

<div class="grid cards" markdown>

- :material-download: **[安装](installation.md)** —— conda + uv，extras（dev / all / langgraph / construct / docs）
- :material-rocket-launch: **[快速上手](quickstart.md)** —— `make demo` / 五接缝挂载 / 真实 benchmark 实验
- :material-sitemap: **[架构设计](DESIGN.md)** —— 接缝职责 + 节点契约 + 组件全景
- :material-puzzle: **[开发指南](contributing.md)** —— 六步配方：注册一个类 = 一组消融
- :material-book-open-variant: **[API Reference](reference/lychee_mas/)** —— 每个模块一页，与源码同步

</div>
