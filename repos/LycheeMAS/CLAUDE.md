# CLAUDE.md — LycheeMAS 开发规范

> 本文件供 **Claude Code / 编码代理与开发者** 在本仓库工作时阅读，是**操作规范**的单一事实来源：环境、命令、代码规范、开发流程、检查清单。**架构设计（目标、模块职责、接口契约、组件全景）见 `docs/DESIGN.md`**——动手前先读它，再读你要改的包的 `README.md` 与 `base.py`。

---

## 1. 一句话认知

LycheeMAS 是**多智能体系统（MAS）研究框架**：整个 MAS = 一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**；**五个模块 = 五个挂载式接缝**（build → prerun → memory → compile → processing → postrun），一切算法经 REGISTRY 按 `method` 名挂载在 LangGraph 契约图上。两层结构：**`plugins/` 定义接缝（薄），`methods/` 存方法（厚）**，按接缝互相镜像；`eval/` 评测、`core/` 类型+注册表、`backends/` 生成原语。动手前：`make snapshot` 看现有组件，读 `docs/DESIGN.md` 对应章节。

---

## 2. 环境与常用命令

```bash
# 环境（conda 环境 CDM = 本机唯一装齐 autogen+torch+langgraph 的环境）
conda activate CDM

# 安装（src-layout，可编辑装）
uv pip install -e ".[dev]"        # 骨架 + 开发工具：离线示例即可跑通（无需 torch/API）
uv pip install -e ".[all]"        # 全量：autogen + langgraph + torch/transformers 等（真实跑分用）
                                  # ⚠ 勿加 --upgrade：环境内是 CUDA 版 torch，升级会换成无 CUDA 轮子
# 按需 extras：[langgraph]（LangGraph 后端）/ [benchmark]（基准数据准备）/ [construct]（agentinit）

# 日常（Makefile 用 '>' 作 recipe 前缀）
make demo        # 离线端到端：PYTHONPATH=src python examples/01_static_chain_e2e.py
make test        # PYTHONPATH=src pytest -q
make lint        # ruff check src
make snapshot    # 打印 REGISTRY.snapshot()（看现有组件）
make selfcheck   # 验证零重依赖：必须打印 HEAVY LOADED: NONE

# 真实实验（需 GPU；模型/数据路径走 config 或环境变量，绝不写进代码）
CDM_DATA_ROOT=/data/.../raw CUDA_VISIBLE_DEVICES=0 \
    python scripts/run_mas.py --config configs/<experiment>.yaml \
    [--runtime autogen|langgraph] [--n 3] [--samples 8]
python scripts/analyze_benchmark_run.py <run_dir> --score-predictions   # 事后打分
```

约束：**核心代码与离线示例不得要求 GPU 或 API key**；真实推理才需要，缺依赖时必须显式报错。

---

## 3. 代码规范（黄金法则，违反即返工）

1. **接口/实现/原语三层隔离。** `plugins/` 只放协议、统一入口、method 分发、校验与薄适配（不含论文级算法，目标 <300 行/文件）；论文复现、训练循环、重机器一律在 `methods/`（按接缝镜像分组）；模型库（transformers/openai）只经 `backends/` 触达，methods 里的算法用注入的回调调 LLM。`langgraph` 允许出现在 plugins / methods，但**必须惰性导入**（函数内部）。
2. **重依赖一律惰性导入。** `torch / transformers / autogen_* / langgraph / numpy / yaml / sympy / datasets` 只能在函数/方法内部导入；**注册组件的模块被 import 时不得触发这些库**。校验：`make selfcheck` 必须打印 `HEAVY LOADED: NONE`。
3. **每个算法 = 注册一个类 + method 按名挂载，绝不硬编码。** `@REGISTRY.register(category, name)` 打在 `methods/` 的实现类上；新增方法**不改任何 plugins/ 接口文件**；对照实验只换 `method` 名。
4. **公共类型只放 `core/types.py`**；接缝协议放 `plugins/` 对应接缝文件，方法族内部契约留在 `methods/<接缝>/` 的 base 文件，不塞进 core。
5. **保持类型注解；提交前 `make lint` + `make test` + `make selfcheck` 三者全绿。**
6. **显式错误，禁止静默兜底。** 组件遇到不支持的输入/配置显式 raise；严禁写死数据兜底造成假阳性；严禁静默降级掩盖配置错误。
7. **可复现**：固定种子；实验落 config 快照 + git SHA；评测同时报告 accuracy / token / latency。
8. **不提交密钥与大产物**：`runs/`、checkpoints、`*.jsonl`、`.env` 在 `.gitignore`；模型/数据路径走环境变量或 config。
9. **整合外部论文代码**：先适配对应 `base.py` 协议 + 注册，再迁移逻辑；保留原始引用与许可证；不整包搬仓库。

---

## 4. 开发流程

### 4.1 新增/实现一个组件（六步配方 = 一篇消融）

1. **读接口**：`plugins/<接缝>` 的协议与统一入口 + `docs/DESIGN.md` §6 全景表确认类别与签名。
2. **写实现**：在 `methods/<接缝>/` 实现协议 + `@REGISTRY.register(category, name)`；重依赖在方法内惰性 import；LLM 经注入回调触达（不直接 import 模型库）。
3. **触发注册**：在 `methods/<接缝>/__init__.py` import 你的模块（被 `lychee_mas/__init__` 链式 import）。
4. **加配置**：`configs/<接缝>/<name>.yaml`（超参 + 默认值）。
5. **加测试**：`tests/test_<name>.py`，LLM 全部脚本化、构造输入断言行为（含显式报错路径）。
6. **验证**：`make lint && make test && make selfcheck`，再 `make demo` 看五接缝端到端不回归。

### 4.2 改动前后的固定动作

- 改前：`git pull` → `make snapshot` → 读 `docs/DESIGN.md` 对应章节 + 目标包 `README.md`/`base.py`；不确定接口形状看 `examples/` 与 `tests/`，**不要臆造类型**。
- 改后：三件套全绿；改了执行链路（runtime / injection / memory 消费方）→ 先跑离线自检，再跑小样本真实回归，与改前基线对拍（`predictions.jsonl` 逐字一致、spans 滤易变字段一致）。
- 双后端等价性改动（涉及 autogen/langgraph 共同语义）→ 同配置双后端对拍：final answer、`model_call_start.input_messages`、token 记账三项一致。

---

## 5. 提交前检查清单（PR 自检）

- [ ] `make lint` / `make test` / `make selfcheck`（HEAVY LOADED: NONE）三者全绿。
- [ ] 新组件已：实现协议 + 注册 + 子包 `__init__` 触发 + config + test。
- [ ] 没有在业务层 import 执行引擎；重依赖均惰性导入；无静默兜底。
- [ ] 没有硬编码模型/数据绝对路径。
- [ ] 改了执行链路 → 跑过离线自检 + 小样本回归，数值无回归。
- [ ] 无密钥 / 无大文件 / 无 `runs/` 产物入库。
- [ ] 组件状态或接口变化 → 同步 `docs/DESIGN.md`（§7 组件全景表）与对应包 `README.md`。
