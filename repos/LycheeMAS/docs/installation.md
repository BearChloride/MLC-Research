# 安装

`src-layout`，包名 `lychee_mas`，发行名 `lychee-mas`。**核心骨架零运行依赖**：纯标准库即可 `import lychee_mas`（demo 另需 langgraph extra）。

**环境约定**：conda 环境 `LycheeMAS`（系统 / CUDA 工具链）+ uv 管理的 `.venv`（torch / transformers / vLLM 等重依赖已就位）。包管理统一用 `uv`。

## 日常：进入环境

```bash
conda activate LycheeMAS
source .venv/bin/activate          # 首次创建见下方「初始化 .venv」
```

## 安装可编辑包（在已激活的 `.venv` 内）

```bash
uv pip install -e ".[dev]"         # 仅骨架 + 开发工具：离线示例即可跑通（无需 torch/API）
uv pip install -e ".[all]"         # 全量：autogen + 真实推理/评测依赖（torch/transformers/vLLM 已在 .venv 内）
uv pip install -e ".[docs]"        # 文档站工具：MkDocs Material + mkdocstrings（构建本站）
```

!!! warning "勿加 `--upgrade`"
    `.venv` 内已是 CUDA 版 torch / transformers / vLLM；默认 `uv pip install` 不会改动已满足约束的包。**加 `--upgrade` 会把它们换成无 CUDA 的 PyPI 轮子。**

## 可选依赖组（extras）

| extra | 内容 | 何时需要 |
|---|---|---|
| `dev` | pytest / ruff / mypy | 开发 + 离线测试 |
| `runtime` | autogen 0.7.x + torch + transformers | 真实推理 / latent 注入 |
| `numeric` / `config` | numpy / pyyaml + hydra | 按需 |
| `all` | 上面全部 + datasets / sympy 等评测依赖 | 真实跑分 / 训练 |
| `docs` | mkdocs-material + mkdocstrings 等 | 构建本文档站 |

## 初始化 `.venv`（首次 / 换机）

```bash
conda create -n LycheeMAS python=3.12 -y && conda activate LycheeMAS
uv venv .venv --python 3.12        # 用 uv 托管的 CPython 3.12 建 venv（等价 `make venv`）
source .venv/bin/activate
uv pip install -e ".[all]"
```

重依赖（torch / transformers / autogen / numpy / yaml / sympy …）一律**惰性导入**：缺这些库时 `import lychee_mas` 与 `REGISTRY.snapshot()` 仍可成功（纯离线开发只需 `.[dev]`）。校验：

```bash
make selfcheck    # 应打印：HEAVY LOADED: NONE
```
