# LycheeMAS Makefile —— recipe 前缀用 '>'（CLAUDE.md §2 的 .RECIPEPREFIX 约定）。
# 环境约定：先 `conda activate LycheeMAS && source .venv/bin/activate`（uv 管理的 venv），再跑 install/install-all。
.RECIPEPREFIX := >
.PHONY: venv install install-all demo test lint format typecheck clean snapshot selfcheck \
        docs-install docs docs-remote docs-build

# 用 uv 创建项目 .venv（Python 3.12）；之后 `source .venv/bin/activate` 再 make install-all
venv:
> uv venv .venv --python 3.12

# 仅骨架 + 开发工具：纯离线 mock 即可跑通（无需 autogen/torch/API）。须先激活 .venv。
install:
> uv pip install -e ".[dev]"

# 全量：autogen + torch/transformers + 数据/数学评分依赖 + dev。须先激活 .venv。
# 注意：勿加 --upgrade，避免覆盖 .venv 内已装的 CUDA 版 torch / vLLM。
install-all:
> uv pip install -e ".[all]"

# 离线端到端示例（runtime=mock，零重依赖）
demo:
> PYTHONPATH=src python examples/01_five_seams_demo.py

# 单元测试（mock runtime，无需 API key）
test:
> PYTHONPATH=src pytest -q

# 代码风格检查（只 lint 本框架，src/autogen 与旧 LycheeMAS 已排除）
lint:
> ruff check src

format:
> ruff format src/lychee_mas

typecheck:
> mypy src/lychee_mas

clean:
> find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
> rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info src/*.egg-info

# 打印所有已注册组件（看现有可用实现）
snapshot:
> PYTHONPATH=src python -c "import lychee_mas; from lychee_mas.core.registry import REGISTRY; import pprint; pprint.pprint(REGISTRY.snapshot())"

# 验证零重依赖：import 框架后不应加载 torch/autogen 等（应输出 HEAVY LOADED: NONE）
selfcheck:
> PYTHONPATH=src python -c "import sys, lychee_mas; print('HEAVY LOADED:', [m for m in ('torch','transformers','autogen_core','autogen_agentchat','langgraph','langchain_core','numpy','yaml','sympy','datasets') if m in sys.modules] or 'NONE')"

# ---- 文档站（MkDocs Material + mkdocstrings；griffe 静态解析，构建不 import 框架，无需 torch/autogen）----
# DISABLE_MKDOCS_2_WARNING：屏蔽 gen-files/literate-nav/section-index 依赖的 properdocs 打的推广横幅。
docs-install:
> uv pip install -e ".[docs]"

# 本机热更新预览：改 docstring/页面即刷新。打开 http://127.0.0.1:8000
docs:
> DISABLE_MKDOCS_2_WARNING=true mkdocs serve -a 127.0.0.1:8000

# 远程服务器：绑 0.0.0.0，从本地浏览器访问 http://<服务器IP>:8000
docs-remote:
> DISABLE_MKDOCS_2_WARNING=true mkdocs serve -a 0.0.0.0:8000

# 生成静态站点到 site/（--strict：断链/坏引用即失败）
docs-build:
> DISABLE_MKDOCS_2_WARNING=true mkdocs build --strict
