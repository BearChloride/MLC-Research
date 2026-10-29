"""自动生成 API Reference 页面（mkdocs-gen-files）——遍历 src/lychee_mas，每个模块一页。

对标 vLLM 的 API reference：mkdocstrings 用 griffe **静态解析源码**（不 import 模块），为每个
`.py` 生成一个虚拟 markdown 页（内容 = `::: lychee_mas.<dotted.path>` 指令），并写
`reference/SUMMARY.md` 供 literate-nav 按子包分组。构建文档因此**不需要** torch/autogen，
`make selfcheck` 仍 `HEAVY LOADED: NONE`。

本脚本由 mkdocs.yml 的 gen-files 插件在构建时执行；不参与 `ruff check src`（在 docs/ 下）。
"""
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
ROOT = Path(__file__).parent.parent  # 仓库根（docs/ 的上一级）
SRC = ROOT / "src"
PKG = SRC / "lychee_mas"  # 只文档化本框架包；vendored src/autogen、src/LycheeMAS 自然排除

for path in sorted(PKG.rglob("*.py")):
    module_path = path.relative_to(SRC).with_suffix("")  # lychee_mas/memory/base
    doc_path = path.relative_to(SRC).with_suffix(".md")  # lychee_mas/memory/base.md
    parts = tuple(module_path.parts)  # ('lychee_mas', 'memory', 'base')

    if parts[-1] == "__init__":  # 包 __init__ -> 包级页 index.md
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    if not parts:
        continue

    full_doc_path = Path("reference", doc_path)
    # 侧栏分组：去掉冗余的 "lychee_mas" 根，让子包（core/runtime/memory/…）直接作 API 顶层分组，
    # 每个模块都在侧栏可见（否则整棵树塞在一个折叠的 lychee_mas 根下，只显示一项）。
    nav_parts = parts[1:] or ("lychee_mas（包总览）",)
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# `{ident}`\n\n::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(ROOT))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
