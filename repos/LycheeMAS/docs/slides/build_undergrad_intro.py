"""Generate the 6-page LycheeMAS undergraduate intro deck (python-pptx, no renderer needed).

Usage (from repo root):
    python docs/slides/build_undergrad_intro.py --out runs/slides/lycheemas_undergrad_intro.pptx
    python docs/slides/build_undergrad_intro.py --check        # enforce layout limits, print report
    python docs/slides/build_undergrad_intro.py --dump-text    # print every string on every slide

Every code snippet is read verbatim from the repo by path + line range; every number on the
experiment page is read from runs/ metrics (falls back to the literal recorded in SLIDES only
when the run directory is absent, and says so in --check).
Requires `python-pptx` (pure python; `pip install python-pptx`). Not a runtime dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- palette / type
ACCENT = "b5303d"   # 荔枝红
PAPER = "f7f7f8"
INK = "181a1f"
MUTED = "6b6f76"
GREY = "9a9ca3"
LINE = "e3e4e8"
CODE_BG = "eeeef1"
GREEN = "2f7d4f"
SOFT = "f5e6e8"
FONT = "Microsoft YaHei"
MONO = "Consolas"

SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.6

# limits enforced by --check
MAX_TITLE = 16       # CJK-equivalent chars (latin/digit = 0.5)
MAX_KICKER = 30
MAX_BULLETS = 5
MAX_BULLET = 34
MAX_CODE_LINES = 8
MAX_CODE_COLS = 80
MAX_TABLE = (5, 5)


def cjk_len(s: str) -> float:
    return sum(1.0 if ord(c) > 0x2E80 else 0.5 for c in s)


def read_lines(rel: str, start: int, end: int) -> list[str]:
    """Verbatim lines [start, end] (1-based, inclusive) from a repo file."""
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
    if end > len(lines):
        raise ValueError(f"{rel}: range {start}-{end} beyond EOF ({len(lines)} lines)")
    return [ln.rstrip() for ln in lines[start - 1 : end]]


def load_metrics(rel: str) -> dict | None:
    p = REPO / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ----------------------------------------------------------------------------- content
# Keys: n, title, kicker, bullets, figure, [active], [code], [table], [table_note], source_footer, notes
# figure ∈ cover | two_node | seams | table | seams+table | none
RUN_BASE = "runs/maspo/benchmarks_v2/Qwen3-8B/maspo_original/math500/metrics.json"
RUN_V1 = "runs/maspo/benchmarks_v2/Qwen3-8B_prompts-v1/maspo_optimized/math500/metrics.json"
RUN_V2 = "runs/maspo/benchmarks_v2/Qwen3-8B/maspo_optimized/math500/metrics.json"
# normalized re-scoring is reported in docs/example-maspo.md:136-140 (not in metrics.json)
NORMALIZED = {"base": "0.78", "v1": "0.85", "v2": "0.85"}
# literal values recorded from the metrics files above; used only when runs/ is absent (see --check)
FALLBACK = {
    "base": {"accuracy": 0.76, "mean_input_total_positions_per_case": 981.3, "mean_model_latency_s_per_case": 47.77},
    "v1": {"accuracy": 0.77, "mean_input_total_positions_per_case": 1891.8, "mean_model_latency_s_per_case": 71.808},
    "v2": {"accuracy": 0.8, "mean_input_total_positions_per_case": 1428.5, "mean_model_latency_s_per_case": 55.62},
}


def experiment_table() -> tuple[list[list[str]], list[str]]:
    """Baseline / v1 / v2 columns read from runs/ metrics.json; returns (rows, notes-about-fallbacks)."""
    warn: list[str] = []
    cols = {}
    for key, rel in (("base", RUN_BASE), ("v1", RUN_V1), ("v2", RUN_V2)):
        m = load_metrics(rel)
        if m is None:
            warn.append(f"{rel} absent -> using recorded literal for '{key}'")
            m = FALLBACK[key]
        cols[key] = m
    rows = [
        ["", "基线", "MASPO v1", "MASPO v2"],
        ["原始打分", *(f'{cols[k]["accuracy"]:.2f}' for k in ("base", "v1", "v2"))],
        ["归一化打分", NORMALIZED["base"], NORMALIZED["v1"], NORMALIZED["v2"]],
        ["prompt tok / 题", *(f'{cols[k]["mean_input_total_positions_per_case"]:.0f}' for k in ("base", "v1", "v2"))],
        ["延迟 s / 题", *(f'{cols[k]["mean_model_latency_s_per_case"]:.1f}' for k in ("base", "v1", "v2"))],
    ]
    return rows, warn


_EXP_ROWS, EXP_WARNINGS = experiment_table()

SLIDES: list[dict] = [
    {
        "n": 1,
        "title": "LycheeMAS",
        "kicker": "把多智能体系统当成一张图：G=(V,E,W,T,M)",
        "bullets": [
            "讲者：____    2026-09",
            "LycheeMAS v0.3 · github.com/ecoli-hit/LycheeMAS",
            "面向：会 Python、还没接触过 agent（智能体）的你",
        ],
        "figure": "cover",
        "source_footer": "docs/DESIGN.md · docs/ONBOARDING.md · docs/assets/main.png",
        "notes": "大家好。今天用一道数学题带大家认识 LycheeMAS：它把多个大模型协作的系统看成一张图，"
                 "任何算法都是往图上挂一个零件。五页讲完：一道题、框架长什么样、一次真实实验、"
                 "以及我负责的两个接缝——记忆中枢和运行前优化——各自能做什么。",
    },
    {
        "n": 2,
        "title": "一道题，加一个检查员",
        "kicker": "两个节点就是最小的 MAS（多智能体系统），它天然是一张图",
        "bullets": [
            "MATH-500 第 0 题：把点 (0,3) 化成极坐标 (r,θ)。",
            "一个 LLM（大语言模型）独自答题会出错——再请一个检查员。",
            "答题者 predictor_0 → 检查员 reflector_0，就是最小的 MAS。",
            "V=智能体节点，E=通信边，W=边权，T=多轮时序，M=记忆状态。",
            "提示词是节点 V 的属性（AgentSpec.system_prompt），不在边上。",
        ],
        "figure": "two_node",
        "source_footer": "scripts/run_maspo_langgraph.py:152-170 build_reflect_graph · docs/DESIGN.md:11 · "
                         "src/lychee_mas/core/types.py:26-37 · runs/maspo/…/maspo_original/math500/outputs.jsonl case 0",
        "notes": "这道题真的在我们的实验集里：把 (0,3) 化成极坐标。一个模型独答可能出错，最朴素的补救是再请一个检查员："
                 "predictor_0 先答，reflector_0 检查改写。两个框是代码里的真实节点名，箭头是通信边；"
                 "五个字母各管一件事。重点：提示词挂在节点上，不在边上——后面运行前优化改的就是它。",
    },
    {
        "n": 3,
        "title": "框架总览：五接缝、三层、注册表",
        "kicker": "改图的每个字母都有一个接缝；换算法 = 换一个 method 名",
        "bullets": [
            "build 造 V/E；prerun 运行前改 V/W/E/T；memory 挂 M。",
            "compile 编成可跑的图；processing 决定怎么跑；postrun 事后归因。",
            "三层：plugins/ 薄接口，methods/ 厚算法，只有 backends/ 碰模型库。",
            "@REGISTRY.register(category, name) 注册实现类，入口按名取用。",
            "make demo：无 GPU、无 API key，约 2 秒离线跑通五个接缝。",
        ],
        "figure": "seams",
        "code": {"path": "src/lychee_mas/plugins/prerun/base.py", "start": 60, "end": 64,
                 "caption": "统一入口 optimize_langgraph 的全部函数体：校验 → 按名取 → 优化 → 再校验"},
        "source_footer": "docs/DESIGN.md:11-40,116-136 · src/lychee_mas/plugins/prerun/base.py:46-64 · "
                         "src/lychee_mas/core/registry.py · examples/01_five_seams_demo.py",
        "notes": "框架就是给这张图的每个字母配一个改它的接口，我们叫接缝。plugins 只放接口和按名分发，"
                 "算法都在 methods 里注册一个类，模型库只在 backends 里出现。右边五行就是整个入口：校验、"
                 "从注册表按名字取算法、优化、再校验——新增算法接缝一行不改。没有 GPU 也能 make demo 跑通。",
    },
    {
        "n": 4,
        "title": "真实实验：MASPO × MATH-500",
        "kicker": "只改两个节点的提示词，归一化准确率 0.78 → 0.85",
        "bullets": [
            "Qwen3-8B、100 道 MATH-500、两节点拓扑，每题 3 次模型调用。",
            "MASPO（免标答的成对判官提示优化）只改 V 的提示词，+7pt。",
            "教训一：第 0 题答 (3, π/2) 被字面打分判 0，归一化后才见真差距。",
            "教训二：一次 17 小时优化因 API 额度耗尽全丢 → 每轮原子落盘 ckpt。",
            "代价：v2 优化 17.1 h，agent 2530 次 + 判官 1438 次模型调用。",
        ],
        "figure": "table",
        "table": _EXP_ROWS,
        "table_note": "v1 = gemini 评估端；v2 = 本地 Qwen3-32B 评估端（修正配置）。同口径 @4096、100 题；"
                      "归一化打分来自事后重打分。",
        "source_footer": "runs/maspo/benchmarks_v2/Qwen3-8B{,_prompts-v1}/*/math500/metrics.json · "
                         "docs/example-maspo.md:134-144 · runs/maspo/v2_optimize_eval.log:28-30",
        "notes": "同一张两节点图，只换提示词，归一化准确率从 0.78 到 0.85。但看原始打分几乎没动：第 0 题优化后答对了 (3, π/2)，"
                 "字面比对却判 0——判官看意思、打分器看字面，口径能翻转结论。另外一次 17 小时的优化因额度耗尽全丢，"
                 "从此每轮原子落盘，这是最能共情的工程教训。最后一条是代价：优化本身不便宜。",
    },
    {
        "n": 5,
        "title": "记忆中枢：给图挂上 M",
        "kicker": "接缝已立、契约已定；挂载实现待 P3——这里是空位",
        "bullets": [
            "M=记忆状态：reflector_0 记住 predictor_0 的什么、以什么表征传。",
            "入口 attach_memory(sg, method, **kwargs)：重包每个节点，图进图出。",
            "设计六步：observe→route→recall→system 段注入→生成→记账。",
            "现状：入口直接 raise NotImplementedError（桩 = 占好名的空位）。",
            "算法库已可跑：cdm 双通道管理器、static/fixed 路由、nl/latent 通道。",
        ],
        "figure": "seams+table",
        "active": ["memory"],
        "table": [
            ["部件", "REGISTRY 名", "现状", "可做"],
            ["入口 attach_memory", "—", "桩（P3）", "结对实现六步包裹"],
            ["memory_manager", "cdm / mem0 / ama", "cdm 已实现；mem0、ama 桩", "接 mem0 + 单测"],
            ["memory_router", "static / fixed / learned / soft_gate", "static、fixed 已实现；其余回退 static", "写真正的 learned 路由"],
            ["channels", "nl / latent（soft_token, c2c）", "已实现；tests/ 无单测", "补 DualChannel 测试"],
        ],
        "source_footer": "src/lychee_mas/plugins/memory/base.py:19-23 · docs/DESIGN.md:89-91,126-127 · "
                         "src/lychee_mas/methods/memory/{base.py,managers/,routing/,channels/} · tests/test_router.py",
        "notes": "M 管的是 reflector_0 该记住 predictor_0 的什么、用自然语言还是隐向量传。接口定了：六步包进每个节点；"
                 "但入口现在还是桩，调用就抛 NotImplementedError，这是 P3 要做的。算法库里 cdm 管理器、static 路由、"
                 "nl/latent 通道都能跑。没有 GPU 也能先接 mem0、补单测——这一页是空位，也是招新点。",
    },
    {
        "n": 6,
        "title": "运行前优化：改 V/W/E/T",
        "kicker": "optimize_langgraph(sg, method, **kw) → sg，图进图出",
        "bullets": [
            "两段式：optimize 离线重活落 JSON；apply 加载产物即插即用。",
            "入口三步：校验 StateGraph → REGISTRY 按名取 → optimize(graph)。",
            "改图只经 GraphView：extract_view 读视图，rebuild 写提示 / 邻接。",
            "节点函数与 metadata 共享同一 AgentSpec：改 spec 即改运行行为。",
            "可做：给 GEPA 接图进图出适配；agentvocab 桩待认领；新方法零改接缝。",
        ],
        "figure": "seams+table",
        "active": ["prerun"],
        "table": [
            ["方法", "改哪个字母", "对 predictor_0→reflector_0 做什么", "状态"],
            ["maspo", "V（节点提示词）", "重写两节点的 system_prompt", "已实现；apply / optimize"],
            ["agentprune", "W → E", "训逐边 logit，剪掉低权通信边", "已实现（apply）；训练走脚本"],
            ["agentdropout", "V/W/E/T（逐轮）", "每轮淘汰一节点并剪边，按 round 挂载", "已实现（apply）；v2 桩"],
            ["gepa / agentvocab", "V 提示词 / V 词表", "反思式提示演化 / 词表缩减", "算法已有待接图 / 桩"],
        ],
        "source_footer": "src/lychee_mas/plugins/prerun/{base.py:1-8,46-64, graphview.py:3-11, agentprune_lg.py, agentdropout_lg.py} · "
                         "src/lychee_mas/methods/prerun/ · docs/DESIGN.md:80-87,122-125",
        "notes": "回到第三页那五行代码，这就是运行前接缝：执行前改图。MASPO 只改两个节点的提示词，AgentPrune 训边权再剪边，"
                 "AgentDropout 每轮淘汰节点换拓扑。optimize 是离线重活、产物落 JSON，apply 加载产物即插即用。"
                 "GEPA 算法已经在库里但还不是图进图出，给它接上就是一个完整的 PR；想做新方法，在 methods/prerun 注册一个类即可。",
    },
]


# ----------------------------------------------------------------------------- drawing helpers
def _rgb(hexstr: str):
    from pptx.dml.color import RGBColor

    return RGBColor.from_string(hexstr)


def _font(run, size: float, *, bold=False, color=INK, name=FONT, italic=False):
    from pptx.util import Pt

    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = name
    run.font.color.rgb = _rgb(color)


def add_text(slide, x, y, w, h, text, size, *, bold=False, color=INK, name=FONT, align="left",
             anchor="top", italic=False):
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches

    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
    p = tf.paragraphs[0]
    p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
    r = p.add_run()
    r.text = text
    _font(r, size, bold=bold, color=color, name=name, italic=italic)
    return tb


def draw_box(slide, x, y, w, h, *, fill=None, line=LINE, line_w=1.0, text="", size=14, bold=False,
             color=INK, name=FONT, shape="rect", align="center"):
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches, Pt

    kind = {"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE,
            "chevron": MSO_SHAPE.CHEVRON, "pent": MSO_SHAPE.PENTAGON}[shape]
    sp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == "round":
        sp.adjustments[0] = 0.12
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = _rgb(fill)
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = _rgb(line)
        sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    tf = sp.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER}[align]
    if text:
        r = p.add_run()
        r.text = text
        _font(r, size, bold=bold, color=color, name=name)
    return sp


def draw_arrow(slide, x, y, w, h, *, fill=GREY):
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches

    sp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = _rgb(fill)
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def add_bullets(slide, x, y, w, h, items: list[str], size=17):
    from pptx.enum.text import MSO_ANCHOR
    from pptx.util import Inches, Pt

    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        p.line_spacing = 1.15
        mark = p.add_run()
        mark.text = "▪ "
        _font(mark, size, color=ACCENT, bold=True)
        # inline `code` spans rendered in mono
        parts = item.split("`")
        for j, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            if j % 2 == 1:
                _font(r, size - 2, name=MONO, color="2a2d34")
            else:
                _font(r, size)
    return tb


def add_code(slide, x, y, w, h, lines: list[str], caption: str, size=11.5):
    from pptx.enum.text import MSO_ANCHOR
    from pptx.util import Inches, Pt

    draw_box(slide, x, y, w, h, fill=CODE_BG, line=None)
    draw_box(slide, x, y, 0.06, h, fill=ACCENT, line=None)
    tb = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.08), Inches(w - 0.25), Inches(h - 0.4))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(1)
        r = p.add_run()
        r.text = ln if ln else " "
        is_comment = ln.lstrip().startswith("#")
        _font(r, size, name=MONO, color="8a8f98" if is_comment else "2a2d34")
    add_text(slide, x + 0.15, y + h - 0.32, w - 0.25, 0.28, caption, 9.5, color=MUTED, name=MONO)


def add_table(slide, x, y, w, h, rows: list[list[str]], size=12.5, first_col_bold=True):
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches

    n_r, n_c = len(rows), len(rows[0])
    gt = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(h)).table
    first_w = 0.34 if n_c > 2 else 0.5
    gt.columns[0].width = Inches(w * first_w)
    for c in range(1, n_c):
        gt.columns[c].width = Inches(w * (1 - first_w) / (n_c - 1))
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = gt.cell(r_i, c_i)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _rgb(ACCENT if r_i == 0 else ("ffffff" if r_i % 2 else PAPER))
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            mono = c_i > 0 and any(ch.isdigit() for ch in val) and not any(ord(ch) > 0x2E80 for ch in val)
            _font(r, size, bold=(r_i == 0 or (first_col_bold and c_i == 0)),
                  color="ffffff" if r_i == 0 else INK, name=MONO if mono else FONT)
    return gt


def draw_two_node(slide, x, y, w):
    """predictor_0 -> reflector_0 with the question entering from the left."""
    node_w, node_h = 2.0, 0.95
    yy = y
    draw_box(slide, x, yy + 0.2, 1.15, 0.55, fill=PAPER, line=GREY, text="题目", size=13, color=MUTED)
    draw_arrow(slide, x + 1.2, yy + 0.35, 0.45, 0.25)
    draw_box(slide, x + 1.7, yy, node_w, node_h, fill="ffffff", line=ACCENT, line_w=1.75, shape="round",
             text="predictor_0\n答题者", size=13, bold=True, name=MONO, color=INK)
    draw_arrow(slide, x + 1.7 + node_w + 0.08, yy + 0.35, 0.6, 0.25, fill=ACCENT)
    draw_box(slide, x + 1.7 + node_w + 0.75, yy, node_w, node_h, fill="ffffff", line=ACCENT, line_w=1.75,
             shape="round", text="reflector_0\n检查员", size=13, bold=True, name=MONO, color=INK)
    draw_arrow(slide, x + 1.7 + 2 * node_w + 0.85, yy + 0.35, 0.45, 0.25)
    draw_box(slide, x + 1.7 + 2 * node_w + 1.35, yy + 0.2, 1.15, 0.55, fill=PAPER, line=GREY,
             text="最终答案", size=13, color=MUTED)
    # letter labels
    add_text(slide, x + 1.7, yy + node_h + 0.05, node_w, 0.3, "V：节点 = 一个提示词", 11, color=MUTED, align="center")
    add_text(slide, x + 1.7 + node_w - 0.2, yy - 0.34, 1.3, 0.3, "E / W：边、权重", 11, color=ACCENT, align="center")
    add_text(slide, x + 1.7 + node_w + 0.75, yy + node_h + 0.05, node_w, 0.3, "T：多轮 = 再走一遍", 11,
             color=MUTED, align="center")
    add_text(slide, x + 1.7 + 2 * node_w + 1.35, yy + node_h + 0.05, 1.15, 0.3, "M：记忆", 11, color=MUTED,
             align="center")


SEAMS = [("build", "建图"), ("prerun", "运行前优化"), ("memory", "记忆中枢"),
         ("compile", "编译"), ("processing", "运行/归约"), ("postrun", "运行后")]


def draw_seams(slide, x, y, w, active: set[str] | None = None, h=0.62):
    active = active or set()
    gap = 0.05
    cw = (w - gap * (len(SEAMS) - 1)) / len(SEAMS)
    for i, (key, zh) in enumerate(SEAMS):
        on = key in active
        draw_box(slide, x + i * (cw + gap), y, cw, h, fill=ACCENT if on else "ffffff",
                 line=ACCENT if on else GREY, line_w=1.25, shape="chevron",
                 text=f"{key}\n{zh}", size=10.5, bold=on, color="ffffff" if on else INK,
                 name=FONT)


def add_frame(slide, spec: dict, page: int, total: int):
    """Title bar, kicker, footer shared by every content slide."""
    from pptx.util import Inches

    draw_box(slide, 0, 0, SLIDE_W, SLIDE_H, fill=PAPER, line=None)
    draw_box(slide, MARGIN, 0.55, 0.09, 0.75, fill=ACCENT, line=None)
    add_text(slide, MARGIN + 0.22, 0.45, SLIDE_W - 2 * MARGIN - 0.3, 0.9, spec["title"], 30, bold=True)
    add_text(slide, MARGIN + 0.22, 1.28, SLIDE_W - 2 * MARGIN - 0.3, 0.45, spec["kicker"], 15, color=MUTED)
    # footer
    ln = slide.shapes.add_connector(1, Inches(MARGIN), Inches(SLIDE_H - 0.55), Inches(SLIDE_W - MARGIN),
                                    Inches(SLIDE_H - 0.55))
    ln.line.color.rgb = _rgb(LINE)
    add_text(slide, MARGIN, SLIDE_H - 0.5, SLIDE_W - 2 * MARGIN - 1.2, 0.35, spec.get("source_footer", ""), 9,
             color=GREY, name=MONO)
    add_text(slide, SLIDE_W - MARGIN - 1.2, SLIDE_H - 0.5, 1.2, 0.35, f"{page} / {total}", 9, color=GREY,
             name=MONO, align="right")


def set_notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text


# ----------------------------------------------------------------------------- slide builders
def build_cover(prs, spec, page, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    draw_box(slide, 0, 0, SLIDE_W, SLIDE_H, fill=PAPER, line=None)
    draw_box(slide, 0, 0, 0.35, SLIDE_H, fill=ACCENT, line=None)
    logo = REPO / "images" / "logo.png"
    main = REPO / "docs" / "assets" / "main.png"
    from pptx.util import Inches

    if logo.exists():
        slide.shapes.add_picture(str(logo), Inches(0.9), Inches(0.7), height=Inches(1.0))
    add_text(slide, 0.9, 2.3, 6.4, 1.1, spec["title"], 44, bold=True)
    add_text(slide, 0.9, 3.45, 6.4, 0.9, spec["kicker"], 20, color=INK)
    draw_box(slide, 0.9, 4.55, 1.2, 0.05, fill=ACCENT, line=None)
    for i, b in enumerate(spec.get("bullets", [])):
        add_text(slide, 0.9, 4.85 + i * 0.45, 6.4, 0.42, b, 14, color=MUTED)
    if main.exists():
        # 1280x765 px -> fit a 5.4in-wide box, vertically centred in the right half
        pic = slide.shapes.add_picture(str(main), Inches(7.5), Inches(1.0), width=Inches(5.4))
        pic.top = int((Inches(SLIDE_H) - pic.height) / 2)
    add_text(slide, 0.9, SLIDE_H - 0.6, 8, 0.35, spec.get("source_footer", ""), 9, color=GREY, name=MONO)
    set_notes(slide, spec["notes"])
    return slide


def build_content(prs, spec, page, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_frame(slide, spec, page, total)
    fig = spec.get("figure", "none")
    body_y = 1.9
    body_h = SLIDE_H - body_y - 0.75
    full_w = SLIDE_W - 2 * MARGIN
    left_w = 6.2
    right_x = MARGIN + left_w + 0.35
    right_w = SLIDE_W - MARGIN - right_x

    if fig == "two_node":
        draw_two_node(slide, MARGIN + 0.9, body_y + 0.45, full_w)
        add_bullets(slide, MARGIN, body_y + 2.0, full_w, body_h - 2.0, spec["bullets"], size=17)
    elif fig == "seams":
        draw_seams(slide, MARGIN, body_y, full_w, active=set(spec.get("active", [])))
        top = body_y + 0.85
        add_bullets(slide, MARGIN, top, left_w, body_h - 0.85, spec["bullets"], size=16)
        if spec.get("code"):
            c = spec["code"]
            lines = read_lines(c["path"], c["start"], c["end"])
            add_code(slide, right_x, top, right_w, min(0.28 * len(lines) + 0.55, body_h - 0.85), lines,
                     f'{c["path"]}:{c["start"]}-{c["end"]}  {c.get("caption", "")}')
    elif fig == "table":
        rows = spec["table"]
        add_bullets(slide, MARGIN, body_y, left_w, body_h, spec["bullets"], size=16)
        add_table(slide, right_x, body_y + 0.05, right_w, 0.42 * len(rows), rows)
        if spec.get("table_note"):
            add_text(slide, right_x, body_y + 0.1 + 0.42 * len(rows) + 0.05, right_w, 0.8, spec["table_note"],
                     10.5, color=MUTED)
    elif fig == "seams+table":
        draw_seams(slide, MARGIN, body_y, full_w, active=set(spec.get("active", [])))
        top = body_y + 0.85
        rows = spec["table"]
        add_bullets(slide, MARGIN, top, left_w, body_h - 0.85, spec["bullets"], size=16)
        add_table(slide, right_x, top + 0.05, right_w, 0.42 * len(rows), rows, size=12)
        if spec.get("table_note"):
            add_text(slide, right_x, top + 0.1 + 0.42 * len(rows) + 0.05, right_w, 0.9, spec["table_note"],
                     10.5, color=MUTED)
    else:
        add_bullets(slide, MARGIN, body_y, full_w, body_h, spec["bullets"], size=17)
    set_notes(slide, spec["notes"])
    return slide


def build(out: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    total = len(SLIDES)
    for i, spec in enumerate(SLIDES, start=1):
        (build_cover if spec.get("figure") == "cover" else build_content)(prs, spec, i, total)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


# ----------------------------------------------------------------------------- checks
def check() -> list[str]:
    problems: list[str] = []
    for s in SLIDES:
        n = s["n"]
        if cjk_len(s["title"]) > MAX_TITLE:
            problems.append(f"p{n} title>{MAX_TITLE}: {s['title']!r} ({cjk_len(s['title'])})")
        if cjk_len(s["kicker"]) > MAX_KICKER:
            problems.append(f"p{n} kicker>{MAX_KICKER}: {s['kicker']!r}")
        bl = s.get("bullets", [])
        if len(bl) > MAX_BULLETS:
            problems.append(f"p{n} bullets>{MAX_BULLETS}: {len(bl)}")
        for b in bl:
            if cjk_len(b.replace("`", "")) > MAX_BULLET:
                problems.append(f"p{n} bullet>{MAX_BULLET}: {b!r} ({cjk_len(b)})")
        if s.get("code"):
            c = s["code"]
            lines = read_lines(c["path"], c["start"], c["end"])
            if len(lines) > MAX_CODE_LINES:
                problems.append(f"p{n} code>{MAX_CODE_LINES} lines")
            if any(len(ln) > MAX_CODE_COLS for ln in lines):
                problems.append(f"p{n} code>{MAX_CODE_COLS} cols")
        if s.get("table"):
            r, c = len(s["table"]), max(len(row) for row in s["table"])
            if r > MAX_TABLE[0] or c > MAX_TABLE[1]:
                problems.append(f"p{n} table {r}x{c} exceeds {MAX_TABLE}")
        if not s.get("notes"):
            problems.append(f"p{n} notes empty")
        if "solver" in json.dumps(s, ensure_ascii=False):
            problems.append(f"p{n} contains 'solver'")
    if [s["n"] for s in SLIDES] != list(range(1, len(SLIDES) + 1)):
        problems.append("page numbering not contiguous")
    for img in ("images/logo.png", "docs/assets/main.png"):
        if not (REPO / img).exists():
            problems.append(f"missing image {img}")
    problems.extend(f"metrics fallback: {w}" for w in EXP_WARNINGS)
    return problems


def dump_text() -> str:
    out = []
    for s in SLIDES:
        out.append(f"=== p{s['n']} {s['title']}")
        out.append(f"kicker: {s['kicker']}")
        for b in s.get("bullets", []):
            out.append(f"  - {b}")
        if s.get("code"):
            c = s["code"]
            out.append(f"  code {c['path']}:{c['start']}-{c['end']}")
            out.extend("    | " + ln for ln in read_lines(c["path"], c["start"], c["end"]))
        for row in s.get("table", []) or []:
            out.append("  | " + " | ".join(row))
        if s.get("table_note"):
            out.append(f"  note: {s['table_note']}")
        out.append(f"footer: {s.get('source_footer', '')}")
        out.append(f"notes: {s['notes']}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/slides/lycheemas_undergrad_intro.pptx")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dump-text", action="store_true")
    a = ap.parse_args(argv)
    if a.dump_text:
        print(dump_text())
        return 0
    if a.check:
        probs = check()
        print("\n".join(probs) if probs else f"OK: {len(SLIDES)} slides within limits")
        return 1 if probs else 0
    probs = check()
    if probs:
        print("\n".join(probs), file=sys.stderr)
        return 1
    out = build(REPO / a.out if not Path(a.out).is_absolute() else Path(a.out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
