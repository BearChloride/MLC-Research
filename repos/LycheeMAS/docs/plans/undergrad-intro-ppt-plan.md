# 本科生小介绍 PPT · 计划（6 页定稿）

> 目标：给会 Python、没接触过 agent 的本科生做一个 5 分钟小介绍。
> 产物：`runs/slides/lycheemas_undergrad_intro.pptx`（gitignored）+ `runs/slides/speaker_notes.md`；
> 生成脚本 `docs/slides/build_undergrad_intro.py`（`--check` 校验版式限制，`--dump-text` 打印全文）。

## 页面结构

| 页 | 标题 | 图 | 讲什么 |
| --- | --- | --- | --- |
| 1 | LycheeMAS | 封面（logo + `docs/assets/main.png`） | 一句话：把 MAS 当成一张图 G=(V,E,W,T,M) |
| 2 | 一道题，加一个检查员 | 两节点图 predictor_0 → reflector_0 | MATH-500 第 0 题引入最小 MAS；五个字母各管什么；提示词挂在节点 V 上 |
| 3 | 框架总览：五接缝、三层、注册表 | 五接缝色带 + `plugins/prerun/base.py:60-64` | 每个字母一个接缝；plugins 薄 / methods 厚 / backends 碰模型库；换算法 = 换 method 名 |
| 4 | 真实实验：MASPO × MATH-500 | 指标表（读 `runs/maspo/benchmarks_v2/.../metrics.json`） | 只改提示词 0.78→0.85；两个教训（打分口径、ckpt 落盘）；优化代价 |
| 5 | 记忆中枢：给图挂上 M | 五接缝色带（高亮 memory）+ 组件表 | 接缝已立、入口仍是桩（NotImplementedError，P3）；cdm / static / nl / latent 现状；招新点 |
| 6 | 运行前优化：改 V/W/E/T | 五接缝色带（高亮 prerun）+ 组件表 | MASPO / AgentPrune / AgentDropout 各改哪个字母；optimize 离线产物化 + apply 即插即用；GEPA 待接图 |

## 版式规则（脚本 `--check` 强制）

- 标题 ≤16 字、副标 ≤30 字、每页 ≤5 条要点、每条 ≤34 字（英文/符号按半字计）。
- 代码 ≤8 行 ≤80 列，全部按 `路径:行号` 从仓库逐字读取；数字全部从 `runs/` 指标文件读取，缺文件时回退到脚本内记录的字面值并在 `--check` 里报警。
- 表格 ≤5×5；每页有讲稿备注；页脚标注来源路径。

## 事实来源

`docs/DESIGN.md` · `docs/ONBOARDING.md` · `docs/example-maspo.md:134-144` · `src/lychee_mas/plugins/{prerun,memory}/base.py` ·
`scripts/run_maspo_langgraph.py:152-170` · `runs/maspo/benchmarks_v2/Qwen3-8B{,_prompts-v1}/*/math500/{metrics.json,outputs.jsonl}` · `runs/maspo/v2_optimize_eval.log:28-30`。

## 重建

```bash
conda activate CDM   # 需 python-pptx（已装）
python docs/slides/build_undergrad_intro.py --check
python docs/slides/build_undergrad_intro.py --out runs/slides/lycheemas_undergrad_intro.pptx
```
