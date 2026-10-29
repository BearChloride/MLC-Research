"""两次运行的**配对对照**：逐题对齐 + McNemar 精确检验。

为什么需要它：当评分口径不能改、单例误判又无法完全消除时，"哪边多对 1 题"这种比较
本身就不足以下结论。本脚本把两次运行按 ``case_id`` 配对（同一种子抽样的两次运行
case_id 可一一对应），给出：

* 配对的四格表（双方都对/一方对/双方都错）；
* **McNemar 精确检验** p 值——判断"对错翻转"是否超出随机波动；
* accuracy 与 token / latency 的并排对照（取自各 run 的 metrics.json）。

只读，不修改任何产物。

用法::

    python scripts/compare_runs.py <run_dir_A> <run_dir_B> [--label-a single --label-b chain3]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))


class _NotApplicable(Exception):
    """gold 抽不出数字（如 ASDiv 的 Yes/No），严格口径对该样本不适用。"""


def _load_run(run_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_path = os.path.join(run_dir, "outputs.jsonl")
    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"缺少 outputs.jsonl：{out_path}")
    rows: List[Dict[str, Any]] = []
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
    else:
        metrics = {}
    return rows, metrics


def mcnemar_exact(n01: int, n10: int) -> float:
    """McNemar 精确检验（双侧，二项分布 p=0.5）。n01/n10 是方向相反的翻转数。"""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_a", help="运行 A 的结果目录")
    parser.add_argument("run_b", help="运行 B 的结果目录")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--scorer", choices=("official", "strict"), default="official",
                        help="official=框架 score_exact 口径；strict=审计用的严格数值口径")
    args = parser.parse_args()

    from lychee_mas.eval.audit import strict_correct  # noqa: E402

    rows_a, met_a = _load_run(args.run_a)
    rows_b, met_b = _load_run(args.run_b)
    label_a = args.label_a or os.path.basename(os.path.dirname(args.run_a))
    label_b = args.label_b or os.path.basename(os.path.dirname(args.run_b))

    def _key(row: Dict[str, Any], idx: int) -> str:
        return str(row.get("case_id", idx))

    map_a = {_key(r, i): r for i, r in enumerate(rows_a)}
    map_b = {_key(r, i): r for i, r in enumerate(rows_b)}
    common = sorted(set(map_a) & set(map_b), key=lambda x: int(x) if x.isdigit() else x)
    if not common:
        raise SystemExit("[compare] 两次运行没有可配对的 case_id")

    n11 = n10 = n01 = n00 = 0
    skipped = 0

    def _ok(row: Dict[str, Any]) -> bool:
        if args.scorer == "official":
            return float(row.get("score") or 0.0) == 1.0
        value = strict_correct(row.get("prediction", ""), row.get("gold"))
        if value is None:
            raise _NotApplicable
        return value

    for cid in common:
        try:
            ok_a, ok_b = _ok(map_a[cid]), _ok(map_b[cid])
        except _NotApplicable:
            skipped += 1
            continue
        if ok_a and ok_b:
            n11 += 1
        elif ok_a and not ok_b:
            n10 += 1
        elif ok_b and not ok_a:
            n01 += 1
        else:
            n00 += 1

    total = n11 + n10 + n01 + n00
    acc_a = n11 + n10
    acc_b = n11 + n01
    p_value = mcnemar_exact(n01, n10)

    print(f"[compare] scorer = {args.scorer}")
    print(f"[compare] A = {label_a}   ({acc_a}/{total} = {acc_a / total:.4f})")
    print(f"[compare] B = {label_b}   ({acc_b}/{total} = {acc_b / total:.4f})")
    print(f"[compare] 配对样本数 = {total}"
          f"（仅 A 有 {len(rows_a) - len(common)} 条，仅 B 有 {len(rows_b) - len(common)} 条"
          + (f"，口径不适用 {skipped} 条" if skipped else "") + "）")
    print()
    print("        四格表（行=A，列=B）")
    print("          B 对     B 错")
    print(f"  A 对    {n11:>5}    {n10:>5}")
    print(f"  A 错    {n01:>5}    {n00:>5}")
    print()
    print(f"[compare] 翻转：A 独对 {n10} 题，B 独对 {n01} 题")
    print(f"[compare] McNemar 精确检验 p = {p_value:.4f} "
          f"-> {'差异显著 (p<0.05)' if p_value < 0.05 else '差异不显著 (p>=0.05)'}")
    print()

    def _fmt(metrics: Dict[str, Any]) -> str:
        if not metrics:
            return "（无 metrics.json）"
        return (
            f"调用次数/题={metrics.get('mean_model_calls_per_case')}, "
            f"输入tok/题={metrics.get('mean_input_total_positions_per_case')}, "
            f"输出tok/题={metrics.get('mean_output_text_tokens_per_case')}, "
            f"延迟/题={metrics.get('mean_model_latency_s_per_case')}"
        )

    print(f"[cost] {label_a}: {_fmt(met_a)}")
    print(f"[cost] {label_b}: {_fmt(met_b)}")


if __name__ == "__main__":
    main()
