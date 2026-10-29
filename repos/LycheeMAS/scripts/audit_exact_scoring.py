"""数值型 ``exact`` 评分的**审计口径**：不改动官方打分器，只做离线复算与告警。

背景：``eval/metrics.py::score_exact`` 在两边最后一个数字不相等时，会退化成
「规整字符串包含」判断。这会让"错误答案恰好包含正确答案的数字子串"被误判为对
（实例：gold=64、预测 ``\\boxed{10.648}``，`64` 是 `10648` 的子串）。

评分口径不能改（要与团队/历史结果可比），所以本脚本提供**第二口径**：

* **官方口径**：``outputs.jsonl`` 里已算好的 ``score``；
* **严格数值口径**：抽取双方最后一个数字，按**数值**比较（43 与 43.0 视为相等）。

并列出两个方向的差异样本：

* 疑似误判（虚高）：官方判对、严格判错；
* 疑似漏判（虚低）：官方判错、严格判对。

两个数都报，结论以哪个为准由实验方决定；脚本本身不修改任何产物。

用法::

    python scripts/audit_exact_scoring.py runs/.../langgraph_baseline_single/gsm8k
    python scripts/audit_exact_scoring.py <dir1> <dir2> --json audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lychee_mas.eval.audit import strict_numeric_score  # noqa: E402


def audit_run(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "outputs.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少 outputs.jsonl：{path}")

    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    official_correct = 0
    strict_correct = 0
    applicable = 0
    false_positive: List[Dict[str, Any]] = []
    false_negative: List[Dict[str, Any]] = []

    for row in rows:
        if row.get("kind") != "exact":
            continue
        official = float(row.get("score") or 0.0)
        official_correct += int(official == 1.0)
        strict = strict_numeric_score(row.get("prediction", ""), row.get("gold"))
        if strict is None:
            continue
        applicable += 1
        strict_correct += int(strict == 1.0)
        if official == 1.0 and strict == 0.0:
            false_positive.append(
                {"case_id": row.get("case_id"), "gold": row.get("gold"),
                 "prediction": str(row.get("prediction"))[:120]}
            )
        elif official == 0.0 and strict == 1.0:
            false_negative.append(
                {"case_id": row.get("case_id"), "gold": row.get("gold"),
                 "prediction": str(row.get("prediction"))[:120]}
            )

    return {
        "run_dir": run_dir,
        "num_cases": len(rows),
        "num_exact_cases": applicable,
        "official_correct": official_correct,
        "official_accuracy": (official_correct / applicable) if applicable else None,
        "strict_correct": strict_correct,
        "strict_accuracy": (strict_correct / applicable) if applicable else None,
        "suspect_false_positive": false_positive,
        "suspect_false_negative": false_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", help="包含 outputs.jsonl 的结果目录")
    parser.add_argument("--json", default=None, help="把审计结果另存为 JSON")
    args = parser.parse_args()

    reports = []
    for run_dir in args.run_dirs:
        try:
            report = audit_run(run_dir)
        except Exception as exc:
            raise SystemExit(f"[audit] {run_dir} 失败：{exc}") from exc
        reports.append(report)
        label = os.path.basename(os.path.dirname(run_dir))
        print(f"[audit] {label}")
        print(f"  exact 样本数      : {report['num_exact_cases']}")
        print(f"  官方口径正确数/率 : {report['official_correct']} / "
              f"{report['official_accuracy']:.4f}")
        print(f"  严格数值正确数/率 : {report['strict_correct']} / "
              f"{report['strict_accuracy']:.4f}")
        print(f"  疑似误判（虚高）  : {len(report['suspect_false_positive'])} 条")
        for item in report["suspect_false_positive"]:
            print(f"     case {item['case_id']}: gold={item['gold']} "
                  f"pred={item['prediction']!r}")
        print(f"  疑似漏判（虚低）  : {len(report['suspect_false_negative'])} 条")
        for item in report["suspect_false_negative"]:
            print(f"     case {item['case_id']}: gold={item['gold']} "
                  f"pred={item['prediction']!r}")
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=2)
        print(f"[audit] 已写出 {args.json}")


if __name__ == "__main__":
    main()
