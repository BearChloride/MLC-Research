"""MLC 数学数据集的数据落位自检：把 gold 当预测喂进评分器，逐数据集断言准确率 = 100%。

这是「数据落位」阶段性价比最高的一步：一旦 gold 解析或评分口径有错（例如数值 gold
留成 ``"43.0"``、ASDiv 单位没剥离、MATH 的 ``\\boxed{}`` 没取到），这里会立刻暴露，
不必等到跑完模型才发现。

用法::

    python scripts/selfcheck_mlc_math.py                 # 全部 6 个数据集，全量
    python scripts/selfcheck_mlc_math.py --n 100         # 每个数据集抽前 100 题
    python scripts/selfcheck_mlc_math.py --tasks gsm8k,math

退出码非 0 表示有数据集没通过，可直接接进 CI / 提交前检查。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# GSM8K 一并检查：它是 MLC 7 个数据集里唯一由既有 loader 承载的
DEFAULT_TASKS = ("gsm8k", "addsub", "singleeq", "multiarith", "asdiv", "svamp", "math")


def _parse_tasks(raw: List[str] | None) -> List[str]:
    if not raw:
        return list(DEFAULT_TASKS)
    tasks: List[str] = []
    for item in raw:
        tasks.extend(part.strip() for part in item.split(",") if part.strip())
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help=f"要自检的数据集，默认：{','.join(DEFAULT_TASKS)}")
    parser.add_argument("--n", type=int, default=0, help="每个数据集检查前 n 题（0=全量）")
    args = parser.parse_args()

    from lychee_mas.eval import benchmarks  # noqa: F401  触发注册
    from lychee_mas.eval.metrics import score

    tasks = _parse_tasks(args.tasks)
    failures: List[str] = []
    for task in tasks:
        records = benchmarks.load(task, n=args.n or None)
        correct = 0
        bad: List[str] = []
        for record in records:
            value = score(record["kind"], str(record["gold"]), record["gold"])
            correct += value
            if value != 1.0 and len(bad) < 3:
                bad.append(f"gold={record['gold']!r}")
        accuracy = correct / len(records) if records else 0.0
        status = "OK " if accuracy == 1.0 else "FAIL"
        print(
            f"[selfcheck] {status} {task:<11} kind={records[0]['kind']:<6} "
            f"n={len(records):<5} gold-as-pred accuracy={accuracy:.4f}"
            + (f"  反例: {'; '.join(bad)}" if bad else ""),
            flush=True,
        )
        if accuracy != 1.0:
            failures.append(task)

    if failures:
        raise SystemExit(f"[selfcheck] 未通过：{', '.join(failures)}")
    print("[selfcheck] 全部通过：gold 解析与评分口径一致")


if __name__ == "__main__":
    main()
