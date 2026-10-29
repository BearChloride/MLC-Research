"""把 MLC（ACL 2025, 2025.acl-long.1105）复现所需的本地数学数据集规范化进 prepared 缓存。

这 6 个数据集（AddSub / SingleEQ / MultiArith / ASDiv / SVAMP / MATH）**不联网**：
本脚本只做「本地原始文件 -> prepared 缓存」的规范化转换，产出的 jsonl 字段统一为
``{question, gold}``，之后 ``benchmarks.load("<task>")`` 直接读缓存即可。

用法（本机示例）::

    python scripts/prepare_mlc_math.py --data-root E:/MAS-Research/model
    python scripts/prepare_mlc_math.py --tasks addsub,svamp --force
    python scripts/prepare_mlc_math.py --list

原始数据目录布局（--data-root 指向的目录）::

    mawps/addsub.jsonl
    mawps/singleeq.jsonl
    mawps/multiarith.jsonl
    ASDiv_ASDiv.xml/dataset/ASDiv.xml
    SVAMP_SVAMP.json/SVAMP.json
    Math/<7 个学科>/{train,test}-*.parquet

prepared 根目录由 ``LYCHEE_BENCHMARK_PREPARED_ROOT``（或 ``CDM_DATA_ROOT``）决定，
默认是仓库内 ``data/benchmarks/prepared``（已 gitignore，不进版本库）。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# MLC 论文 7 个数据集里，除 GSM8K（另有 loader）之外的 6 个
MLC_TASKS = ("addsub", "singleeq", "multiarith", "asdiv", "svamp", "math")


def _parse_tasks(raw: List[str] | None) -> List[str]:
    if not raw:
        return list(MLC_TASKS)
    tasks: List[str] = []
    for item in raw:
        tasks.extend(part.strip() for part in item.split(",") if part.strip())
    unknown = [task for task in tasks if task not in MLC_TASKS]
    if unknown:
        raise SystemExit(f"未知数据集 {unknown}；可选：{', '.join(MLC_TASKS)}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default=os.environ.get("LYCHEE_MLC_DATA_ROOT"),
                        help="原始数据集总目录（等价于环境变量 LYCHEE_MLC_DATA_ROOT）")
    parser.add_argument("--prepared-root", default=os.environ.get("LYCHEE_BENCHMARK_PREPARED_ROOT"),
                        help="规范化缓存输出目录（等价于 LYCHEE_BENCHMARK_PREPARED_ROOT）")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help=f"要准备的数据集（逗号或空格分隔），默认全部：{','.join(MLC_TASKS)}")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，强制重新转换")
    parser.add_argument("--list", action="store_true", help="只列出数据集与目标文件，不做转换")
    args = parser.parse_args()

    if args.data_root:
        os.environ["LYCHEE_MLC_DATA_ROOT"] = args.data_root
    if args.prepared_root:
        os.environ["LYCHEE_BENCHMARK_PREPARED_ROOT"] = args.prepared_root

    from lychee_mas.eval.benchmarks import prepare
    from lychee_mas.eval.benchmarks.common import prepared_root

    tasks = _parse_tasks(args.tasks)
    if args.list:
        print(f"[mlc-math] prepared_root={prepared_root()}")
        for task in tasks:
            print(f"  {task}")
        return

    if not os.environ.get("LYCHEE_MLC_DATA_ROOT"):
        raise SystemExit(
            "需要 --data-root 或环境变量 LYCHEE_MLC_DATA_ROOT（指向存放原始数据集的目录）"
        )

    print(f"[mlc-math] data_root={os.environ['LYCHEE_MLC_DATA_ROOT']}")
    print(f"[mlc-math] prepared_root={prepared_root()}")
    for task in tasks:
        print(f"[mlc-math] prepare {task} ...", flush=True)
        try:
            location = prepare(task, force=args.force)
        except Exception as exc:  # 显式失败，不静默跳过
            raise SystemExit(f"[mlc-math] {task} 失败：{exc}") from exc
        print(f"[mlc-math] {task} ready at {location}", flush=True)


if __name__ == "__main__":
    main()
