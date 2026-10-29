# MATH Level 5 Hardness Probe (2026-10-24)

## Objective

Probe whether MATH Level 5 problems from
`/home/med_code/nhzk/yjw/MAS-Research/model/Math` are difficult enough to use as
hard training/evaluation samples for Qwen3-8B.

## Sample

- Source: MATH test split, Level 5 only.
- Size: 21 cases.
- Sampling: three cases per subject, deterministic random seed 20261024.
- Subjects: algebra, counting and probability, geometry, intermediate algebra,
  number theory, prealgebra, precalculus.
- Case ID file:
  `runs/math_hard_level5_case_ids.json`

## Qwen3-8B Single Baseline

```text
task=math cases=21 team=single
accuracy=0.8095 (17/21)
```

Artifacts:

- `runs/benchmarks/Qwen3-8B/langgraph_baseline_single/math/outputs.jsonl`
- `runs/benchmarks/Qwen3-8B/langgraph_baseline_single/math/metrics.json`
- `runs/benchmarks/Qwen3-8B/langgraph_baseline_single/math/case_progress.jsonl`

## Conclusion

The sample is harder than GSM8K, where Qwen3-8B single reaches 91%, but at
80.95% it still leaves only four failures in 21 cases. That is not a strong
hard-sample set for training or for detecting mechanism effects. A larger
Level 5 sample or a competition-level subset is likely needed.
