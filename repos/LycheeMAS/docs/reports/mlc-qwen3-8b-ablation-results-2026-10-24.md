# Qwen3-8B MLC Ablation Results (2026-10-24)

## Status

All nine ablation runs completed with exit code 0. For each ablation variant and
seed, the run contains the same 100 evaluation case IDs as the corresponding
full-MLC seed, allowing paired comparison on identical cases.

Protocol is the same as the full runs: 32 training cases, 100 evaluation cases,
two collaboration rounds, 1024-token generation cap, bfloat16, L20 `cuda:0`,
and activation checkpointing.

## Accuracy Summary

Official exact accuracy:

| Method | Seed 101 | Seed 202 | Seed 303 | Mean |
|---|---:|---:|---:|---:|
| full | 97.0% | 96.0% | 97.0% | 96.7% |
| no role embedding | 98.0% | 98.0% | 98.0% | 98.0% |
| no mixing | 96.0% | 96.0% | 96.0% | 96.0% |
| no `L_dis` | 97.0% | 97.0% | 94.0% | 96.0% |

Strict numeric accuracy:

| Method | Seed 101 | Seed 202 | Seed 303 | Mean |
|---|---:|---:|---:|---:|
| full | 96.0% | 95.0% | 96.0% | 95.7% |
| no role embedding | 97.0% | 97.0% | 97.0% | 97.0% |
| no mixing | 95.0% | 95.0% | 95.0% | 95.0% |
| no `L_dis` | 96.0% | 96.0% | 93.0% | 95.0% |

## Paired McNemar Versus Full

Each ablation seed is paired with the full-MLC run of the same training seed.
All comparisons use 100 common case IDs.

| Variant | Seed | Official p | Strict p | Ablation-only / full-only (official) |
|---|---:|---:|---:|---:|
| no role embedding | 101 | 1.000 | 1.000 | 1 / 0 |
| no role embedding | 202 | 0.500 | 0.500 | 2 / 0 |
| no role embedding | 303 | 1.000 | 1.000 | 1 / 0 |
| no mixing | 101 | 1.000 | 1.000 | 0 / 1 |
| no mixing | 202 | 1.000 | 1.000 | 1 / 1 |
| no mixing | 303 | 1.000 | 1.000 | 0 / 1 |
| no `L_dis` | 101 | 1.000 | 1.000 | 0 / 0 |
| no `L_dis` | 202 | 1.000 | 1.000 | 1 / 0 |
| no `L_dis` | 303 | 0.250 | 0.250 | 0 / 3 |

No ablation differs significantly from full under the paired exact McNemar test.
The numerically higher `no role embedding` results and the lower `no L_dis`
seed 303 result are within sampling uncertainty and are not interpreted as
mechanism effects.

## Interpretation

These ablations are diagnostic only. They do not establish that role embedding,
mixing, or the latent differentiation loss is necessary or beneficial. The 8B
full and ablation runs all remain above the 8B baselines on this 100-case set,
but the ablation-versus-full differences are not significant at this sample
size.

## Artifacts

- Full runs: `runs/benchmarks/Qwen3-8B/mlc_aligned1024_seed{101,202,303}/gsm8k/`
- Ablation runs:
  `runs/benchmarks/Qwen3-8B/mlc_aligned1024_{no_role_embedding,no_mixing,no_l_dis}_seed{101,202,303}/gsm8k/`
- Logs: `runs/benchmarks/Qwen3-8B/mlc_ablation_logs/`
- Full-run report:
  `docs/reports/mlc-qwen3-8b-3seed-results-2026-10-23.md`
