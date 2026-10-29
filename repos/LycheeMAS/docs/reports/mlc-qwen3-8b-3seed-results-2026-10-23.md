# Qwen3-8B MLC Three-Seed Results (2026-10-23)

## Status

Qwen3-8B full MLC three-seed runs are complete. All three runs finished all 32
training episodes and all 100 evaluation cases without OOM after activation
checkpointing was enabled. These are same-set preliminary comparisons, not proof
that the paper mechanism has been reproduced.

## Protocol

| Item | Value |
|---|---|
| Base model | `/home/med_code/models/Qwen3-8B` |
| Device | L20, `cuda:0`, bfloat16 |
| Training | 32 GSM8K cases per seed, seeds 101/202/303 |
| Evaluation | 100 GSM8K cases matched to the 8B single baseline case IDs |
| Rounds | 2 |
| Generation cap | 1024 new tokens |
| Input truncation | disabled |
| Memory fix | `torch.utils.checkpoint` plus expandable CUDA segments |

## Qwen3-8B Results

| Method | Official exact | Strict numeric |
|---|---:|---:|
| single baseline | 91/100 (91.0%) | 91/100 (91.0%) |
| chain3 baseline | 92/100 (92.0%) | 90/100 (90.0%) |
| MLC full, seed 101 | 97/100 (97.0%) | 96/100 (96.0%) |
| MLC full, seed 202 | 96/100 (96.0%) | 95/100 (95.0%) |
| MLC full, seed 303 | 97/100 (97.0%) | 96/100 (96.0%) |
| MLC full, three-seed mean | 96.7% | 95.7% |

Paired exact McNemar p-values, MLC seed versus baseline:

| Seed | vs single official | vs single strict | vs chain3 official | vs chain3 strict |
|---:|---:|---:|---:|---:|
| 101 | 0.03125 | 0.12500 | 0.12500 | 0.07031 |
| 202 | 0.12500 | 0.28906 | 0.21875 | 0.12500 |
| 303 | 0.03125 | 0.12500 | 0.12500 | 0.07031 |

Only the seed 101/303 official comparisons against single are nominally below
0.05. No multiple-comparison correction was applied, and the strict-score
comparisons do not agree, so this is not treated as robust evidence of an
MLC effect.

## Training Check

| Seed | Episodes | Mean training team reward | Last TD loss | Last role grad | Last prefix grad |
|---:|---:|---:|---:|---:|---:|
| 101 | 32/32 | 1.000 | 0.1017 | 0.5345 | 2.9691 |
| 202 | 32/32 | 0.938 | 0.0718 | 0.2842 | 0.5323 |
| 303 | 32/32 | 0.969 | 0.1054 | 0.2285 | 0.1895 |

Every run has a final `mlc_state.pt`, `train_log.json`, `case_progress.jsonl`,
`comparison.json`, and `role_diagnostics.json`.

## Small-Model Reference

The Qwen3-0.6B aligned baselines on the same 100-case protocol were:

| Method | Official exact | Strict numeric |
|---|---:|---:|
| single baseline | 66/100 (66.0%) | 65/100 (65.0%) |
| chain3 baseline | 63/100 (63.0%) | 60/100 (60.0%) |

The earlier 0.6B MLC full runs used a different 100-case subset, three rounds,
and a 48-token generation cap, so they are not directly comparable to this 8B
protocol and are not mixed into the table above. The aligned 0.6B baselines show
that both base accuracy and the full-MLC/baseline gap are much smaller at 0.6B
scale than at 8B scale.

## Artifacts

- 8B baselines:
  `runs/benchmarks/Qwen3-8B/langgraph_baseline_single/gsm8k/`
  and `runs/benchmarks/Qwen3-8B/langgraph_baseline_chain3/gsm8k/`
- 8B full MLC runs:
  `runs/benchmarks/Qwen3-8B/mlc_aligned1024_seed101/gsm8k/`
  through `..._seed303/gsm8k/`
- Run logs:
  `runs/benchmarks/Qwen3-8B/mlc_3seed_logs/`

## Next Step

No blocking issue was found. The three ablations (`no_role_embedding`,
`no_mixing`, `no_l_dis`) are being run with the same three seeds and 8B protocol,
with per-episode atomic checkpoints and resumable per-case evaluation.

The completed ablation comparison is documented in
`docs/reports/mlc-qwen3-8b-ablation-results-2026-10-24.md`.
