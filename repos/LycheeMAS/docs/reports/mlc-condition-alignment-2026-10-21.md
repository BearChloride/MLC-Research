# MLC Condition Alignment (2026-10-21)

## Status

The seed-0 baseline rerun is complete. It reproduces the 10/15 report's headline
accuracy and token counts on the same 100 GSM8K test cases. These baseline runs
are the aligned comparison reference for the next MLC runs. No aligned MLC result
is included yet.

## Fixed Conditions

| Condition | 10/15 baseline reference | Aligned baseline run | MLC run setting | Status |
|---|---|---|---|---|
| Weights | Qwen3-0.6B; precision reported as bf16 in the baseline report | `D:\MAS-Research\models\Qwen3-0.6B`, bf16 | Same absolute path and bf16 | Matched |
| Runtime | Python 3.12/CUDA environment reported; exact patch versions absent from 10/15 report | Bundled Python 3.12.14, Torch 2.14.1+cu126, CUDA 12.6, Transformers 5.20.2, RTX 4060 Laptop GPU | Same interpreter and packages; written to each MLC `config_snapshot.json` | Aligned between new runs; historical patch-level identity unverified |
| Test cases | Seed 0, 100 sampled from pool of 1,319 | Frozen pool indices in `runs/baseline_alignment_2026-10-21/case_ids_seed0.json` | `--eval-case-ids` reads that exact file | Matched; first ten indices match the historical partial ledger |
| Generation cap | 1,024 new tokens | 1,024 baseline completed without OOM | MLC training at 1,024 OOMed during first episode's token-log-probability backprop; MLC uses 512 after the baseline 512 check showed 1/300 chain3 calls (0.33%) truncated | Near-match only; cap differs and is disclosed |
| Input cap | No explicit prompt truncation in baseline tokenizer | No explicit prompt truncation | `--max-input-tokens 0` disables explicit truncation | Matched |
| Decoding | Greedy (`do_sample=false`) | Greedy, no explicit temperature override | `sample=false`, argmax at temperature 1.0 | Matched in decoding behavior |
| Seed | 0 for test subset; greedy inference | 0 for frozen test subset; greedy inference | Training seeds 101/202/303; all use the exact fixed test IDs and greedy evaluation. Seed 101 is the predeclared primary seed; 202/303 are robustness runs | Matched test subset; training seed differs by design |
| Scoring | `M.score` official exact plus strict numeric audit | Same `M.score`, `audit_exact_scoring.py`, and `compare_runs.py` | Same `M.score`; strict results use `strict_numeric_score`; final pairwise tests use `compare_runs.py` | Matched |
| Data and preprocessing | Existing GSM8K test loader; question includes its numeric-answer instruction | Same loader and pool indices | Same loader and pool indices | Matched |
| Answer extraction | Baseline history extractor | Baseline history extractor | Per-agent extractor then majority vote, as specified in paper section 4.3 | Intentionally different method component; must be stated in comparisons |
| Evaluation topology | Single and chain3, max_rounds=2 | Same actual defaults | Three MLC agents per round, `--rounds 2` | Method-specific topology difference |
| Prompt | Native baseline system prompts | Native baseline prompts | MLC shared prompt, prior collaboration history, continuous role soft-prefix in `inputs_embeds` | Method-specific prompt difference |

## Token-Cap Smoke/Impact Check

Both caps completed on the same 100 cases without OOM. At 512 tokens, single had
0/145 truncated calls (0/100 cases); chain3 had 1/300 truncated calls (1/100
cases, 0.33% of calls). The 512-token single accuracy was 66%; chain3 was 63%.
At 1,024 tokens, both teams completed without OOM and had no length-truncated
calls. The official accuracy and output token counts matched the 10/15 report:

| Baseline | Official exact | Strict numeric | Mean input tokens/case | Mean output tokens/case | 1,024 truncation |
|---|---:|---:|---:|---:|---:|
| single | 66/100 (66%) | 65/100 (65%) | 297.4 | 241.8 | 0/145 calls |
| chain3 | 63/100 (63%) | 60/100 (60%) | 909.7 | 353.74 | 0/300 calls |

The single-vs-chain3 exact McNemar p-value is 0.6291 under official scoring and
0.3593 under strict numeric scoring. These are baseline comparisons, not MLC
effect tests.

## Interpretation Boundary

The 1,024-token seed-0 rerun reproduces the 10/15 headline accuracies and input/
output token means. The 10/15 document does not pin all runtime patch versions,
so full historical runtime identity cannot be established from that report alone.
The new baseline and MLC runs use one identical runtime. Baseline history
extraction and MLC per-agent majority voting remain intentionally different
method components; they are not silently harmonized. Accuracy differences may
therefore reflect the full MLC method (including its extraction/aggregation),
not only learned role/mixing mechanisms.

The earlier 100-case table using sample seed 404, a 48-token output cap, a
512-token input cap, and three rounds remains an **internal baseline self-
comparison, not comparable with the 10/15 project baseline report**. It is not
used in the aligned analysis.

The 1,024 MLC training attempt failed before completing an episode or saving a
checkpoint: CUDA ran out of memory during token-log-probability backpropagation.
The reduced 512 cap is an explicit resource-bounded fallback justified by the
measured low baseline truncation rate, not an exact match. The MLC 512 evaluation
will be reported as a near-aligned comparison. No case count, training sample
count, or round count is being reduced to work around memory.

The aligned MLC runs are a resource-bounded method experiment, not a claim that
the paper's effect has been reproduced. The baseline was checked first; MLC
results use the fixed conditions above and disclose the one unresolved output-cap
difference.

## Artifacts

- Case IDs: `runs/baseline_alignment_2026-10-21/case_ids_seed0.json`
- 512-token diagnostics: `runs/baseline_alignment_2026-10-21/cap512/`
- 1,024-token baseline runs: `runs/baseline_alignment_2026-10-21/cap1024/`
- Strict audit: `runs/baseline_alignment_2026-10-21/strict_audit.json`
- MLC runs record runtime, data indices, seed, prompt/extractor method, token caps,
  and decoding in `config_snapshot.json`.
