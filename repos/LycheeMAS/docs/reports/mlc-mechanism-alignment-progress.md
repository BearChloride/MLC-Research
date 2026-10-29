# MLC Mechanism Alignment Progress

**Status: MLC reconstruction in progress; dynamic role assignment remains unverified.** The changes below align the implementation path with the paper's mechanisms, but they are not evidence of reproducing the paper's experimental effect.

## Step 1: Training Signal

| Mechanism | Current implementation | Qualification |
|---|---|---|
| Sampling and log probability | Both use temperature-scaled categorical logits at temperature 0.8 in training. Teacher-forced replay includes the same role-derived prefix and scores the generated token sequence. | Tested with a random causal Transformer and KV-cache decoding; the replayed mean token log probability matched sampled per-token values within `1e-5`. Greedy evaluation uses temperature 1.0. |
| Token and role policy gradients | Token log probability and continuous Gaussian role log probability are separate terms with separate advantages. `role_pg_coef=0.01`; diversity score-function coefficients are separately configurable. | No longer sums a categorical role log probability with token log probability. |
| Team TD target | Team Q is trained only on majority team reward at the terminal round and zero reward at earlier rounds, with target-network bootstrap. Individual correctness is not mixed into team TD reward. | Target Q heads and mixer are Polyak-updated with `tau=0.01`. The two-episode Qwen smoke had target variances `0.00410` and `0.00401`; these estimates are from only two transitions/episode and are not stable variance estimates. |
| Per-agent baselines | Token actor baseline uses each agent's Q estimate; role actor baseline uses a separate role-value estimate. Role-value heads regress the mean per-agent round reward. | Actor advantages are independent; the team TD target remains team-only. |

The old runner did not log actual training gradient norms. Its prior synthetic fixed-input gradient check reported GRU role-state `1.9077`, discrete selector `1.9595`, and output-adapter down projection `0.00360`; these were not gradients from an old training episode and are not directly comparable to the new architecture. On a fixed-seed, one-episode fake-runner smoke check after this change, latent loss was `0.463`, role-encoder gradient norm `0.882`, and policy-prefix gradient norm `0.678`. Before latent stabilization, the same smoke setup produced latent loss about `1.05e4`, role-encoder gradient norm about `6.93e3`, and prefix gradient norm about `8.50`. This motivated bounded Gaussian parameters, normalized latent inputs, and centered/scaled per-agent score-function advantages. These smoke values verify signal scale and connectivity only; they are not Qwen training statistics.

## Step 2: Role-Conditioned Policy And Differentiation

- Each agent has an independent recurrent role encoder. It receives the current shared-state feature and its recurrent state, then emits a continuous Gaussian role distribution conditioned on state and agent history.
- A sampled continuous role vector is projected to virtual prefix embeddings and prepended to the prompt embeddings passed as `inputs_embeds` to the frozen causal LM. It therefore conditions the Transformer at its input, rather than adding a final-logit bias.
- The discrete role classifier has been removed from policy decisions. No hand-authored role categories select prompts or policy behavior.
- The differentiable latent objective updates role-conditioned prior/posterior heads and reaches the sampled role through their inputs. Since LM-generated tokens are discrete and the LM is frozen, detached per-agent differentiation scores also form score-function losses against both role log probability and token log probability. The latter backpropagates through teacher-forced LM input embeddings into the soft-prefix map and role encoder.
- The base LM remains frozen as an 8GB resource-constrained downgrade. Trainable components are role encoders/distributions, prefix maps, value/mixing modules and latent modules.

## Proxy And Approximation Audit

| Quantity | Status |
|---|---|
| Latent posterior/prior | Implemented as diagonal Gaussian heads conditioned on state and continuous role; parameters are bounded for stable small-data training. |
| `MI(z_i, a_i)` | Contrastive similarity objective using the three agents as the batch/negative set: each sample has only two in-batch negatives and the InfoNCE score is upper-bounded by `ln(3) = 1.099` nats. This is a high-variance, weak-negative proxy, not a full-batch MI estimate. |
| `KL(N_i || p(z_i|o_i))` | Implemented as diagonal Gaussian KL between posterior and prior. It is an approximation to the paper's stated latent objective because the available observation/state representation is compressed. |
| Pairwise `alpha KL - beta MI` | Symmetric Gaussian KL plus cosine similarity proxy between posterior means. The pairwise MI term is not an exact MI estimator. |
| Entropy | Analytic diagonal-Gaussian entropy. |
| State feature | Mean pooling of frozen input-token embeddings. A permutation-invariance test confirms that reordering the same token IDs produces the identical feature; token order and cross-token interactions are discarded before role recurrence, so state conditioning is structurally present but information-limited. |
| Continuous assignment | State-conditioned Gaussian role embeddings plus joint policy/TD/differentiation training. The paper's description is underspecified; this is an explicit explanatory implementation, not a claim of algorithmic identity. |

These proxies now have gradient routes to the role encoder and policy prefix, but that alone does not establish that differentiation improves team collaboration. That causal claim requires training on the task and the planned ablations.

## Gate Results

The focused gradient/mechanism suite passes 12 tests. It verifies monotonic mixing, state-dependent role means, role-dependent soft-prefix/policy distributions, mean-pooling order loss, matching temperature sampling and replayed log probabilities through a random causal Transformer, nonzero differentiation gradients to the role encoder and prefix map, separate role/token losses in the runner, target-network updates, and team-reward TD wiring.

The real Qwen smoke used two training examples and two diagnostic examples, two collaboration rounds, at most 32 generated tokens and 512 prompt tokens. It completed without OOM. Episode role-encoder/prefix gradient norms were `0.581/4.050` and `0.331/0.968`. Across the two diagnostic cases and two rounds, per-agent mean role norms were `0.540`, `0.694` and `0.638`; mean coordinate variances across the four state observations were `5.63e-5`, `6.27e-5` and `4.17e-5`. These are nonzero but small changes on a tiny sample.

Using the trained checkpoint and one fixed prompt, the same agent's role mean changed by `L2=0.115` between the actual first-round state and the next-round state containing collaboration history. Substituting those two natural state-conditioned roles into the same prompt changed the next-token distribution by symmetric KL `0.259` nats and total variation `0.305`. A separate direct role intervention (`role + 0.5`) changed it more strongly (symmetric KL `8.169` nats, TV `0.810`); that intervention only verifies the policy input path and is not treated as evidence of natural task-driven role change. The production path prepends projected role embeddings to prompt embeddings through `inputs_embeds` in `MLCChat.generate` and `action_log_probability`.

### Runtime Ledger

| Result | Interpreter/runtime | Torch / CUDA | Weights | Outcome |
|---|---|---|---|---|
| Focused 12-test gradient/mechanism suite | `C:\Users\34174\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`, Python 3.12.14, Transformers 5.20.2 | `2.14.1+cu126` / CUDA 12.6; RTX 4060 Laptop 8GB | Random tiny GPT-2 in integration test; no Qwen weights for unit tests | 12 passed; no model OOM. |
| Initial Qwen 1-train/1-diagnostic case, one round | Same bundled runtime | `2.14.1+cu126` / CUDA 12.6; RTX 4060 Laptop 8GB | `D:\MAS-Research\models\Qwen3-0.6B` | Completed; gradient norms role/prefix `1.73/18.7`; diagnostic only. Auto-generated baseline sidecar invalidated. |
| Qwen 1-train/1-diagnostic case, two rounds | Same bundled runtime | `2.14.1+cu126` / CUDA 12.6; RTX 4060 Laptop 8GB | `D:\MAS-Research\models\Qwen3-0.6B` | Completed; gradient norms `0.375/1.268`; state-conditioned roles changed. Auto-generated baseline sidecar invalidated. |
| Qwen exploratory 1-train/1-diagnostic case, two rounds | `D:\Anaconda3\envs\geo_env\python.exe` (environment identified by user) | `2.14.1+cu126` / CUDA available; RTX 4060 Laptop 8GB | `D:\MAS-Research\models\Qwen3-0.6B` | Completed separately; gradient norms `0.384/0.221`. Auto-generated baseline sidecar invalidated because baseline runtime was unverified. |
| Main mechanism smoke: 2 train / 2 diagnostic cases, two rounds | Same bundled runtime, Python 3.12.14, Transformers 5.20.2 | `2.14.1+cu126` / CUDA 12.6; RTX 4060 Laptop 8GB | `D:\MAS-Research\models\Qwen3-0.6B` | Completed without OOM; role/prefix gradient norms and role-state distribution above. |
| Fixed-prompt role intervention on the 2/2 checkpoint | Same bundled runtime | `2.14.1+cu126` / CUDA 12.6; RTX 4060 Laptop 8GB | `D:\MAS-Research\models\Qwen3-0.6B` | Natural adjacent-state and direct role interventions changed next-token distribution as quantified above. |

The verified bundled runtime was `C:\Users\34174\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`, Python `3.12.14`, Torch `2.14.1+cu126`, CUDA runtime `12.6`, Transformers `5.20.2`, GPU `NVIDIA GeForce RTX 4060 Laptop GPU` with 8GB. The runner now records interpreter, Python/Torch/CUDA/Transformers versions, GPU name and resolved weights path in each new diagnostic and run metadata record.

The mechanism input/gradient gate has passed at structural and tiny real-model scale: role enters `inputs_embeds`, direct role changes alter Qwen's next-token distribution, natural adjacent shared states change role means and the same-prompt distribution, and training records nonzero role/prefix gradients. However, two training and two diagnostic examples cannot establish robust automatic role assignment or that diversity causes better collaboration. Therefore the project remains **MLC reconstruction in progress; dynamic role assignment not yet validated**. At the time this initial 2/2 smoke note was written, no 100-case run or ablation had started; the later agreed runs are recorded below.

## Three-Seed Attribution Gate (32 Train Cases)

After the earlier two-case smoke, the full agreed gate completed with 32 training cases, three collaboration rounds, and fixed seeds 101, 202, and 303. Each run used the same bundled runtime and local Qwen3-0.6B weights, with a 48-token generation cap and 512-token prompt cap. All three training/evaluation runs completed without OOM. The fixed 32-case gate evaluation official accuracies were 25.0%, 25.0%, and 18.75%; these are diagnostics, not a baseline comparison.

| Seed | Official accuracy (32 cases) | Role coordinate RMS | Mean adjacent-round role displacement | Displacement / mean role norm |
|---:|---:|---:|---:|---:|
| 101 | 25.0% | 0.08196 | 0.07177 | 11.01% |
| 202 | 25.0% | 0.07732 | 0.08186 | 13.26% |
| 303 | 18.75% | 0.07947 | 0.07358 | 11.59% |

Across the 32 training episodes per seed, the median unnormalized role-encoder gradient norms were 0.303 (seed 101), 0.331 (202), and 0.314 (303); median soft-prefix parameter gradient norms were 1.026, 1.394, and 1.245. The parameter groups differ in size and structure, so these raw norms are shown separately and are not interpreted as a direct magnitude ratio.

The displacement is an L2 difference between deterministic role means for the same agent/case on adjacent collaboration rounds, normalized by the mean embedding L2 norm. This shows measurable state/interaction-associated role movement, but it is not a state-shuffle-controlled causal test and has no bootstrap interval yet. A robust natural-state policy-distribution JSD/TV gate was not completed; prior two-case KL/TV values do not satisfy that standard. Thus the attribution gate is **partial, not passed**. 100-case results may be reported only as same-set preliminary comparisons, not attributed to the paper mechanism.

The runtime-ledger serialization and checkpoint diagnostic-device defects discovered during seed 101 were corrected. Its 32-case inference was resumed from the already completed checkpoint and per-case progress, not retrained. Evaluation now appends each case and its role diagnostic to `case_progress.jsonl`; baseline runs likewise append each completed case and resume by case ID.

The 100-case comparison uses benchmark-pool indices selected by `sample_seed=404`, explicitly passed to all runs through the same case-ID file. All use the bundled Python/Torch/CUDA/Transformers runtime, local weights, greedy decoding, three rounds, a 48-token generation cap and 512-token prompt cap. Each method's prompt template is fixed across its runs; the baseline prompt/agent topologies remain their native single and chain3 implementations, so prompt text is not identical between methods. No prompt tuning or case selection was done after seeing results. Per-case outputs and resumable `case_progress.jsonl` ledgers are retained under `runs/mlc_main_100` and `runs/mlc_ablation_100`.

### Same-Set 100-Case Preliminary Comparison

| Method | Official | Strict numeric |
|---|---:|---:|
| single | 15/100 (15%) | 4/100 (4%) |
| chain3 | 18/100 (18%) | 7/100 (7%) |
| MLC full, seed 101 | 18/100 (18%) | 13/100 (13%) |
| MLC full, seed 202 | 21/100 (21%) | 16/100 (16%) |
| MLC full, seed 303 | 16/100 (16%) | 11/100 (11%) |

Paired exact McNemar tests compare each MLC seed separately against each baseline on the same 100 case IDs. Official-score p-values for seeds 101/202/303 versus single were `0.6900/0.2863/1.0000`; versus chain3, `1.0000/0.6636/0.8318`. Strict-score p-values versus single were `0.0490/0.0075/0.0923`; versus chain3, `0.1796/0.0636/0.4807`. None of the 12 comparisons remains significant under one Holm correction over all 12 seed-by-baseline-by-scoring-rule tests (the smallest adjusted p-value is 0.0900). The two nominal strict-score findings are therefore not robust evidence of an advantage. These are preliminary paired comparisons, not mechanism attribution: the role-dynamicity gate remains partial, training is limited to 32 cases, and the Qwen base is frozen at 0.6B scale.

### Ablations (Seed 101)

| Variant | Official | Strict numeric | Paired McNemar vs full (official / strict) |
|---|---:|---:|---:|
| full | 18/100 (18%) | 13/100 (13%) | reference |
| no role embedding | 24/100 (24%) | 16/100 (16%) | `0.2632 / 0.6072` |
| no mixing | 17/100 (17%) | 12/100 (12%) | `1.0000 / 1.0000` |
| no `L_dis` | 23/100 (23%) | 15/100 (15%) | `0.3323 / 0.7905` |

None of the three ablations shows a significant paired difference from full. The numerically higher no-role and no-`L_dis` results do not establish that role embeddings or the diversity objective are ineffective; each is a single-seed, 100-case comparison with substantial sampling uncertainty. The no-mixing result is likewise inconclusive. Ablation comparisons are diagnostic, not causal proof, because there is only one training seed per ablation.

All eight runs (two baselines, three full seeds and three ablations) contain exactly the same 100 benchmark-pool case IDs; the IDs were checked from each per-case ledger. The full case-level predictions, scores and resumable ledgers are retained with their runs rather than being replaced by aggregate-only results. The final `no_l_dis` run completed without OOM. Its mean recorded latency was 25.87 seconds/case versus 10.04 seconds/case for full seed 101 during these runs; this operational difference is reported as observed and is not interpreted as a mechanism effect.

### Interpretation Boundary

The observed 100-case results do not establish whether neutral/negative differences mean that the mechanisms do not help or that the frozen 0.6B model, 32-case training budget, and compressed implementation lack capacity/data to express their benefit. Mechanism contribution and resource limitation cannot be separated by this experiment. The results are labeled **same-set preliminary comparison; not fully attributable to the paper mechanism**. The status remains **MLC reconstruction in progress; dynamic role assignment remains unverified**. No result is described as reproducing the paper's effect or reported accuracy.

The old discrete-role checkpoint is architecture-incompatible with the continuous-role/soft-prefix model and remains only an old-mechanism prototype artifact. All reported comparisons use one bundled runtime: Python 3.12.14, Torch 2.14.1+cu126, CUDA 12.6, Transformers 5.20.2, RTX 4060 Laptop 8GB, with weights at `D:\MAS-Research\models\Qwen3-0.6B`. The runner records this environment and resolved model path in run metadata.

Final verification after all runs: bundled-runtime `python -m pytest tests -q` returned `230 passed, 2 skipped`; `py_compile` succeeded for `scripts/run_mlc_gsm8k.py` and `scripts/run_langgraph_baseline.py`. No evaluation or test process remains running.
