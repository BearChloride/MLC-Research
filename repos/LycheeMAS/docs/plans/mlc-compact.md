# Compact MLC implementation

This module is a resource-bounded implementation of the core method in
`2025.acl-long.1105`, using the repository's GSM8K baseline and prepared data.
Run it from the LycheeMAS repository with the existing `LycheeMAS` runtime:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_mlc_gsm8k.py --model-path ..\..\models\Qwen3-0.6B `
  --train-n 20 --eval-n 20 --sample-seed 0 --device cuda:0
```

The runner reads the raw GSM8K training JSONL from the workspace by default;
`--train-jsonl` or `LYCHEE_GSM8K_DATA_DIR` can point to another local copy. It
randomly samples training examples, then evaluates on the test split using the
same seed and official `exact` scorer as `run_langgraph_baseline.py`. Existing
single and chain3 run files are matched by `case_id`; their paired accuracies
and percentage-point deltas are written to `comparison.json` when present.

Each of three agents has an independent GRU role state, an automatic selector
over three general reasoning styles, a role-conditioned low-rank vocabulary
adapter, and an individual Q head. All agents in a round read the same prior
history and publish together; the next round receives all previous outputs.
A state-conditioned hypernetwork produces positive mixing weights, preserving
the QMIX monotonicity constraint. Per-agent Gaussian latent modules use action
and observation features for conditional KL, action/latent contrastive MI,
pairwise KL/cosine dissimilarity, and entropy regularization. The per-episode
objective combines bootstrapped team TD error, reward-advantage policy gradient,
and the role differentiation objective.

The base Qwen model is shared and frozen to fit the local GPU. The independent
low-rank vocabulary adapters are the trainable LLM policy parameters; the role
network, Q heads, mixing network, and latent modules train jointly. The paper's
cosine text reward is underspecified, so this implementation uses the existing
binary GSM8K exact-match reward and official evaluation scorer. It also uses
mean input-token embeddings for compact state/action features, uses one shared
base model instead of separate model copies, and trains one pass over a small
random sample. These choices make this a runnable method prototype, not a
paper-scale result. The checkpoint (`mlc_state.pt`) and per-episode
`train_log.json` are saved beside normal `outputs.jsonl`, `metrics.json`, and
`config.yaml` results.

## Initial run

On Qwen3-0.6B / RTX 4060 Laptop, four GSM8K train examples were used for one
update each and ten test cases were evaluated for two rounds. The test indices
were taken from the existing 100-case baseline sample so all three methods use
identical questions. Official `exact` accuracy was 7/10 (70%) for MLC, 8/10
(80%) for `langgraph_baseline_single`, and 9/10 (90%) for
`langgraph_baseline_chain3`; strict numeric accuracy was 6/10 (60%), 7/10
(70%), and 7/10 (70%), respectively. Official paired McNemar p-values were
1.0 against single and 0.5 against chain3; strict-score p-values were 1.0 for
both comparisons. The four training team rewards were 0, 1, 0, 0. Scoring
audit found one optimistic official-score case (pool index 151, gold 5,
prediction 3.25).

This confirms that the end-to-end training, checkpoint, evaluation, and paired
case matching paths work. These tiny samples do not establish an accuracy
improvement. The current runner is a **sidecar prototype**, not an optimizer
registered at `optimize_langgraph`. A role-diagnostics report is emitted as
`role_diagnostics.json`; a separate run name (`--method`) keeps this follow-up
from replacing the original 10-case artifacts.

The planned first comparison uses the same 100 case IDs already present in the
single and chain3 baseline outputs. It reports both framework `exact` scores
and strict numeric scores. Even a measured role choice distribution, agent
accuracy gap, output distance, or adapter update only establishes observed
role differentiation; none of these alone establishes reproduction of the
paper's effect.
