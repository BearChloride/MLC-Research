# MLC integration boundary

## Current status

MLC is currently a sidecar research prototype. `scripts/run_mlc_gsm8k.py`
owns training and inference because it controls token-level role-conditioned
generation, synchronous rounds, rewards, and the joint TD/role losses. It is
not registered with `optimize_langgraph`.

## Minimum integration target

The first plugin should apply an already-trained checkpoint at inference time;
training remains an offline runner concern. This keeps the existing generic
`StateGraph -> StateGraph` contract and avoids putting benchmark data,
optimizers, and long-running episode training inside graph construction.

Proposed call:

```python
graph = optimize_langgraph(
    graph,
    method="mlc",
    mode="apply",
    checkpoint="runs/.../mlc_state.pt",
    model_path="models/Qwen3-0.6B",
    agent_nodes={"solver_a": ..., "solver_b": ..., "solver_c": ...},
    state_extractor=...,  # graph state -> frozen-LM state feature
    rounds=2,
)
```

Contract:

- `checkpoint` contains the MLC config and network weights; the plugin validates
  agent count, hidden size, vocabulary size, and checkpoint schema before
  modifying the graph.
- `agent_nodes` names the graph nodes to wrap. Each must expose an
  MLC-compatible generation hook accepting the selected role and vocabulary
  logit bias. Opaque generic LLM nodes are unsupported until they expose that
  hook.
- `state_extractor` maps the current graph state/history to the same feature
  representation used at training time.
- At each round, the role selector runs once for all agents from the shared
  state, then all selected agents generate from the same prior history. Their
  outputs are committed together before the next round.
- `mode="apply"` only loads weights and wraps compatible nodes. No training
  data, reward callback, or optimizer is accepted by this inference plugin.
- Unsupported graph structure or model shape raises an explicit error; it must
  not silently fall back to ordinary debate.

This is a narrow compatibility contract, not a generic retrofit of arbitrary
LangGraph applications. The existing runner remains the reference execution
path until this plugin and an integration test are implemented. Results from
the runner must continue to be labeled **sidecar prototype** before then.
