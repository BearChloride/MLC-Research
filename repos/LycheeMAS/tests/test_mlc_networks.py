"""Core MLC mechanism invariants; skipped when the optional torch runtime is absent."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from lychee_mas.methods.prerun.mlc import MLCConfig, MLCNetworks  # noqa: E402
from scripts.run_mlc_gsm8k import (  # noqa: E402
    MLCChat,
    _train_episode,
    action_token_log_probability,
    sample_action_token,
)
from torch.nn import functional as F  # noqa: E402


def small_network() -> MLCNetworks:
    return MLCNetworks(MLCConfig(n_agents=3, state_dim=12, lm_hidden_size=10,
                                 role_dim=8, prefix_tokens=2, latent_dim=4))


def test_state_conditioned_mixing_is_monotonic_in_agent_q_values() -> None:
    network = small_network()
    state = torch.randn(12)
    hidden = network.initial_hidden(device=state.device, dtype=state.dtype)
    _recurrent, _means, _stds, q_values, _role_values, q_total = network.step(state, hidden)

    gradients, = torch.autograd.grad(q_total, q_values)

    assert gradients.shape == (3,)
    assert torch.all(gradients > 0)


def test_temperature_sampling_and_log_probability_are_identical() -> None:
    torch.manual_seed(7)
    logits = torch.tensor([[0.2, -0.4, 1.1, 0.0]])
    temperature = 0.73
    token, sampled_log_prob = sample_action_token(logits, temperature)
    rescored_log_prob = action_token_log_probability(logits, token, temperature)
    assert torch.allclose(sampled_log_prob, rescored_log_prob)


def test_cached_sampling_matches_prefix_teacher_forced_log_probability(monkeypatch) -> None:
    pytest.importorskip("transformers")
    import scripts.run_mlc_gsm8k as runner
    from transformers import GPT2Config, GPT2LMHeadModel

    class Tokenizer:
        eos_token_id = None

        @staticmethod
        def decode(_tokens, skip_special_tokens=True):
            return "sample"

    torch.manual_seed(11)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=31, n_positions=32,
                                       n_embd=16, n_layer=2, n_head=2)).eval()
    chat = MLCChat.__new__(MLCChat)
    chat.tok = Tokenizer()
    chat.model = model
    chat.embedding = model.get_input_embeddings()
    chat.device = torch.device("cpu")
    chat.gradient_checkpointing = False
    observed_log_probs = []
    sample_original = runner.sample_action_token

    def record_sample(logits, temperature):
        token, log_prob = sample_original(logits, temperature)
        observed_log_probs.append(float(log_prob.item()))
        return token, log_prob

    monkeypatch.setattr(runner, "sample_action_token", record_sample)
    prompt = torch.tensor([[1, 3, 4]])
    network = MLCNetworks(MLCConfig(n_agents=3, state_dim=12, lm_hidden_size=16,
                                   role_dim=8, prefix_tokens=2, latent_dim=4))
    state = torch.randn(12)
    _recurrent, role_means, role_log_stds, *_ = network.step(
        state, network.initial_hidden(device=state.device, dtype=state.dtype))
    roles = torch.stack([network.sample_role(role_means[i], role_log_stds[i])[0]
                         for i in range(3)])
    prefix = network.policy_prefix(0, roles[0])
    temperature = 0.63
    _text, generated, _prompt_tokens, _elapsed = chat.generate(
        prompt, prefix, max_new_tokens=4, temperature=temperature, sample=True)
    rescored = chat.action_log_probability(prompt, generated, prefix,
                                          temperature=temperature)
    assert len(observed_log_probs) == generated.shape[1] == 4
    assert torch.allclose(rescored, torch.tensor(sum(observed_log_probs) / len(observed_log_probs)),
                          atol=1e-5, rtol=1e-5)
    latent_loss, diversity_scores = network.role_differentiation_loss(
        state, roles, torch.randn(3, 12))
    policy_diversity_loss = -diversity_scores[0] * rescored
    (latent_loss + policy_diversity_loss).backward()
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in network.role_rnns.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in network.policy_prefixes.parameters())


def test_state_and_role_change_soft_prefix_and_policy_distribution() -> None:
    torch.manual_seed(3)
    network = small_network()
    state_a, state_b = torch.randn(12), torch.randn(12)
    hidden = network.initial_hidden(device=state_a.device, dtype=state_a.dtype)
    recurrent_a, means_a, _std_a, *_ = network.step(state_a, hidden)
    recurrent_b, means_b, _std_b, *_ = network.step(state_b, hidden)
    assert not torch.allclose(means_a[0], means_b[0])
    assert not torch.allclose(recurrent_a[0], recurrent_b[0])

    role_a = means_a[0]
    role_b = role_a + torch.ones_like(role_a) * 0.25
    prefix_a = network.policy_prefix(0, role_a)
    prefix_b = network.policy_prefix(0, role_b)
    assert prefix_a.shape == (2, 10)
    assert not torch.allclose(prefix_a, prefix_b)

    # A policy distribution over these input-conditioned features changes with role.
    policy_head = torch.nn.Linear(prefix_a.numel(), 7)
    logits_a = policy_head(prefix_a.flatten())
    logits_b = policy_head(prefix_b.flatten())
    assert not torch.allclose(F.softmax(logits_a, -1), F.softmax(logits_b, -1))


def test_mean_pooled_state_feature_discards_token_order() -> None:
    embedding = torch.nn.Embedding(17, 6)
    ids = torch.tensor([[2, 5, 9, 3]])
    reversed_ids = torch.flip(ids, dims=(1,))
    pooled = embedding(ids).float().mean(dim=1)
    reversed_pooled = embedding(reversed_ids).float().mean(dim=1)
    assert torch.allclose(pooled, reversed_pooled)


def test_diversity_scores_reach_role_encoder_and_policy_prefix() -> None:
    torch.manual_seed(5)
    network = small_network()
    state = torch.randn(12)
    hidden = network.initial_hidden(device=state.device, dtype=state.dtype)
    _recurrent, means, log_stds, *_ = network.step(state, hidden)
    roles, role_log_probs = zip(*(network.sample_role(means[i], log_stds[i])
                                  for i in range(3)))
    roles = torch.stack(roles)
    actions = torch.randn(3, 12)
    latent_loss, diversity_scores = network.role_differentiation_loss(state, roles, actions)

    # Stand-in categorical policy head validates the same score-function path used by
    # token log-probabilities through the soft-prefix-conditioned frozen LM.
    policy_head = torch.nn.Linear(2 * 10, 11, bias=False)
    token_log_probs = []
    for i, role in enumerate(roles):
        prefix = network.policy_prefix(i, role)
        logits = policy_head(prefix.flatten())
        token_log_probs.append(F.log_softmax(logits, -1)[i % logits.numel()])
    score_function = -(diversity_scores * torch.stack(token_log_probs)).mean()
    role_score_function = -(diversity_scores * torch.stack(role_log_probs)).mean()
    loss = latent_loss + score_function + role_score_function
    loss.backward()

    assert any(p.grad is not None and p.grad.norm() > 0
               for p in network.role_rnns.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in network.policy_prefixes.parameters())
    assert all(head.weight.grad is not None and head.weight.grad.norm() > 0
               for head in network.latent_posteriors)


def test_runner_uses_separate_role_token_signals_and_team_td_target() -> None:
    class FakeChat:
        device = torch.device("cpu")

        @staticmethod
        def state_feature(text):
            return torch.arange(12, dtype=torch.float32) * 0.01 + len(text) % 13

        @staticmethod
        def make_prompt(_question, _history):
            return torch.tensor([[1, 2]])

        @staticmethod
        def generate(_prompt, _prefix, *, max_new_tokens, temperature, sample):
            assert temperature == 0.8 and sample
            return "APPROVE: 1", torch.tensor([[3, 4, 5]]), 2, 0.0

        @staticmethod
        def action_feature(_generated):
            return torch.linspace(-1, 1, 12)

        @staticmethod
        def action_log_probability(_prompt, _generated, prefix, *, temperature):
            assert temperature == 0.8
            return -prefix.square().mean()

    network = small_network()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    stats = _train_episode({"question": "1+0?", "gold": "1"}, FakeChat(), network,
                           optimizer, rounds=2, max_new_tokens=8, gamma=0.9)
    assert stats["team_reward"] == 1.0
    assert stats["role_encoder_grad_norm"] > 0
    assert stats["policy_prefix_grad_norm"] > 0
    assert "token_policy_loss" in stats and "role_policy_loss" in stats
    assert stats["td_target_variance"] >= 0


def test_target_network_is_frozen_and_polyak_updates() -> None:
    network = small_network()
    target_before = next(network.target_q_heads.parameters()).detach().clone()
    online = next(network.q_heads.parameters())
    with torch.no_grad():
        online.add_(1.0)
    network.update_targets()
    target_after = next(network.target_q_heads.parameters()).detach()
    assert not torch.equal(target_before, target_after)
    assert all(not parameter.requires_grad for parameter in network.target_q_heads.parameters())


def test_role_differentiation_uses_three_agent_batch_and_is_finite() -> None:
    network = small_network()
    state = torch.randn(12)
    roles = torch.randn(3, 8)
    actions = torch.randn(3, 12)
    latent_loss, per_agent_scores = network.role_differentiation_loss(state, roles, actions)
    assert torch.isfinite(latent_loss)
    assert per_agent_scores.shape == (3,)
