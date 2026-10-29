"""Role-conditioned MARL components for the resource-bounded MLC reproduction."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MLCConfig:
    n_agents: int = 3
    state_dim: int = 1024
    lm_hidden_size: int = 1024
    role_dim: int = 64
    latent_dim: int = 32
    vocab_size: int = 151936
    prefix_tokens: int = 4
    alpha: float = 0.1
    beta: float = 0.1
    w_mi: float = 0.1
    w_kl: float = 0.1
    w_di: float = 0.1
    w_entropy: float = 0.01
    target_tau: float = 0.01
    role_pg_coef: float = 0.01
    diversity_token_pg_coef: float = 0.05


class MLCNetworks(nn.Module):
    """State-conditioned continuous role policies and monotonic team value."""

    def __init__(self, config: MLCConfig) -> None:
        super().__init__()
        if config.n_agents < 2 or config.role_dim < 1:
            raise ValueError("MLC requires at least two agents and nonempty role embeddings")
        self.config = config
        n, d, role, latent = (config.n_agents, config.state_dim,
                              config.role_dim, config.latent_dim)
        self.role_rnns = nn.ModuleList(nn.GRUCell(d, role) for _ in range(n))
        self.role_means = nn.ModuleList(nn.Linear(role, role) for _ in range(n))
        self.role_log_stds = nn.ModuleList(nn.Linear(role, role) for _ in range(n))
        self.role_values = nn.ModuleList(
            nn.Sequential(nn.Linear(d + role, role), nn.Tanh(), nn.Linear(role, 1))
            for _ in range(n))
        self.q_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d + role, role), nn.Tanh(), nn.Linear(role, 1))
            for _ in range(n))
        # A continuous role vector becomes virtual input embeddings inside the LM.
        self.policy_prefixes = nn.ModuleList(
            nn.Linear(role, config.prefix_tokens * config.lm_hidden_size)
            for _ in range(n))

        self.latent_priors = nn.ModuleList(
            nn.Linear(d + role, latent * 2) for _ in range(n))
        self.latent_posteriors = nn.ModuleList(
            nn.Linear(d + role + d, latent * 2) for _ in range(n))
        self.action_projections = nn.ModuleList(nn.Linear(d, latent) for _ in range(n))

        self.mix_hyper = nn.Sequential(nn.Linear(d, role), nn.ReLU(),
                                       nn.Linear(role, n + 1))
        # Slowly updated value/mixer targets lower bootstrapping variance without replay.
        self.target_q_heads = copy.deepcopy(self.q_heads)
        self.target_mix_hyper = copy.deepcopy(self.mix_hyper)
        for module in (self.target_q_heads, self.target_mix_hyper):
            module.requires_grad_(False)

    def initial_hidden(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(self.config.n_agents, self.config.role_dim,
                           device=device, dtype=dtype)

    def step(self, state: Tensor, hidden: Tensor, *, use_mixing: bool = True
             ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return recurrent state, continuous role distribution, Qs and Q-values."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state_vector = state.squeeze(0)
        recurrent = torch.stack([cell(state, hidden[i:i + 1]).squeeze(0)
                                 for i, cell in enumerate(self.role_rnns)])
        role_means = torch.stack([head(recurrent[i])
                                  for i, head in enumerate(self.role_means)])
        role_log_stds = torch.stack([
            head(recurrent[i]).clamp(-5.0, 2.0)
            for i, head in enumerate(self.role_log_stds)])
        qs = torch.stack([head(torch.cat((state_vector, recurrent[i]), dim=-1)).squeeze(-1)
                          for i, head in enumerate(self.q_heads)])
        role_values = torch.stack([
            head(torch.cat((state_vector, recurrent[i]), dim=-1)).squeeze(-1)
            for i, head in enumerate(self.role_values)])
        mix = self.mix_hyper(state_vector)
        weights = F.softplus(mix[:self.config.n_agents]) + 1e-6
        q_total = torch.sum(weights * qs) + mix[-1] if use_mixing else qs.sum()
        return recurrent, role_means, role_log_stds, qs, role_values, q_total

    def sample_role(self, mean: Tensor, log_std: Tensor) -> tuple[Tensor, Tensor]:
        distribution = torch.distributions.Normal(mean, log_std.exp())
        role = distribution.rsample()
        return role, distribution.log_prob(role).sum(dim=-1)

    def role_log_probability(self, role: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        return torch.distributions.Normal(mean, log_std.exp()).log_prob(role).sum(dim=-1)

    def policy_prefix(self, agent: int, role: Tensor) -> Tensor:
        return self.policy_prefixes[agent](role).reshape(
            self.config.prefix_tokens, self.config.lm_hidden_size)

    def target_value(self, state: Tensor, recurrent: Tensor, *, use_mixing: bool = True) -> Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state_vector = state.squeeze(0)
        q_values = torch.stack([
            head(torch.cat((state_vector, recurrent[i].detach()), dim=-1)).squeeze(-1)
            for i, head in enumerate(self.target_q_heads)])
        mix = self.target_mix_hyper(state_vector)
        weights = F.softplus(mix[:self.config.n_agents]) + 1e-6
        return torch.sum(weights * q_values) + mix[-1] if use_mixing else q_values.sum()

    @torch.no_grad()
    def update_targets(self) -> None:
        tau = self.config.target_tau
        for online, target in zip(self.q_heads.parameters(), self.target_q_heads.parameters()):
            target.lerp_(online, tau)
        for online, target in zip(self.mix_hyper.parameters(), self.target_mix_hyper.parameters()):
            target.lerp_(online, tau)

    @staticmethod
    def _split_gaussian(params: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = params.chunk(2, dim=-1)
        return mu.tanh(), logvar.clamp(-2.0, 2.0)

    @staticmethod
    def _kl_diag(mu_a: Tensor, lv_a: Tensor, mu_b: Tensor, lv_b: Tensor) -> Tensor:
        return 0.5 * (lv_b - lv_a + (lv_a.exp() + (mu_a - mu_b).square())
                      / lv_b.exp() - 1.0).sum(dim=-1)

    def role_differentiation_loss(self, state: Tensor, roles: Tensor, actions: Tensor
                                  ) -> tuple[Tensor, Tensor]:
        """Return latent loss and per-agent scores for a score-function policy update.

        Returns the differentiable latent objective and detached per-agent scores.
        The score-function path in the runner connects those scores to sampled
        discrete tokens and the continuous role policy.
        """
        state = state.reshape(1, -1)
        roles = roles.reshape(self.config.n_agents, -1)
        actions = actions.reshape(self.config.n_agents, -1).detach()
        posterior_inputs = [torch.cat((state, roles[i:i + 1], actions[i:i + 1]), dim=-1)
                            for i in range(self.config.n_agents)]
        posterior = [self._split_gaussian(self.latent_posteriors[i](F.layer_norm(
            posterior_inputs[i], (2 * self.config.state_dim + self.config.role_dim,))))
            for i in range(self.config.n_agents)]
        prior_inputs = [torch.cat((state, roles[i:i + 1]), dim=-1)
                        for i in range(self.config.n_agents)]
        prior = [self._split_gaussian(self.latent_priors[i](F.layer_norm(
            prior_inputs[i], (self.config.state_dim + self.config.role_dim,))))
            for i in range(self.config.n_agents)]
        samples = [mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                   for mu, lv in posterior]

        action_z = torch.stack([F.normalize(self.action_projections[i](F.layer_norm(
            actions[i], (self.config.state_dim,))), dim=-1)
                                for i in range(self.config.n_agents)])
        latent_z = torch.stack([F.normalize(z.squeeze(0), dim=-1) for z in samples])
        similarities = latent_z @ action_z.T / 0.1
        log_probs = F.log_softmax(similarities, dim=-1)
        mi_scores = log_probs.diagonal() + math_log(self.config.n_agents)
        kl_scores = torch.stack([
            self._kl_diag(posterior[i][0], posterior[i][1], prior[i][0], prior[i][1]).mean()
            for i in range(self.config.n_agents)])
        entropy_scores = torch.stack([
            0.5 * (1.0 + math_log(2.0 * torch.pi) + lv).sum(dim=-1).mean()
            for _mu, lv in posterior])

        pair_scores = roles.new_zeros(self.config.n_agents)
        pair_count = 0
        for i in range(self.config.n_agents):
            for j in range(i + 1, self.config.n_agents):
                mu_i, lv_i = posterior[i]
                mu_j, lv_j = posterior[j]
                # Symmetric KL is an explicit finite-sample proxy for distributional MI.
                pair_kl = 0.5 * (self._kl_diag(mu_i, lv_i, mu_j, lv_j)
                                 + self._kl_diag(mu_j, lv_j, mu_i, lv_i)).mean()
                pair_mi_proxy = F.cosine_similarity(mu_i, mu_j, dim=-1).mean()
                pair_reward = self.config.alpha * pair_kl - self.config.beta * pair_mi_proxy
                pair_scores[i] += pair_reward / 2.0
                pair_scores[j] += pair_reward / 2.0
                pair_count += 1
        pair_scores = pair_scores / max(pair_count, 1)

        per_agent_loss = (-self.config.w_mi * mi_scores
                          + self.config.w_kl * kl_scores
                          - self.config.w_di * pair_scores
                          - self.config.w_entropy * entropy_scores)
        scores = -per_agent_loss.detach()
        scores = scores - scores.mean()
        scores = scores / scores.std(unbiased=False).clamp_min(1.0)
        return per_agent_loss.mean(), scores


def math_log(value: float) -> float:
    import math

    return math.log(value)
