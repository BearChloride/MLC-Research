"""Train and evaluate a compact MLC team on the existing GSM8K benchmark.

Example::

    python scripts/run_mlc_gsm8k.py --model-path ../../models/Qwen3-0.6B \
        --train-n 8 --eval-n 20 --sample-seed 0 --device cuda:0

This resource-bounded reproduction freezes the shared base LM and jointly trains
per-agent low-rank vocabulary adapters plus MLC's role, Q-mixing, and latent heads.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.audit import strict_numeric_score  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark  # noqa: E402
from lychee_mas.eval.sampling import sample_records  # noqa: E402
from lychee_mas.methods.prerun.mlc import MLCConfig, MLCNetworks  # noqa: E402
from run_langgraph_baseline import APPROVE_RULE  # noqa: E402


def sample_action_token(logits: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    """Sample from exactly the temperature-scaled categorical used by its log-prob."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits.float() / temperature
    distribution = torch.distributions.Categorical(logits=scaled)
    token = distribution.sample()
    return token, distribution.log_prob(token)


def action_token_log_probability(logits: Tensor, token: Tensor,
                                 temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.distributions.Categorical(
        logits=logits.float() / temperature).log_prob(token)


def _output_distance(texts: list[str]) -> float:
    """Mean normalized token edit distance for all within-round agent pairs."""
    tokenized = [re.findall(r"\w+|[^\w\s]", text.lower()) for text in texts]
    distances = []
    for left in range(len(tokenized)):
        for right in range(left + 1, len(tokenized)):
            a, b = tokenized[left], tokenized[right]
            ratio = difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()
            distances.append(1.0 - ratio)
    return sum(distances) / len(distances) if distances else 0.0


def _trainable_snapshot(network: MLCNetworks) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone()
            for name, parameter in network.named_parameters() if parameter.requires_grad}


def _trainable_update_report(network: MLCNetworks,
                             initial: dict[str, torch.Tensor]) -> dict[str, float]:
    deltas = [parameter.detach().cpu() - initial[name].detach().cpu()
              for name, parameter in network.named_parameters()
              if parameter.requires_grad and name in initial]
    before = [initial[name].detach().cpu() for name, parameter in network.named_parameters()
              if parameter.requires_grad and name in initial]
    initial_norm = torch.sqrt(sum(value.square().sum() for value in before))
    update_norm = torch.sqrt(sum(value.square().sum() for value in deltas))
    return {"initial_parameter_norm": float(initial_norm),
            "update_parameter_norm": float(update_norm),
            "relative_update_norm": float(update_norm / initial_norm.clamp_min(1e-12))}


def _runtime_metadata(model_path: str) -> dict[str, Any]:
    import platform

    import transformers

    cuda_available = torch.cuda.is_available()
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "transformers_version": transformers.__version__,
        "weights_path": str(Path(model_path).resolve()),
    }


def _paired_accuracy_test(mlc_rows: list[dict[str, Any]],
                          baseline_rows: list[dict[str, Any]], *,
                          strict: bool) -> dict[str, Any]:
    baseline_by_id = {str(row.get("case_id")): row for row in baseline_rows}
    mlc_by_id = {str(row.get("case_id")): row for row in mlc_rows}
    common = sorted(set(mlc_by_id) & set(baseline_by_id))
    mlc_only = baseline_only = 0
    applicable = 0
    for case_id in common:
        mlc_row, baseline_row = mlc_by_id[case_id], baseline_by_id[case_id]
        if strict:
            mlc_value = strict_numeric_score(mlc_row.get("prediction", ""),
                                             mlc_row.get("gold"))
            baseline_value = strict_numeric_score(baseline_row.get("prediction", ""),
                                                  baseline_row.get("gold"))
            if mlc_value is None or baseline_value is None:
                continue
            mlc_correct, baseline_correct = mlc_value == 1.0, baseline_value == 1.0
        else:
            mlc_correct = float(mlc_row.get("score") or 0.0) == 1.0
            baseline_correct = float(baseline_row.get("score") or 0.0) == 1.0
        applicable += 1
        mlc_only += int(mlc_correct and not baseline_correct)
        baseline_only += int(baseline_correct and not mlc_correct)
    discordant = mlc_only + baseline_only
    tail = sum(math.comb(discordant, i)
               for i in range(min(mlc_only, baseline_only) + 1)) / (2 ** discordant)
    return {"paired_n": applicable, "mlc_only_correct": mlc_only,
            "baseline_only_correct": baseline_only,
            "mcnemar_exact_two_sided_p": min(1.0, 2 * tail)}


def _answer(text: str) -> str:
    match = re.search(r"APPROVE:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return text.strip()
    value = match.group(1).strip()
    boxed = re.search(r"\\boxed\{(.*?)\}", value, re.DOTALL)
    if boxed:
        return boxed.group(1).strip()
    return value.splitlines()[0].strip() if value.splitlines() else ""


def _answer_key(answer: str) -> str:
    return re.sub(r"\s+", "", answer).replace(",", "").lower()


def _majority(answers: list[str]) -> str:
    valid = [answer for answer in answers if answer]
    if not valid:
        return ""
    counts = Counter(_answer_key(answer) for answer in valid)
    winner = max(counts, key=lambda key: (counts[key], -valid.index(next(
        answer for answer in valid if _answer_key(answer) == key))))
    return next(answer for answer in valid if _answer_key(answer) == winner)


class MLCChat:
    """Frozen base LM conditioned by continuous role-derived soft-prefix tokens."""

    def __init__(self, model_path: str, device: str, dtype: str,
                 max_input_tokens: int, gradient_checkpointing: bool = False) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.gradient_checkpointing = gradient_checkpointing
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        self.embedding = self.model.get_input_embeddings()
        self.hidden_size = int(self.model.config.hidden_size)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _ids(self, text: str) -> Tensor:
        if self.max_input_tokens > 0:
            encoded = self.tok(text, return_tensors="pt", truncation=True,
                               max_length=self.max_input_tokens)
        else:
            encoded = self.tok(text, return_tensors="pt", truncation=False)
        return encoded["input_ids"].to(self.device)

    def state_feature(self, text: str) -> Tensor:
        ids = self._ids(text)
        with torch.no_grad():
            return self.embedding(ids).float().mean(dim=1).squeeze(0)

    def action_feature(self, token_ids: Tensor) -> Tensor:
        if token_ids.numel() == 0:
            token_ids = torch.tensor([[self.tok.eos_token_id]], device=self.device)
        with torch.no_grad():
            return self.embedding(token_ids).float().mean(dim=1).squeeze(0)

    def make_prompt(self, question: str, history: list[dict[str, str]]) -> Tensor:
        context = "\n\n".join(
            f"{entry['source']}: {entry['content']}" for entry in history)
        user = f"Problem:\n{question}"
        if context:
            user += f"\n\nPrevious collaboration:\n{context}"
        messages = [
            {"role": "system", "content": (
                "You are one member of a team solving a math problem. "
                f"{APPROVE_RULE}")},
            {"role": "user", "content": user},
        ]
        try:
            rendered = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return self._ids(rendered)

    def _prefix(self, role: Tensor) -> Tensor:
        return role

    def generate(self, prompt_ids: Tensor, prefix_embeds: Tensor, *, max_new_tokens: int,
                 temperature: float, sample: bool) -> tuple[str, Tensor, int, float]:
        start = time.perf_counter()
        with torch.no_grad():
            prompt_embeds = self.embedding(prompt_ids)
            prefix_embeds = prefix_embeds.to(dtype=prompt_embeds.dtype).unsqueeze(0)
            inputs_embeds = torch.cat((prefix_embeds, prompt_embeds), dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                        device=self.device)
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                use_cache=True)
            past = output.past_key_values
            next_logits = output.logits[:, -1, :].float()
            tokens = []
            for _ in range(max_new_tokens):
                if sample:
                    token, _ = sample_action_token(next_logits, temperature)
                else:
                    token = (next_logits / temperature).argmax(dim=-1)
                tokens.append(token)
                if self.tok.eos_token_id is not None and int(token.item()) == self.tok.eos_token_id:
                    break
                output = self.model(input_ids=token[:, None], past_key_values=past,
                                    attention_mask=torch.ones(
                                        (1, inputs_embeds.shape[1] + len(tokens)),
                                        dtype=torch.long, device=self.device), use_cache=True)
                past = output.past_key_values
                next_logits = output.logits[:, -1, :].float()
            generated = (torch.stack(tokens, dim=1) if tokens else
                         torch.empty((1, 0), dtype=torch.long, device=self.device))
        elapsed = time.perf_counter() - start
        return (self.tok.decode(generated[0], skip_special_tokens=True).strip(),
                generated, int(prompt_ids.shape[1]), elapsed)

    def action_log_probability(self, prompt_ids: Tensor, generated: Tensor,
                               prefix_embeds: Tensor, *, temperature: float) -> Tensor:
        if generated.shape[1] == 0:
            return prefix_embeds.sum() * 0.0
        prompt_embeds = self.embedding(prompt_ids)
        prefix_embeds = prefix_embeds.to(dtype=prompt_embeds.dtype).unsqueeze(0)
        generated_embeds = self.embedding(generated)
        inputs_embeds = torch.cat((prefix_embeds, prompt_embeds, generated_embeds[:, :-1]), dim=1)
        if self.gradient_checkpointing:
            def _forward_logits(embeds: Tensor) -> Tensor:
                return self.model(inputs_embeds=embeds, use_cache=False).logits
            output_logits = torch.utils.checkpoint.checkpoint(
                _forward_logits, inputs_embeds, use_reentrant=False)
        else:
            output_logits = self.model(inputs_embeds=inputs_embeds, use_cache=False).logits
        start = prefix_embeds.shape[1] + prompt_embeds.shape[1] - 1
        logits = output_logits[:, start:start + generated.shape[1], :].float()
        return action_token_log_probability(
            logits, generated, temperature).sum(dim=-1).squeeze(0) / generated.shape[1]


def _load_train_records(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            answer = row["answer"].rsplit("####", 1)[-1].strip()
            records.append({"question": row["question"], "gold": answer, "kind": "exact"})
    if not records:
        raise ValueError(f"GSM8K training file is empty: {path}")
    return records


def _training_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
    elif os.environ.get("LYCHEE_GSM8K_DATA_DIR"):
        path = Path(os.environ["LYCHEE_GSM8K_DATA_DIR"]) / "train.jsonl"
    else:
        path = (ROOT / "model" / "GSM8K_train.jsonl_test.jsonl" / "grade_school_math"
                / "data" / "train.jsonl")
    if not path.is_file():
        raise FileNotFoundError(
            f"GSM8K train file not found: {path}; pass --train-jsonl or set "
            "LYCHEE_GSM8K_DATA_DIR")
    return path


def _record_question(record: dict[str, Any]) -> str:
    question = str(record["question"])
    context = record.get("context")
    return f"{context}\n\n{question}" if context else question


def _train_episode(record: dict[str, Any], chat: MLCChat, network: MLCNetworks,
                   optimizer: torch.optim.Optimizer, *, rounds: int,
                   max_new_tokens: int, gamma: float, ablation: str = "full") -> dict[str, float]:
    device = chat.device
    question = _record_question(record)
    hidden = network.initial_hidden(device=device, dtype=torch.float32)
    history: list[dict[str, str]] = []
    q_totals: list[Tensor] = []
    states: list[Tensor] = []
    recurrent_states: list[Tensor] = []
    rewards: list[list[float]] = []
    token_log_probs: list[Tensor] = []
    role_log_probs: list[Tensor] = []
    token_advantages: list[Tensor] = []
    role_advantages: list[Tensor] = []
    role_value_terms: list[Tensor] = []
    diversity_token_terms: list[Tensor] = []
    diversity_role_terms: list[Tensor] = []
    latent_losses: list[Tensor] = []
    answers: list[str] = []
    round_diagnostics: list[dict[str, Any]] = []
    calls = 0

    for _round in range(rounds):
        state = chat.state_feature(question + "\n" + "\n".join(
            f"{x['source']}: {x['content']}" for x in history))
        recurrent, role_means, role_log_stds, qs, role_values, q_total = network.step(
            state, hidden, use_mixing=ablation != "no_mixing")
        hidden = recurrent
        states.append(state)
        recurrent_states.append(recurrent)
        q_totals.append(q_total)
        round_rewards = []
        action_features = []
        sampled_roles = []
        round_role_log_probs = []
        round_token_log_probs = []
        round_qs = []
        round_role_values = []
        round_history = []
        round_agents = []
        for i in range(network.config.n_agents):
            role_vector, role_log_prob = network.sample_role(role_means[i], role_log_stds[i])
            prefix_role = torch.zeros_like(role_vector) if ablation == "no_role_embedding" else role_vector
            prefix = network.policy_prefix(i, prefix_role)
            prompt_ids = chat.make_prompt(question, history)
            response, generated, _prompt_tokens, _latency = chat.generate(
                prompt_ids, prefix, max_new_tokens=max_new_tokens, temperature=0.8, sample=True)
            answer = _answer(response)
            score = float(M.score("exact", answer, record["gold"]))
            strict_score = strict_numeric_score(answer, record["gold"])
            action_feature = chat.action_feature(generated)
            action_features.append(action_feature)
            token_log_prob = chat.action_log_probability(
                prompt_ids, generated, prefix, temperature=0.8)
            sampled_roles.append(role_vector)
            round_role_log_probs.append(role_log_prob)
            round_token_log_probs.append(token_log_prob)
            round_qs.append(qs[i])
            round_role_values.append(role_values[i])
            round_rewards.append(score)
            answers.append(answer)
            round_history.append({"source": f"agent_{i}", "content": response})
            round_agents.append({"agent": i, "role_vector": role_vector.detach().cpu().tolist(),
                                 "role_norm": float(role_vector.detach().norm()),
                                 "prediction": answer,
                                 "official_correct": score == 1.0,
                                 "strict_correct": strict_score == 1.0
                                 if strict_score is not None else None})
            calls += 1
        history.extend(round_history)
        rewards.append(round_rewards)
        role_log_probs.append(torch.stack(round_role_log_probs))
        token_log_probs.append(torch.stack(round_token_log_probs))
        token_advantages.append(torch.as_tensor(round_rewards, device=device) -
                                torch.stack(round_qs).detach())
        role_advantages.append(torch.full((network.config.n_agents,),
                                          sum(round_rewards) / len(round_rewards), device=device) -
                               torch.stack(round_role_values).detach())
        role_value_terms.append(F.smooth_l1_loss(torch.stack(round_role_values),
                                                 torch.full((network.config.n_agents,),
                                                            sum(round_rewards) / len(round_rewards),
                                                            device=device)))
        if ablation == "no_l_dis":
            latent_loss = state.new_zeros(())
            diversity_scores = state.new_zeros(network.config.n_agents)
        else:
            latent_loss, diversity_scores = network.role_differentiation_loss(
                state, torch.stack(sampled_roles), torch.stack(action_features))
        latent_losses.append(latent_loss)
        diversity_token_terms.append(-diversity_scores * torch.stack(round_token_log_probs))
        diversity_role_terms.append(-diversity_scores * torch.stack(round_role_log_probs))
        round_diagnostics.append({"round": _round, "agents": round_agents,
                                  "role_mean_norms": [float(x.detach().norm())
                                                      for x in role_means],
                                  "output_distance": _output_distance(
                                      [entry["content"] for entry in round_history])})

    team_answer = _majority(answers)
    team_reward = float(M.score("exact", team_answer, record["gold"]))
    td_terms = []
    td_targets = []
    for t, q_total in enumerate(q_totals):
        target = torch.as_tensor(team_reward if t == len(q_totals) - 1 else 0.0,
                                 device=device)
        if t + 1 < len(q_totals):
            target = target + gamma * network.target_value(
                states[t + 1], recurrent_states[t + 1], use_mixing=ablation != "no_mixing")
        td_targets.append(target.detach())
        td_terms.append(F.smooth_l1_loss(q_total, target))
    td_loss = torch.stack(td_terms).mean()
    token_actor_loss = -(torch.stack(token_advantages) * torch.stack(token_log_probs)).mean()
    role_actor_loss = -(torch.stack(role_advantages) * torch.stack(role_log_probs)).mean()
    role_actor_loss = network.config.role_pg_coef * role_actor_loss
    diversity_token_loss = (torch.stack(diversity_token_terms).mean() *
                            network.config.diversity_token_pg_coef)
    diversity_role_loss = (torch.stack(diversity_role_terms).mean() *
                           network.config.role_pg_coef)
    latent_loss = torch.stack(latent_losses).mean()
    role_value_loss = torch.stack(role_value_terms).mean()
    total_loss = (td_loss + token_actor_loss + role_actor_loss + latent_loss +
                  diversity_token_loss + diversity_role_loss + role_value_loss)
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    def grad_norm(parameters: Any) -> float:
        grads = [p.grad.detach().norm().square() for p in parameters if p.grad is not None]
        return float(torch.sqrt(torch.stack(grads).sum())) if grads else 0.0
    role_grad = grad_norm(list(network.role_rnns.parameters()) +
                          list(network.role_means.parameters()) +
                          list(network.role_log_stds.parameters()))
    prefix_grad = grad_norm(network.policy_prefixes.parameters())
    td_target_variance = float(torch.stack(td_targets).var(unbiased=False))
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()
    network.update_targets()
    return {"td_loss": float(td_loss.detach()),
            "token_policy_loss": float(token_actor_loss.detach()),
            "role_policy_loss": float(role_actor_loss.detach()),
            "latent_loss": float(latent_loss.detach()),
            "diversity_token_policy_loss": float(diversity_token_loss.detach()),
            "diversity_role_policy_loss": float(diversity_role_loss.detach()),
            "role_value_loss": float(role_value_loss.detach()),
            "team_reward": team_reward, "td_target_variance": td_target_variance,
            "role_encoder_grad_norm": role_grad, "policy_prefix_grad_norm": prefix_grad,
            "calls": float(calls), "round_diagnostics": round_diagnostics}


def _evaluate(records: list[dict[str, Any]], chat: MLCChat, network: MLCNetworks,
              *, rounds: int, max_new_tokens: int, task: str,
              method: str, ablation: str = "full", progress_path: str | None = None
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    if progress_path and os.path.isfile(progress_path):
        with open(progress_path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                completed[str(row["sample"]["case_id"])] = (row["sample"], row["diagnostic"])
    network.eval()
    with torch.no_grad():
        for index, record in enumerate(records):
            case_id = str(record.get("pool_index", index))
            if case_id in completed:
                continue
            question = _record_question(record)
            hidden = network.initial_hidden(device=chat.device, dtype=torch.float32)
            history: list[dict[str, str]] = []
            predictions = []
            round_diagnostics = []
            prompt_tokens = completion_tokens = 0
            latency = 0.0
            for _round in range(rounds):
                state = chat.state_feature(question + "\n" + "\n".join(
                    f"{x['source']}: {x['content']}" for x in history))
                hidden, role_means, _role_log_stds, _qs, _role_values, _q_total = network.step(
                    state, hidden, use_mixing=ablation != "no_mixing")
                round_history = []
                round_agents = []
                for i in range(network.config.n_agents):
                    role = role_means[i]
                    prefix_role = torch.zeros_like(role) if ablation == "no_role_embedding" else role
                    prefix = network.policy_prefix(i, prefix_role)
                    prompt_ids = chat.make_prompt(question, history)
                    response, generated, n_prompt, elapsed = chat.generate(
                        prompt_ids, prefix, max_new_tokens=max_new_tokens,
                        temperature=1.0, sample=False)
                    prompt_tokens += n_prompt
                    completion_tokens += int(generated.shape[1])
                    latency += elapsed
                    round_history.append({"source": f"agent_{i}", "content": response})
                    answer = _answer(response)
                    predictions.append(answer)
                    official_score = float(M.score("exact", answer, record.get("gold")))
                    strict_score = strict_numeric_score(answer, record.get("gold"))
                    round_agents.append({"agent": i,
                                         "role_vector": role.detach().cpu().tolist(),
                                         "role_norm": float(role.detach().norm()),
                                         "prediction": answer,
                                         "official_correct": official_score == 1.0,
                                         "strict_correct": strict_score == 1.0
                                         if strict_score is not None else None})
                history.extend(round_history)
                round_diagnostics.append({"round": _round, "agents": round_agents,
                                          "output_distance": _output_distance(
                                              [entry["content"] for entry in round_history])})
            prediction = _majority(predictions)
            scorer_kind = record.get("kind") or "exact"
            score = M.score(scorer_kind, prediction, record.get("gold"))
            sample = {
                "case_id": str(record.get("pool_index", index)), "task": task,
                "kind": scorer_kind, "method": method, "question": record["question"],
                "gold": record.get("gold"), "prediction": prediction, "score": score,
                "is_correct": bool(score == 1.0) if M.is_binary_scorer(scorer_kind) else None,
                "stop_reason": "fixed_rounds", "model_call_count": rounds * network.config.n_agents,
                "message_count": len(history), "input_positions_total": prompt_tokens,
                "gen_tokens": completion_tokens, "latency_s": round(latency, 3),
                "case_wall_time_s": round(latency, 3), "messages": history,
            }
            diagnostic = {"case_id": str(record.get("pool_index", index)),
                          "rounds": round_diagnostics}
            completed[case_id] = (sample, diagnostic)
            if progress_path:
                os.makedirs(os.path.dirname(progress_path), exist_ok=True)
                with open(progress_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"sample": sample, "diagnostic": diagnostic},
                                            ensure_ascii=False) + "\n")
                    stream.flush()
            print(f"[mlc] evaluated {len(completed)}/{len(records)} case_id={case_id}", flush=True)
    ordered = [completed[str(record.get("pool_index", i))] for i, record in enumerate(records)]
    return [pair[0] for pair in ordered], [pair[1] for pair in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--train-n", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="separate seed for the fixed evaluation subset")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=2048,
                        help="prompt token cap; 0 disables explicit input truncation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="trade recomputation for lower activation memory")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--method", default="mlc_compact",
                        help="result directory name; use a new name to preserve prior runs")
    parser.add_argument("--load-checkpoint", default=None,
                        help="load trained MLC weights and skip training")
    parser.add_argument("--resume-checkpoint", default=None,
                        help="resume training, optimizer, and episode log from a checkpoint")
    parser.add_argument("--diagnostics-from", default=None,
                        help="reuse training adapter-update diagnostics from an earlier run")
    parser.add_argument("--ablation", choices=["full", "no_role_embedding", "no_mixing", "no_l_dis"], default="full")
    parser.add_argument("--eval-case-ids", default=None,
                        help="JSON array of fixed GSM8K pool indices")
    args = parser.parse_args()
    if args.train_n < 1 or args.eval_n < 1 or args.rounds < 1:
        parser.error("train-n, eval-n, rounds must be >= 1")
    if args.max_input_tokens < 0:
        parser.error("max-input-tokens must be >= 0")

    torch.manual_seed(args.sample_seed)
    random.seed(args.sample_seed)
    train_records, train_meta = sample_records(
        _load_train_records(_training_path(args.train_jsonl)), n=args.train_n,
        seed=args.sample_seed)
    test_pool = load_benchmark(args.task, n=None)
    model_tag = args.model_tag or Path(args.model_path).name
    out_dir = M.result_dir(model_tag, args.method, args.task, root=args.out_root)
    state_path = os.path.join(out_dir, "mlc_state.pt")
    run_root = Path(out_dir).parents[2]
    baseline_seed_path = (run_root / model_tag / "langgraph_baseline_single" / args.task
                          / "outputs.jsonl")
    eval_indices = []
    if args.eval_case_ids:
        with open(args.eval_case_ids, encoding="utf-8") as stream:
            eval_indices = [int(x) for x in json.load(stream)]
        if len(eval_indices) != args.eval_n or len(set(eval_indices)) != len(eval_indices):
            parser.error("eval-case-ids must contain exactly eval-n unique pool indices")
        if any(index < 0 or index >= len(test_pool) for index in eval_indices):
            parser.error("eval-case-ids contains an index outside the benchmark pool")
    elif baseline_seed_path.is_file():
        baseline_seed_rows = M.read_outputs_jsonl(str(baseline_seed_path))
        if len(baseline_seed_rows) >= args.eval_n:
            eval_indices = [int(row.get("pool_index", row["case_id"]))
                            for row in baseline_seed_rows[:args.eval_n]]
            if any(index < 0 or index >= len(test_pool) for index in eval_indices):
                eval_indices = []
    if args.eval_case_ids or eval_indices:
        eval_records = [test_pool[index] for index in eval_indices]
        eval_meta = {"sample_seed": args.sample_seed, "pool_size": len(test_pool),
                     "sampled": len(eval_records), "sampled_indices": eval_indices,
                     "subset_source": "explicit_case_ids" if args.eval_case_ids else "langgraph_baseline_single"}
    else:
        eval_records, eval_meta = sample_records(test_pool, n=args.eval_n,
                                                  seed=(args.eval_seed if args.eval_seed is not None
                                                        else args.sample_seed))
        eval_meta["subset_source"] = "seeded_random_sample"
        eval_indices = eval_meta["sampled_indices"] or list(range(len(eval_records)))
    for pool_index, record in zip(eval_indices, eval_records):
        record["pool_index"] = pool_index

    chat = MLCChat(args.model_path, args.device, args.dtype, args.max_input_tokens,
                   gradient_checkpointing=args.gradient_checkpointing)
    runtime = _runtime_metadata(args.model_path)
    train_indices = train_meta["sampled_indices"] or list(range(len(train_records)))
    config_snapshot = {
        "script": "run_mlc_gsm8k.py",
        "task": args.task,
        "method": args.method,
        "ablation": args.ablation,
        "model_path": str(Path(args.model_path).resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "gradient_checkpointing": args.gradient_checkpointing,
        "runtime": runtime,
        "train_jsonl": str(_training_path(args.train_jsonl).resolve()),
        "train_n": args.train_n,
        "train_sample_seed": args.sample_seed,
        "train_pool_indices": train_indices,
        "eval_n": len(eval_records),
        "eval_sample_seed": eval_meta["sample_seed"],
        "eval_pool_size": eval_meta["pool_size"],
        "eval_pool_indices": eval_indices,
        "eval_subset_source": eval_meta.get("subset_source"),
        "rounds": args.rounds,
        "agents_per_round": 3,
        "max_new_tokens_per_agent_call": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "input_truncation": ("disabled" if args.max_input_tokens == 0
                              else "enabled"),
        "evaluation_decoding": {"do_sample": False, "temperature": 1.0,
                                "selection": "argmax"},
        "training_action_sampling": {"do_sample": True, "temperature": 0.8,
                                     "selection": "categorical"},
        "prompt_and_answer_method": {
            "prompt": "MLC shared system prompt with question and prior collaboration; "
                      "continuous role soft-prefix prepended to inputs_embeds",
            "answer_extraction": "per-agent extractor followed by normalized majority vote",
            "scoring": "lychee_mas.eval.metrics.M.score; strict audit uses "
                       "lychee_mas.eval.audit.strict_numeric_score",
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_snapshot.json"), "w", encoding="utf-8") as stream:
        json.dump(config_snapshot, stream, ensure_ascii=False, indent=2)
    config = MLCConfig(n_agents=3, state_dim=chat.hidden_size,
                       lm_hidden_size=chat.hidden_size)
    network = MLCNetworks(config).to(chat.device)
    trainable_initial = _trainable_snapshot(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    train_log = []
    checkpoint_path = args.resume_checkpoint or args.load_checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=chat.device, weights_only=False)
        network.load_state_dict(checkpoint["network"])
        train_log = checkpoint.get("train_log", [])
        trainable_initial = checkpoint.get("trainable_initial", trainable_initial)
        if args.resume_checkpoint and checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[mlc] loaded checkpoint {checkpoint_path}; "
              f"completed episodes={len(train_log)}")
    if not args.load_checkpoint:
        train_indices = train_meta["sampled_indices"] or list(range(len(train_records)))
        for episode_index in range(len(train_log), len(train_records)):
            train_index = train_indices[episode_index]
            record = train_records[episode_index]
            epoch = episode_index + 1
            stats = _train_episode(record, chat, network, optimizer,
                                   rounds=args.rounds, max_new_tokens=args.max_new_tokens,
                                   gamma=args.gamma, ablation=args.ablation)
            train_log.append({"episode": epoch, "train_pool_index": train_index,
                              **stats})
            checkpoint = {"config": config.__dict__, "network": network.state_dict(),
                          "trainable_initial": trainable_initial, "train_log": train_log,
                          "optimizer": optimizer.state_dict()}
            temporary_state_path = state_path + ".tmp"
            torch.save(checkpoint, temporary_state_path)
            os.replace(temporary_state_path, state_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[mlc] train {epoch}/{len(train_records)} reward={stats['team_reward']:.0f} "
                  f"td={stats['td_loss']:.4f} token_pg={stats['token_policy_loss']:.4f} "
                  f"role_pg={stats['role_policy_loss']:.4f} "
                  f"role_grad={stats['role_encoder_grad_norm']:.3g} "
                  f"prefix_grad={stats['policy_prefix_grad_norm']:.3g}", flush=True)

    method = args.method
    samples, eval_diagnostics = _evaluate(
        eval_records, chat, network, rounds=args.rounds,
        max_new_tokens=args.max_new_tokens, task=args.task, method=method,
        ablation=args.ablation, progress_path=os.path.join(out_dir, "case_progress.jsonl"))
    out_dir = M.result_dir(model_tag, method, args.task, root=args.out_root)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"config": config.__dict__, "network": network.state_dict(),
                "trainable_initial": trainable_initial, "train_log": train_log,
                "optimizer": optimizer.state_dict()}, state_path)
    log_path = os.path.join(out_dir, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as stream:
        json.dump(train_log, stream, indent=2)
    agent_totals = [{"agent": agent, "official_correct": 0, "strict_correct": 0,
                     "strict_applicable": 0, "responses": 0}
                    for agent in range(config.n_agents)]
    role_norms = []
    roles_by_agent: list[list[Tensor]] = [[] for _ in range(config.n_agents)]
    output_distances = []
    for case in eval_diagnostics:
        for round_item in case["rounds"]:
            output_distances.append(round_item["output_distance"])
            for agent_result in round_item["agents"]:
                total = agent_totals[agent_result["agent"]]
                total["responses"] += 1
                total["official_correct"] += int(agent_result["official_correct"])
                if agent_result["strict_correct"] is not None:
                    total["strict_applicable"] += 1
                    total["strict_correct"] += int(agent_result["strict_correct"])
                role_norms.append(agent_result["role_norm"])
                roles_by_agent[agent_result["agent"]].append(
                    torch.as_tensor(agent_result["role_vector"]))
    for total in agent_totals:
        total["official_accuracy"] = (total["official_correct"] / total["responses"]
                                      if total["responses"] else None)
        total["strict_accuracy"] = (total["strict_correct"] / total["strict_applicable"]
                                    if total["strict_applicable"] else None)
    diagnostics = {
        "status": "role differentiation is measured, not evidence of paper-effect reproduction",
        "evaluation_cases": eval_diagnostics,
        "per_agent_accuracy": agent_totals,
        "role_embedding_state_variation_by_agent": [
            {"agent": agent,
             "coordinate_variance_mean": float(torch.stack(values).var(dim=0,
                                                                        unbiased=False).mean())
             if len(values) > 1 else 0.0,
             "embedding_count": len(values)}
            for agent, values in enumerate(roles_by_agent)],
        "mean_role_embedding_norm": (sum(role_norms) / len(role_norms)
                                     if role_norms else None),
        "mean_pairwise_output_distance": (sum(output_distances) / len(output_distances)
                                          if output_distances else None),
        "output_distance_definition": (
            "mean pairwise normalized SequenceMatcher token edit distance "
            "within each case and round"),
        "trainable_parameter_updates": _trainable_update_report(network, trainable_initial),
        "training_checkpoint": args.load_checkpoint or state_path,
        "policy_conditioning": ("continuous state-conditioned role -> learned soft-prefix "
                                "embeddings"),
        "frozen_base_lm": True,
        "state_feature": ("mean pooling of frozen input-token embeddings; loses order "
                          "and interactions"),
        "mi_estimator": "contrastive batch has 3 agents; high variance and weak negatives",
        "runtime": runtime,
    }
    if args.diagnostics_from:
        with open(args.diagnostics_from, encoding="utf-8") as stream:
            source_diagnostics = json.load(stream)
        diagnostics["trainable_parameter_updates"] = source_diagnostics.get(
            "trainable_parameter_updates", source_diagnostics.get("adapter_updates"))
        diagnostics["trainable_update_source"] = args.diagnostics_from
    diagnostics_path = os.path.join(out_dir, "role_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
    diagnostic_lines = [
        "# MLC role differentiation diagnostics", "",
        "Observed role differentiation is not the same as reproducing the paper's effect.",
        "",
        f"Evaluation cases: {len(eval_diagnostics)}",
        f"Mean within-round pairwise output distance: "
        f"{diagnostics['mean_pairwise_output_distance']:.4f}",
        "",
        "| Agent | Official accuracy / responses | Strict numeric accuracy / applicable |",
        "|---:|---:|---:|",
    ]
    for total in agent_totals:
        strict_acc = (f"{total['strict_accuracy']:.1%} / {total['strict_applicable']}"
                      if total["strict_accuracy"] is not None else "n/a")
        diagnostic_lines.append(
            f"| {total['agent']} | {total['official_accuracy']:.1%} / {total['responses']} "
            f"| {strict_acc} |" if total["responses"] else
            f"| {total['agent']} | n/a | {strict_acc} |")
    diagnostic_lines.extend(["", "## Continuous role diagnostics", "",
                             f"Mean role embedding norm: {diagnostics['mean_role_embedding_norm']}",
                             "Roles are continuous; no discrete selector classifier participates "
                             "in policy decisions.", ""])
    for variation in diagnostics["role_embedding_state_variation_by_agent"]:
        diagnostic_lines.append(
            f"- Agent {variation['agent']}: state-conditioned coordinate variance mean="
            f"{variation['coordinate_variance_mean']:.6g} "
            f"(n={variation['embedding_count']})")
    update = diagnostics["trainable_parameter_updates"]
    diagnostic_lines.extend(["", "## Trainable parameter updates", "",
                             f"- Update norm: {update['update_parameter_norm']:.6g}",
                             f"- Relative update norm: {update['relative_update_norm']:.6g}",
                             "", "## Mechanism limitations", "",
                             "- Mean pooled frozen embeddings discard token order and "
                             "cross-token interactions before the recurrent role encoder.",
                             "- Contrastive MI uses three agents as its batch/negative set; this "
                             "estimator is high variance and weakly identified.",
                             "- Latent MI/pairwise terms are finite-sample proxies, not an exact "
                             "reimplementation of the paper's estimator.",
                             "- Base LM is frozen for 8GB; role, prefix, value and latent "
                             "modules are trained."])
    diagnostic_lines.extend(["", f"Raw per-case/per-round records: `{diagnostics_path}`.",
                             ""])
    diagnostic_report = os.path.join(out_dir, "role_diagnostics.md")
    with open(diagnostic_report, "w", encoding="utf-8") as stream:
        stream.write("\n".join(diagnostic_lines))
    summary = M.aggregate_samples(samples, {"task": args.task, "method": method,
                                            "model": model_tag})
    mlc_strict = [strict_numeric_score(sample["prediction"], sample.get("gold"))
                  for sample in samples]
    mlc_strict = [score for score in mlc_strict if score is not None]
    comparison = {method: {
        "n": len(samples), "accuracy_official": summary.get("accuracy"),
        "accuracy_strict": (sum(mlc_strict) / len(mlc_strict) if mlc_strict else None),
    }}
    for baseline in ("single", "chain3"):
        baseline_path = (run_root / model_tag / f"langgraph_baseline_{baseline}" / args.task
                         / "outputs.jsonl")
        if not baseline_path.is_file():
            continue
        baseline_rows = M.read_outputs_jsonl(str(baseline_path))
        selected_ids = {str(sample["case_id"]) for sample in samples}
        matched = [row for row in baseline_rows if str(row.get("case_id")) in selected_ids]
        matched_ids = {str(row.get("case_id")) for row in matched}
        if len(matched) != len(samples) or matched_ids != selected_ids:
            print(f"[mlc] baseline {baseline}: only {len(matched)}/{len(samples)} "
                  "matching case ids; omitted from paired accuracy comparison")
            continue
        base_acc = sum(float(row.get("score", 0.0)) for row in matched) / len(matched)
        strict_scores = [strict_numeric_score(row.get("prediction", ""), row.get("gold"))
                         for row in matched]
        strict_scores = [score for score in strict_scores if score is not None]
        comparison[f"baseline_{baseline}"] = {
            "n": len(matched), "accuracy_official": round(base_acc, 4),
            "accuracy_strict": (round(sum(strict_scores) / len(strict_scores), 4)
                                if strict_scores else None),
            "delta_pp": round((float(summary.get("accuracy") or 0.0) - base_acc) * 100, 2),
            "scorer": "official",
            "paired_official": _paired_accuracy_test(samples, matched, strict=False),
            "paired_strict": _paired_accuracy_test(samples, matched, strict=True),
        }
    with open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8") as stream:
        json.dump(comparison, stream, indent=2)
    table = ["Sidecar MLC prototype; not a reproduction claim.", "",
             "| Method | Cases | Official exact | Strict numeric |",
             "|---|---:|---:|---:|"]
    for name, values in comparison.items():
        official = values.get("accuracy_official")
        strict = values.get("accuracy_strict")
        table.append(f"| {name} | {values['n']} | "
                     f"{official:.1%} | {strict:.1%} |"
                     if official is not None and strict is not None else
                     f"| {name} | {values['n']} | {official} | {strict} |")
    table.extend(["", "Paired test-set identity is checked by `case_id`. These pilot "
                  "accuracies do not establish reproduction of the paper's effect.", "",
                  "Paired exact McNemar tests (MLC vs baseline):"])
    for name, values in comparison.items():
        if name.startswith("baseline_"):
            for scorer_name in ("official", "strict"):
                paired = values[f"paired_{scorer_name}"]
                table.append(
                    f"- {scorer_name} vs {name}: n={paired['paired_n']}, "
                    f"MLC-only={paired['mlc_only_correct']}, "
                    f"baseline-only={paired['baseline_only_correct']}, "
                    f"p={paired['mcnemar_exact_two_sided_p']:.4f}")
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(table) + "\n")
    M.write_results(out_dir, samples, summary, {
        "script": "run_mlc_gsm8k.py", "task": args.task, "method": method,
        "model_path": args.model_path, "model_tag": model_tag, "team_size": config.n_agents,
        "runtime": runtime,
        "rounds": args.rounds, "train_n": len(train_records), "eval_n": len(samples),
        "sample_seed": args.sample_seed, "eval_seed": eval_meta["sample_seed"],
        "train_sampled_indices": train_meta["sampled_indices"],
        "pool_size": eval_meta["pool_size"], "sampled_indices": eval_meta["sampled_indices"],
        "eval_subset_source": eval_meta["subset_source"],
        "gamma": args.gamma, "learning_rate": args.lr,
        "loss": ("team-only TD + separate token/continuous-role policy advantages + "
                 "latent differentiation with policy score-function paths"),
        "base_lm_frozen": True,
        "policy_conditioning": "continuous role embedding mapped to soft-prefix input embeddings",
        "role_assignment": "state-conditioned continuous Gaussian role distribution",
        "ablation": args.ablation,
        "ablation_definition": {
            "no_role_embedding": "zero role vector at policy-prefix projection; agent-specific prefix bias remains",
            "no_mixing": "sum individual Q values for team TD instead of learned monotonic mixer",
            "no_l_dis": "disable latent differentiation loss and its role/token score-function terms",
        }.get(args.ablation, "none"),
        "state_file": state_path, "train_log": log_path,
        "comparison_file": os.path.join(out_dir, "comparison.json"),
        "role_diagnostics_file": diagnostics_path,
        "role_diagnostics_report": diagnostic_report,
        "role_differentiation_status": (
            "continuous role/state diagnostics; not paper-effect reproduction"),
    })
    acc = summary.get("accuracy", summary.get("mean_score"))
    print(f"[mlc] accuracy={acc} results={out_dir}")


if __name__ == "__main__":
    main()
