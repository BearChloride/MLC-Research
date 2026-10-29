"""评测训练好的 C2C 投影器 —— AIME 上「融合 vs 不融合」对照（支持跨模型 / 同模型）。

- 跨模型：source（如 Qwen2.5-Math-1.5B）与 target（Qwen3-4B）喂同一份 prompt ids（位置对齐），
  source 的 KV 经 projector 融进 target；对照 = target 单独作答。
- 同模型：分析者前缀 KV → 求解者（角色前缀已补到等长，位置对齐）。
两者同抽取+评分。模式/源模型从 ckpt 的 meta 自动读取（可用 --source-model 覆盖）。

跑法：CUDA_VISIBLE_DEVICES=0 python scripts/eval_c2c_aime.py \
        --ckpt runs/c2c/<run>/final --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, _HERE)


def extract_int(text: str) -> str:
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return re.sub(r"[^0-9\-]", "", m[-1]) or m[-1].strip()
    m = re.findall(r"-?[0-9][0-9,]*", text)
    return m[-1].replace(",", "") if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="AIME 上评测 C2C 融合 vs 不融合")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL",
                    "/data/mxy/Models/Qwen/Qwen3-4B"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source-model", default=None, help="覆盖 ckpt 里记录的源模型")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gate", choices=["soft", "hard"], default="soft")
    ap.add_argument("--out", default=os.path.join("runs", "c2c", "eval_aime"))
    args = ap.parse_args()

    import torch
    from lychee_mas.eval.benchmarks import load as load_task
    from lychee_mas.eval.metrics import score
    from lychee_mas.methods.memory.channels.c2c_projector import (
        build_projector_stack,
        map_source_to_target_layers,
    )
    from lychee_mas.backends.hf_backend import HFBackend, common_suffix_len
    from train_c2c_projector import ANALYST_SYS, SOLVER_SYS

    blob = torch.load(os.path.join(args.ckpt, "projectors.pt"), map_location="cuda:0")
    b, meta = blob["build"], blob.get("meta", {})
    source_model = args.source_model or meta.get("source_model")
    cross = bool(source_model)

    tgt = HFBackend(args.model_path, device="cuda:0", dtype=torch.bfloat16, enable_thinking=False)
    nL, _, _ = tgt.kv_dims()
    tok = tgt.tok
    src = (HFBackend(source_model, device="cuda:0", dtype=torch.bfloat16, enable_thinking=False,
                     strict_hidden=False) if cross else tgt)

    projectors = build_projector_stack(
        b["num_hidden_layers"], b["head_dim"], b["num_kv_heads"],
        hidden_dim=b["hidden_dim"], intermediate_dim=b["intermediate_dim"],
        num_layers=b["num_layers"], dtype=torch.float32, zero_init=False,
        src_num_kv_heads=b.get("src_num_kv_heads"), src_head_dim=b.get("src_head_dim")).to("cuda:0")
    for proj, sd in zip(projectors, blob["state_dicts"]):
        proj.load_state_dict(sd)
    projectors.eval()
    for proj in projectors:
        proj.hard_gate = (args.gate == "hard")
    print(f"[eval] {len(projectors)} projectors | cross={cross} "
          f"source={os.path.basename(source_model) if cross else '(self)'} gate={args.gate}",
          flush=True)

    def chat_ids(sysp, problem):
        t = tok.apply_chat_template(
            [{"role": "system", "content": sysp}, {"role": "user", "content": problem}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(t, return_tensors="pt").input_ids[0].tolist()

    # 同模型：复刻训练时的等长前缀对齐
    analyst_sys, solver_sys = ANALYST_SYS, SOLVER_SYS
    if not cross:
        while len(chat_ids(analyst_sys, "x")) < len(chat_ids(SOLVER_SYS, "x")):
            analyst_sys += "\n"
        while len(chat_ids(solver_sys, "x")) < len(chat_ids(analyst_sys, "x")):
            solver_sys += "\n"

    data = load_task("aime_2024", n=(args.n or None))
    samples, fused_ok, plain_ok = [], 0, 0
    for i, it in enumerate(data):
        prob, gold = it["question"], it["gold"]
        solver = [{"role": "system", "content": solver_sys}, {"role": "user", "content": prob}]
        tgt_prompt = chat_ids(solver_sys, prob)
        if cross:
            src_prompt = tgt_prompt
            Lp = len(tgt_prompt)
            src_span = tgt_span = (0, Lp)
        else:
            src_prompt = chat_ids(analyst_sys, prob)
            ln = common_suffix_len(src_prompt, tgt_prompt)
            src_span, tgt_span = (len(src_prompt) - ln, ln), (len(tgt_prompt) - ln, ln)
        src_layers = map_source_to_target_layers(src.encode_kv_cache_ids(src_prompt), nL)

        g_fused = tgt.generate_chat_with_cache_fusion(solver, src_layers, projectors,
                                                      max_new_tokens=args.max_new_tokens,
                                                      src_span=src_span, tgt_span=tgt_span)
        g_plain = tgt.generate_chat(solver, max_new_tokens=args.max_new_tokens)
        a_f, a_p = extract_int(g_fused.text), extract_int(g_plain.text)
        c_f, c_p = score("aime", a_f, gold), score("aime", a_p, gold)
        fused_ok += c_f
        plain_ok += c_p
        samples.append({"q": prob[:200], "gold": gold, "fused_ans": a_f, "fused_correct": c_f,
                        "plain_ans": a_p, "plain_correct": c_p})
        print(f"  [{i+1}/{len(data)}] f={c_f:.0f}({a_f!r}) p={c_p:.0f}({a_p!r}) gold={gold!r}",
              flush=True)

    n = len(data)
    res = {"n": n, "fused_acc": round(fused_ok / n, 4), "plain_acc": round(plain_ok / n, 4),
           "cross": cross, "source_model": source_model, "ckpt": args.ckpt,
           "gate": args.gate, "max_new_tokens": args.max_new_tokens}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"metrics": res, "samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"\n[result] fused_acc={res['fused_acc']} vs plain_acc={res['plain_acc']} (n={n}) "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
