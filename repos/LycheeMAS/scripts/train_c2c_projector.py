"""训练 C2C 投影器（latent 通道）—— 支持两种设定，都保证 KV 融合的**位置对齐**。

A) 跨模型（`--source-model PATH`，忠实复现 C2C）：
   source/target 是两个不同模型，喂**同一份 token ids**（特殊 token id 跨 Qwen 家族共享）。
   位置天然 1:1 对齐；融合整个 prompt 段。层数/ kv 头数不同由层映射 + 跨维 projector 处理。
   例：source=Qwen2.5-Math-1.5B（数学知识），target=Qwen3-4B。

B) 同模型角色条件（默认）：source=分析者前缀+问题，target=求解者前缀+问题。
   **修复位置对齐**：先把两个角色 system 前缀**补到等 token 长度**，使共享的「问题+生成提示」后缀
   落在**相同绝对位置**（否则同一 token 在 source/target 的 RoPE 相位不一致——旧实现的 bug）。

公共：冻结所有 base，只训 projector（C2C 原则）；损失=只在答案 token 上的 CLM 交叉熵；
在共享对齐段上用逐层 C2CProjector 把 source KV 融进 target KV。

跑法：
  # 跨模型（推荐，忠实 C2C）
  CUDA_VISIBLE_DEVICES=0 python scripts/train_c2c_projector.py \
      --source-model /data/mxy/Models/Qwen/Qwen2.5-Math-1.5B --n 2000 --steps 400 \
      --out runs/c2c/qwen2.5math1.5b__qwen3-4b_gsm8k
  # 同模型（已修对齐）
  CUDA_VISIBLE_DEVICES=0 python scripts/train_c2c_projector.py --n 2000 --steps 400 \
      --out runs/c2c/qwen3-4b_self_gsm8k
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

ANALYST_SYS = ("You are a math analyst. Carefully identify the quantities, relations, and the "
               "solution strategy for the problem. Think about what computations are needed.")
SOLVER_SYS = ("You are a math solver. Solve the problem step by step and end with the final "
              "numeric answer after '#### '.")


def load_gsm8k(n: int):
    import pandas as pd
    f = glob.glob("/data/mxy/Project/CDM/Data/raw/gsm8k/main/train-*.parquet")
    df = pd.read_parquet(f[0])
    if n:
        df = df.iloc[:n]
    return list(zip(df["question"].tolist(), df["answer"].tolist()))


_OPENHERMES = ("/data/mxy/Project/CDM/Data/raw/c2c/teknium__OpenHermes-2.5/"
               "c2c_pairs.jsonl")


def load_openhermes(n: int, path: str = _OPENHERMES):
    """OpenHermes-2.5（C2C 论文原始训练语料），预处理成 {q,a} 的 jsonl。"""
    import json
    pairs = []
    with open(path) as fh:
        for line in fh:
            if n and len(pairs) >= n:
                break
            d = json.loads(line)
            q, a = d.get("q"), d.get("a")
            if q and a:
                pairs.append((q, a))
    return pairs


def load_data(name: str, n: int, path: str | None):
    if name == "gsm8k":
        return load_gsm8k(n)
    if name == "openhermes":
        return load_openhermes(n, path or _OPENHERMES)
    raise ValueError(f"unknown --data {name!r} (gsm8k|openhermes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 C2C 投影器（跨模型 / 同模型对齐修复）")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL",
                    "/data/mxy/Models/Qwen/Qwen3-4B"), help="target/receiver 模型")
    ap.add_argument("--source-model", default=None,
                    help="给定则跨模型（source/sharer）；不给则同模型角色条件")
    ap.add_argument("--data", default="gsm8k", choices=["gsm8k", "openhermes"],
                    help="训练语料：gsm8k(易) 或 openhermes(C2C 原始语料, 更难更杂)")
    ap.add_argument("--data-path", default=None, help="覆盖 openhermes jsonl 路径")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--proj-hidden", type=int, default=512)
    ap.add_argument("--proj-intermediate", type=int, default=512)
    ap.add_argument("--proj-layers", type=int, default=3)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-ans-len", type=int, default=320)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("runs", "c2c", "run"))
    args = ap.parse_args()

    import torch
    from lychee_mas.methods.memory.channels.c2c_projector import (
        build_projector_stack,
        map_source_to_target_layers,
    )
    from lychee_mas.backends.hf_backend import (
        HFBackend,
        common_suffix_len,
        make_fusion_cache,
    )
    from torch.optim import AdamW

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cross = args.source_model is not None

    # ---- target（receiver），冻结 ----
    tgt = HFBackend(args.model_path, device="cuda:0", dtype=torch.bfloat16, enable_thinking=False)
    tgt.model.eval()
    for p in tgt.model.parameters():
        p.requires_grad_(False)
    nL, nKV, hd = tgt.kv_dims()
    tok = tgt.tok

    # ---- source（sharer）：跨模型则另载，同模型则就是 target ----
    if cross:
        src = HFBackend(args.source_model, device="cuda:0", dtype=torch.bfloat16,
                        enable_thinking=False, strict_hidden=False)
        src.model.eval()
        for p in src.model.parameters():
            p.requires_grad_(False)
        nL_s, nKV_s, hd_s = src.kv_dims()
        print(f"[train] CROSS-MODEL  source={os.path.basename(args.source_model)} "
              f"(L={nL_s} kv={nKV_s} hd={hd_s}) -> target (L={nL} kv={nKV} hd={hd})", flush=True)
    else:
        src = tgt
        nKV_s, hd_s = nKV, hd
        print(f"[train] SAME-MODEL role-conditioned (L={nL} kv={nKV} hd={hd})", flush=True)

    # ---- projector 栈（target 层数；跨维由 src kv/hd 指定）----
    projectors = build_projector_stack(nL, hd, nKV, hidden_dim=args.proj_hidden,
                                       intermediate_dim=args.proj_intermediate,
                                       num_layers=args.proj_layers, dtype=torch.float32,
                                       zero_init=True, src_num_kv_heads=nKV_s,
                                       src_head_dim=hd_s).to("cuda:0")
    projectors.train()
    for proj in projectors:
        proj.anneal_steps = max(1, args.steps // 2)
    print(f"[train] projector params: {sum(p.numel() for p in projectors.parameters())/1e6:.1f}M "
          f"({nL} layers)", flush=True)

    def chat_ids(sys_prompt, problem):
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": problem}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return tok(text, return_tensors="pt").input_ids[0].tolist()

    # 同模型：把两个角色 system 前缀补到等 token 长度（=> 问题后缀绝对位置一致，修 RoPE 对齐）
    analyst_sys = ANALYST_SYS
    if not cross:
        dummy = "x"
        while len(chat_ids(analyst_sys, dummy)) < len(chat_ids(SOLVER_SYS, dummy)):
            analyst_sys += "\n"
        solver_sys = SOLVER_SYS
        while len(chat_ids(solver_sys, dummy)) < len(chat_ids(analyst_sys, dummy)):
            solver_sys += "\n"
        assert len(chat_ids(analyst_sys, dummy)) == len(chat_ids(solver_sys, dummy))
        print("[train] equalized role-prefix lengths -> aligned positions", flush=True)
    else:
        solver_sys = SOLVER_SYS

    opt = AdamW(list(projectors.parameters()), lr=args.lr, weight_decay=0.01)
    data = load_data(args.data, args.n, args.data_path)
    print(f"[train] data={args.data}({len(data)}) steps={args.steps} ga={args.grad_accum} "
          f"cross={cross}", flush=True)

    def build_example(problem, answer):
        """返回 (tgt_full_ids, labels, mapped_src_layers, src_span, tgt_span) 或 None（跳过）。"""
        tgt_prompt = chat_ids(solver_sys, problem)          # target 端 prompt
        src_prompt = tgt_prompt if cross else chat_ids(analyst_sys, problem)
        if len(tgt_prompt) > args.max_prompt_len or len(src_prompt) > args.max_prompt_len:
            return None
        if cross:                                           # 同一 ids 喂两模型，整段对齐
            Lp = len(tgt_prompt)
            src_span = tgt_span = (0, Lp)
        else:                                               # 等长前缀 -> 公共后缀同位对齐
            ln = common_suffix_len(src_prompt, tgt_prompt)
            if ln < 8:
                return None
            src_span = (len(src_prompt) - ln, ln)
            tgt_span = (len(tgt_prompt) - ln, ln)
        src_layers = map_source_to_target_layers(src.encode_kv_cache_ids(src_prompt), nL)
        ans = tok(answer, add_special_tokens=False).input_ids[:args.max_ans_len]
        ans = ans + [tok.eos_token_id]
        return tgt_prompt + ans, [-100] * len(tgt_prompt) + ans, src_layers, src_span, tgt_span

    step, running, n_acc, t0, di = 0, 0.0, 0, time.time(), 0
    opt.zero_grad()
    while step < args.steps:
        ex = build_example(*data[di % len(data)])
        di += 1
        if ex is None:
            continue
        tgt_full, labels, src_layers, src_span, tgt_span = ex
        cache = make_fusion_cache(src_layers, projectors, tgt.model.config, src_span, tgt_span)
        out = tgt.model(input_ids=torch.tensor([tgt_full], device="cuda:0"),
                        labels=torch.tensor([labels], device="cuda:0"),
                        past_key_values=cache, use_cache=True)
        (out.loss / args.grad_accum).backward()
        running += float(out.loss.detach())
        n_acc += 1
        if n_acc % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(list(projectors.parameters()), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            for proj in projectors:
                proj.update_temperature(step)
            if step % args.log_every == 0:
                gate = float(torch.sigmoid(projectors[nL // 2].key_gate_logit).detach())
                print(f"[step {step}/{args.steps}] loss={running/n_acc:.4f} mid_gate={gate:.3f} "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
                running, n_acc = 0.0, 0

    _save(projectors, args, nL, nKV, hd, nKV_s, hd_s, cross, os.path.join(args.out, "final"))
    print(f"[done] -> {os.path.join(args.out, 'final')}", flush=True)


def _save(projectors, args, nL, nKV, hd, nKV_s, hd_s, cross, out_dir):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dicts": [p.state_dict() for p in projectors],
        "build": {"num_hidden_layers": nL, "head_dim": hd, "num_kv_heads": nKV,
                  "src_num_kv_heads": nKV_s, "src_head_dim": hd_s,
                  "hidden_dim": args.proj_hidden, "intermediate_dim": args.proj_intermediate,
                  "num_layers": args.proj_layers},
        "meta": {"cross": cross, "source_model": args.source_model, "model_path": args.model_path,
                 "data": args.data, "n": args.n, "steps": args.steps}},
        os.path.join(out_dir, "projectors.pt"))


if __name__ == "__main__":
    main()
