"""generate 模式冒烟脚本 —— 用真实 LLM（如 vLLM 部署的 Qwen3-14B）测「构建 team」这一步。

只打 selector 的角色生成 LLM（CreateRoles→Check↔Check→SelectGroup），**不碰推理 runtime**。
产出 list[AgentSpec] 并打印。端点来自命令行或 env（LYCHEE_LLM_*），base_url 自动补 http:// 与 /v1。

编码器（Pareto 选择用）：
  - 若设了 env LYCHEE_EMBED_MODEL → 用真编码器（all-MiniLM-L6-v2，需本地权重）；
  - 否则退回 pool 的确定性 hash 嵌入（不下载模型也能跑通生成链；多样性/相关性为近似）。

用法：
  PYTHONPATH=src python scripts/test_generate_team.py \
      --model qwen --base-url 100.64.3.28:8010 --api-key None \
      --question "Solve this AIME problem: ..." --critique-rounds 1 --max-roles 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.core.types import TaskQuery  # noqa: E402
from lychee_mas.methods.build.selectors._llm import chat as _chat  # noqa: E402


def _norm_url(u: str) -> str:
    """补全 vLLM OpenAI 端点：加 http:// 和 /v1。"""
    u = u.strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    if not u.rstrip("/").endswith("/v1"):
        u = u.rstrip("/") + "/v1"
    return u


def _make_embed_fn(model_path=None):
    """给了路径（--embedder 或 env LYCHEE_EMBED_MODEL）用真编码器；否则退回确定性 hash 嵌入。"""
    emb_model = model_path or os.getenv("LYCHEE_EMBED_MODEL")
    if emb_model:
        from lychee_mas.methods.build.selectors.embedder import HFEmbedder
        return HFEmbedder(emb_model).embed, f"real encoder ({emb_model})"

    import numpy as np
    from lychee_mas.methods.build.selectors.pool import _hash_embed

    def embed_fn(sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        return np.stack([_hash_embed(s) for s in sentences])

    return embed_fn, "hash-embed stand-in（设 LYCHEE_EMBED_MODEL 换成真 all-MiniLM-L6-v2）"


def main() -> None:
    ap = argparse.ArgumentParser(description="generate 模式构建 team 冒烟测试")
    ap.add_argument("--model", default=os.getenv("LYCHEE_LLM_MODEL", "qwen"))
    ap.add_argument("--base-url", dest="base_url",
                    default=os.getenv("LYCHEE_LLM_BASE_URL"))
    ap.add_argument("--api-key", dest="api_key",
                    default=os.getenv("LYCHEE_LLM_API_KEY") or "EMPTY")
    ap.add_argument("--question", default="Solve this algebra problem: find x for 3x+7=22, "
                    "showing every step.")
    ap.add_argument("--critique-rounds", dest="critique_rounds", type=int, default=1,
                    help="CreateRoles↔Check 迭代上限（1=只生成不批判，最省调用；论文用 3）")
    ap.add_argument("--max-roles", dest="max_roles", type=int, default=5)
    ap.add_argument("--min-roles", dest="min_roles", type=int, default=1)
    ap.add_argument("--embedder", default=None,
                    help="句向量编码器路径（优先于 env LYCHEE_EMBED_MODEL；不给则退回 hash 嵌入）")
    ap.add_argument("--verbose", action="store_true", help="打印每个角色的完整 system prompt")
    args = ap.parse_args()

    base_url = _norm_url(args.base_url)
    api_key = "EMPTY" if str(args.api_key) in ("None", "none", "") else args.api_key
    print(f"[endpoint] model={args.model!r} base_url={base_url!r} api_key={api_key!r}")

    # chat_fn：显式绑定端点（不依赖 env，脚本自控打哪个模型）
    def chat_fn(messages):
        return _chat(messages, model=args.model, base_url=base_url, api_key=api_key)

    # 连通性自检：先打一发最小请求，失败给清晰报错
    print("[1/2] 连通性自检 ...", end=" ", flush=True)
    try:
        txt, usage = chat_fn([{"role": "user", "content": "reply with the single word: ok"}])
        print(f"OK（{usage.total} tokens, 回复={txt[:40]!r}）")
    except Exception as e:
        print("FAILED")
        print(f"[error] 打不通 {base_url}: {type(e).__name__}: {e}")
        print("  排查：① 远端 vLLM 是否在跑；② 本机能否访问该 IP:端口（内网/防火墙）；"
              "③ base_url 是否需要不同路径。")
        sys.exit(1)

    embed_fn, emb_note = _make_embed_fn(args.embedder)
    print(f"[embedder] {emb_note}")

    sel = REGISTRY.create("agent_selector", "agentinit", mode="generate",
                          chat_fn=chat_fn, embed_fn=embed_fn,
                          critique_rounds=args.critique_rounds,
                          min_roles=args.min_roles, max_roles=args.max_roles)

    print(f"[2/2] 构建团队（critique_rounds={args.critique_rounds}, "
          f"max_roles={args.max_roles}）...")
    t0 = time.time()
    team = sel.select(TaskQuery(question=args.question))
    dt = time.time() - t0

    print(f"\n== 生成的团队：{len(team)} 个 agent（gen_tokens={sel.last_gen_tokens}, "
          f"{dt:.1f}s）==")
    if len(team) == 1 and team[0].name == "Normal":
        print("  ⚠️ 只得到降级的 'Normal' —— 多半是模型没按 ## + JSON 格式输出，解析失败。"
              "可加 --verbose 看，或换 --critique-rounds / 检查模型是否输出了 <think> 等干扰。")
    for a in team:
        role = a.meta.get("role_name", a.name)
        print(f"  · {role}")
        if args.verbose:
            print(f"      {a.system_prompt}\n")


if __name__ == "__main__":
    main()
