"""MASPO 提示资产（MATH 任务族，逐字 vendored）。

来源：https://github.com/wangzx1219/MASPO 的 ``prompts.py``（ICML 2026；
arXiv:2605.06623 "MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems"）。
上游仓库未附 LICENSE（匿名发布）——仅迁移学术复现所需的 MATH 任务资产并保留本出处引用，
不整包搬运（CLAUDE.md 规则 9）；对外发布前需与作者确认许可。

本框架当前只接 MATH 任务族（math500 等），其余任务族（choice/code）待需要时按同款方式增补，
取用不存在的任务族/角色显式 KeyError。
"""
from __future__ import annotations

# ------------------------- 角色描述（ROLE_DESCRIPTIONS） -------------------------
# key = AgentSpec.role（与原版 AgentType.value 对应）
ROLE_DESCRIPTIONS = {
    "predictor": "Primary solver that provides step-by-step mathematical reasoning and final answer.",
    "reflector": ("Critical reviewer that identifies errors in previous solutions and "
                  "provides corrected reasoning."),
}

# ------------------------ 种子提示模板（AGENT_TEMPLATES[MATH]） ------------------------
# 占位符：{question} / {context}（MASPO 语义：整条模板即该 agent 的单条 user 消息）
AGENT_TEMPLATES = {
    "predictor": (
        "Let's think step by step. Show your final answer bracketed between <answer> and </answer> tags.\n"
        "{context}\n"
        "Question: {question}\nReasoning:"
    ),
    "reflector": (
        "Please review the solution above and criticize where it might be wrong. "
        "Show your final answer bracketed between <answer> and </answer>.\n"
        "Question: {question}\nSolution: {context}\nFeedback: indicating the reflection of the solution given the question.\nCorrect answer:"
    ),
}

# ------------------------- 压缩提示（COMPRESS_PROMPTS[MATH]） -------------------------
COMPRESS_PROMPT = (
    "Below is a solution to a math problem. "
    "Summarize the key reasoning steps and the final answer in at most 30 words.\n\n"
    "{raw}\n\nSummary:"
)

# --------------------- 反思变异模板（PROMPT_OPTIMIZE_TEMPLATE[MATH]） ---------------------
PROMPT_OPTIMIZE_TEMPLATE = """
You are optimizing a prompt for a specific agent in a multi-agent mathematical reasoning system.
CRITICAL: The agent's core role and responsibilities MUST be preserved in the optimized prompt.

Agent Type: {agent_type}
Current System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```
Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:
<analyse>Analyse what drawbacks exist in the results produced by the reference prompt and how to improve them.</analyse>
<modification>One sentence summary of the key improvement</modification>
<prompt>Provide the complete optimized prompt</prompt>
"""

# ------------------- 终端成对评估（ANSWER_EVALUATE_TEMPLATE[MATH]） -------------------
ANSWER_EVALUATE_TEMPLATE = """
You are evaluating two outputs (A and B) from an agent of type {agent_type}.
Based on the input, requirements and the agent's role, evaluate the two responses, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Input to the Agent
Question: {question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

Guidelines:
- If one contains a clearly correct final answer and the other does not, choose the correct one — even if its explanation is shorter.
- If both answers appear correct, prefer the one with clearer, logically sound reasoning.
- Do NOT favor verbose, or confident-sounding outputs if they are wrong.
- Pay attention to the format of final answer (<answer>...</answer>).

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Some analysis</analyse>
<choose>A/B</choose>
"""

# ---------------- 中间输出成对比较（INTERMEDIATE_COMPARE_TEMPLATE[MATH]） ----------------
INTERMEDIATE_COMPARE_TEMPLATE = """
You are comparing two intermediate outputs from an agent in a multi-agent mathematical reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

Prefer the output that:
- Contains mathematically accurate reasoning (no factual/logical errors)
- Clearly states intermediate results or assumptions
- Avoids misleading statements or ambiguous conclusions
- Provides enough detail for downstream agents to verify or build upon

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
"""

# ---------------- 最终答案成对比较（FINAL_ANSWER_COMPARE_TEMPLATE[MATH]） ----------------
FINAL_ANSWER_COMPARE_TEMPLATE = """
You are comparing two FINAL answers to a math problem.
Your SOLE task is to determine which answer is more likely to be **mathematically correct**.

Do NOT favor longer, more detailed, or better-formatted responses unless they are also correct.
If one answer is clearly correct and the other is wrong, choose the correct one—even if its reasoning is minimal.
If both seem plausible, prefer the one with clearer, error-free reasoning.

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Respond with only "A" or "B".
"""

# ------------------- 优化目标约束（OPTIMIZATION_REQUIREMENTS[MATH]） -------------------
OPTIMIZATION_REQUIREMENT = (
    "The agent must produce a clear, step-by-step reasoning process. "
    "Crucially, the **final answer must appear ONLY once**, enclosed strictly between <answer> and </answer> tags, "
    "with **no additional text, explanation, unit, or reasoning inside these tags**. "
    "The content within `<answer>...</answer>` must be the minimal, canonical mathematical answer."
)


def role_description(role: str) -> str:
    """取角色描述；未知角色显式报错（不静默回退通用描述——那会掩盖建图配置错误）。"""
    try:
        return ROLE_DESCRIPTIONS[role]
    except KeyError:
        raise KeyError(
            f"MASPO(MATH) 不认识角色 {role!r}；已 vendored 的角色: {sorted(ROLE_DESCRIPTIONS)}"
        ) from None


def seed_template(role: str) -> str:
    """取角色的种子提示模板；未知角色显式报错。"""
    try:
        return AGENT_TEMPLATES[role]
    except KeyError:
        raise KeyError(
            f"MASPO(MATH) 没有角色 {role!r} 的种子模板；已 vendored 的角色: {sorted(AGENT_TEMPLATES)}"
        ) from None
