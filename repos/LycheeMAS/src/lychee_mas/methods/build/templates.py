"""按任务定制的 MAS 队伍模板（网络定义，迁移自 agents/roles.py）。

设计：
- Role  = 一个角色（agent 名 + system prompt + 可选 model_id）。纯数据，不依赖 backend/runtime。
- TEAMS = 命名的队伍 profile（每个 = 有序的 Role 列表）。同一 profile 可被多个 task 复用。

各 prompt 的 # INPUT 段引用「上一个 agent 输入」的来源标志 = `PREV_OUTPUT_HEADER`，由 NL 通道注入。
本文件直接 import 该常量并用它拼 prompt，消除重复字符串（全包唯一来源在 memory/channels/nl.py）。

⚠️ 约定（与 runtime 后端耦合，改队伍时必须遵守）：
  1) 每支队伍的【最后一个角色】负责用 'APPROVE: <final answer>' 收尾——后端据此终止 + 抽答案。
  2) 队伍可有任意数量角色；终止轮数按实际 agent 数计算。
  3) 每个 agent 独占一个绑定到自身 role 的注入 client（路由器按 role 条件化），共享 backend + ctx。

注册 `topology_generator/static`：按 team 名产出 AgentSpec 列表（封装成 MASGraph 顺序链）。
prompt 是给模型的指令，保持英文；注释用中文。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ...core.types import AgentSpec
from ..memory.channels.nl import PREV_OUTPUT_HEADER  # 来源标志的唯一来源


@dataclass
class Role:
    name: str  # 角色名 = agent 名（路由条件化 / ctx 登记，需合法标识符、队内唯一）
    system: str  # 该角色的 system prompt
    model_id: Optional[str] = None  # 可选：异构模型时给不同 model_id；None=用队伍统一 model_id
    # 可选：给 selector 选发言者用的一句话简介；None=从 system 取首句
    description: Optional[str] = None
    agent_type: str = "assistant"  # assistant/coder/file_surfer/web_surfer/computer_terminal
    tools: Optional[list[str]] = None  # 普通 AssistantAgent function tools（字符串由 runtime 解析）
    meta: Optional[dict] = None


# ---- 基础通用三件套（默认队伍用；其它队伍也可复用这些文本）----
ROLE_SYSTEM = {
    "manager": (
        "You are the MANAGER. Read the task and any provided memory. Decompose it into "
        "a short, concrete plan (2-4 steps) for the worker. Do not solve it yourself."),
    "worker": (
        "You are the WORKER. Follow the manager's plan and any provided memory to solve "
        "the task. Show the key steps briefly, then state your candidate answer."),
    "verifier": (
        "You are the VERIFIER. Check the worker's answer against the task and memory. "
        "If correct, reply exactly 'APPROVE: <final answer>'. If not, explain the fix in "
        "one line so the worker can retry."),
}


# --- Aime ---
# 各 prompt # INPUT 段引用的来源标志由 NL 通道注入，与 PREV_OUTPUT_HEADER 保持一致（同源，无需手抄
# ）。
AIME_ANALYST = """\
# ROLE
You are the Problem Analyst for AIME math problems. Parse the problem and produce a short
solving plan. You do NOT compute the final answer.

# NOTE
Every AIME answer is a single INTEGER between 0 and 999.

# INPUT
The math problem to solve (the user's question).

# RULES
- Restate the problem; list ASKED_FOR, GIVEN_CONDITIONS, implicit constraints.
- Propose 1-2 concrete strategies. Do NOT solve.

# OUTPUT
PROBLEM_RESTATEMENT / ASKED_FOR / GIVEN_CONDITIONS / STRATEGIES
"""

AIME_SOLVER = f"""\
# ROLE
You are the Solver. Give a full chain-of-thought solution WITHOUT running code.

# INPUT
The Analyst's plan, given to you as a note headed "{PREV_OUTPUT_HEADER}".
Follow that plan together with the original problem.

# RULES
- Show every non-trivial step.
- The AIME answer is a SINGLE INTEGER between 0 and 999. Compute it all the way to that
  integer: evaluate EVERY expression (binomial, factorial, fraction, radical, power, sum) to
  a concrete number. NEVER leave the answer as an unevaluated expression and NEVER as a
  decimal.
- If your result is not an integer in [0, 999], you made an error — recheck before answering.
- Do NOT declare the official answer (the Verifier does).

# OUTPUT
SOLUTION_STEPS / SOLVER_ANSWER: \\boxed{{<integer 0-999>}} /
KEY_ASSUMPTIONS: [steps a checker should scrutinize]
"""

AIME_VERIFIER = f"""\
# ROLE
You are the Verifier and the ONLY agent that declares the final answer.

# INPUT
The Solver's full solution and its SOLVER_ANSWER, given to you as a note headed
"{PREV_OUTPUT_HEADER}". Follow that solution together with the original problem.

# HARD RULES
1. INTEGER ONLY: the final answer MUST be a single INTEGER between 0 and 999 inside
   \\boxed{{}}. NEVER approve an unevaluated expression or a decimal — if SOLVER_ANSWER is not
   a concrete integer, compute it to one before approving.
2. NO ASSUMPTIONS: use only values given in the problem / produced by the Solver.
3. INDEPENDENT CHECK: re-derive the key step(s) yourself, with a different method if possible.
   - DEFAULT TO KEEPING SOLVER_ANSWER: approve it unless you find a SPECIFIC, concrete error.
   - Only change the answer if you found such an error AND can compute the correct integer
     with confidence — never replace it with a guess.

# OUTPUT
A one-line justification, then on the FINAL line exactly (a bare integer inside \\boxed):
APPROVE: \\boxed{{<integer 0-999>}}
"""


# ---- 命名队伍 profile（每个 profile 的最后一个角色必须用 APPROVE 收尾）----
TEAMS: dict[str, List[Role]] = {
    # 单模型 baseline：1 个 agent 直接作答（配 method=none 即「单模型、无记忆、单次推理」对照）
    "single": [
        Role("solver",
             "You solve the given problem on your own. Think briefly, then give the final "
             "answer. On the FINAL line output exactly 'APPROVE: <final answer>'. For math "
             "problems wrap the answer in \\boxed{}, e.g. 'APPROVE: \\boxed{204}'."),
    ],

    # 默认：通用 manager -> worker -> verifier
    "default": [
        Role("manager", ROLE_SYSTEM["manager"]),
        Role("worker", ROLE_SYSTEM["worker"]),
        Role("verifier", ROLE_SYSTEM["verifier"]),
    ],

    # Aime（无 Executor，故不设 coder）：分析 -> 求解 -> 独立复核
    "aime": [
        Role("analyst", AIME_ANALYST),
        Role("solver", AIME_SOLVER),
        Role("verifier", AIME_VERIFIER),
    ],

    # 推理极（gsm8k / aime ...）：规划 -> 求解 -> 复核，强调分步推导与重算
    "reason": [
        Role("planner",
             "You are the PLANNER. Read the problem and lay out a concise step-by-step "
             "plan / decomposition for solving it. Do NOT give the final number yet; hand "
             "the plan to the solver."),
        Role("solver",
             "You are the SOLVER. Execute the plan, doing the arithmetic/derivation step "
             "by step. Then state your candidate final answer clearly, e.g. 'Answer: 42'."),
        Role("verifier",
             "You are the VERIFIER. Re-check the solver's reasoning and recompute the key "
             "step independently. If it is correct, reply exactly 'APPROVE: <final answer>'. "
             "If wrong, point out the error in one line so the solver can retry."),
    ],

    # 事实/精确极（medqa / arc / openbookqa ...）：2 角色，事实型问答不需重规划
    "fact": [
        Role("solver",
             "You are the SOLVER. This is a factual / multiple-choice question. Use the "
             "question and any provided memory/notes to pick the single best answer. Give a "
             "one-line justification, then state the answer as "
             "'Answer: <option letter and text>'."),
        Role("verifier",
             "You are the VERIFIER. Check the chosen option against the question and memory. "
             "If correct, reply exactly 'APPROVE: <final answer>'. Otherwise give the correct "
             "option in one line for a retry."),
    ],

    # 长程记忆极（locomo / longmemeval ...）：检索 -> 作答 -> 复核，强调「答案须由记忆支撑」
    "memory": [
        Role("retriever",
             "You are the RETRIEVER. You are given long conversation history / notes as "
             "memory. Extract and quote ONLY the facts from that memory that are relevant to "
             "the question. Do NOT answer the question yet."),
        Role("answerer",
             "You are the ANSWERER. Using the retriever's quoted facts and the question, give "
             "a short, direct answer grounded in the memory. If the memory does not contain "
             "it, say so explicitly."),
        Role("verifier",
             "You are the VERIFIER. Check that the answer is actually supported by the "
             "retrieved memory facts. If correct, reply exactly 'APPROVE: <final answer>'. "
             "Otherwise correct it in one line."),
    ],

    # 代码工具链：Coder 生成代码，ComputerTerminal 在隔离 workspace 中执行。
    "human_eval": [
        Role("Coder", "", agent_type="coder",
             description="Writes and revises code for the programming task."),
        Role("ComputerTerminal", "", agent_type="computer_terminal",
             description="Executes code produced by Coder in the sandbox.",
             meta={"sources": ["Coder"]}),
    ],

    # GAIA/Magentic-One 风格工具链：文件、网页、代码执行都在同一条 autogen/CDM 主链路中。
    "gaia": [
        Role("FileSurfer", "", agent_type="file_surfer",
             description="Reads and navigates files available in the case workspace."),
        Role("WebSurfer", "", agent_type="web_surfer",
             description="Browses web pages and performs web search when needed."),
        Role("Coder", "", agent_type="coder",
             description="Writes code to analyze files or compute intermediate results."),
        Role("ComputerTerminal", "", agent_type="computer_terminal",
             description="Executes code produced by Coder in the sandbox.",
             meta={"sources": ["Coder"]}),
    ],
}


TEAM_META: dict[str, dict] = {
    "human_eval": {"team_preset": "coder_executor"},
    "gaia": {"team_preset": "magentic_one"},
}


def _role_description(role: Role) -> str:
    """selector 选发言者用的一句话简介：显式给则用之，否则取 system 里第一句非标题行。"""
    if role.description:
        return role.description
    return next((ln.strip() for ln in role.system.splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")), role.name)


def team_to_agentspecs(team: str, model: Optional[str] = None) -> List[AgentSpec]:
    """把命名 profile 解析成 AgentSpec 列表（图节点）。未登记则报错。"""
    if team not in TEAMS:
        raise ValueError(f"unknown team profile {team}; choices: {list(TEAMS)}")
    specs: List[AgentSpec] = []
    for r in TEAMS[team]:
        meta = {"description": _role_description(r), "agent_type": r.agent_type}
        if r.tools:
            meta["tools"] = list(r.tools)
        if r.meta:
            meta.update(r.meta)
        specs.append(AgentSpec(
            name=r.name, role=r.name, system_prompt=r.system,
            model=r.model_id or model,
            tools=list(r.tools or []),
            meta=meta))
    return specs
