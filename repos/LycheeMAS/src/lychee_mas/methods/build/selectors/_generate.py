"""AgentInit `generate` 模式 —— 忠实复现官方「LLM 现场生成角色」半边（M2）。

对齐官方 `AgentInit/agentinit/manager.py::_act` 的 4 状态机（prompt 逐字移植自官方
create_roles_format / check_roles / check_plans / select_group）：

  CreateRoles(生成→格式化) → CheckRoles ⇄ CheckPlans（迭代批判到 "No Suggestions" 共识）
    → 解析 {角色:prompt} + 执行计划 → embedder 嵌入角色 prompt
    → _pareto（相关性 + Vendi 多样性 + 非支配排序，与 pool 模式共享）
    → SelectGroup（LLM 从前沿挑一组）→ list[AgentSpec]

与官方差异（黄金法则 §7/§8）：
  - LLM 走注入的 chat_fn（默认 `_llm.chat`，env 驱动；官方硬编码 achat）。
  - embedder 走 env `LYCHEE_EMBED_MODEL`（官方硬编码 model_path）。
  - 解析用容错正则（官方裸 `raise` / 依赖 pydantic str repr），失败降级到 {'Normal': ...}。
  - torch/transformers/openai/numpy 全惰性；本模块 import 期零重依赖。
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Callable, Optional

from ....core.types import AgentSpec, Budget, BudgetUnit, TaskQuery
from ._llm import ChatUsage

if TYPE_CHECKING:
    from .agentinit import AgentInitSelector

_DEFAULT_ROLE = {"Normal": "You are a helpful assistant."}
_SYSTEM = "You are a helpful assistant."


# ============================ Prompts（逐字移植官方）============================
_CREATE_PROMPT = '''
-----
You are a manager and an expert-level ChatGPT prompt engineer with expertise in multiple fields. Your goal is to break down tasks by creating exactly multiple LLM agents, assign them roles, analyze their dependencies, and provide a detailed execution plan. You should continuously improve the role list and plan based on the suggestions in the History section.

# Question or Task
{context}

# Existing Expert Roles
{existing_roles}

# History
{history}

# Steps
You will come up with solutions for any task or problem by following these steps:
1. You should first understand, analyze, and break down the human's problem/task.
2. According to the problem, create expert roles needed to solve it. Each expert role should have a clear, non-overlapping responsibility. For each new expert role, determine its name, a detailed description of its expertise domain, execution suggestions, and a prompt template.
3. The prompt template MUST follow the format "You are [description], named [name]. Your goal is [goal], and your constraints are [constraints]. You could follow these execution suggestions: [suggestions].".
4. You MUST output each created expert role as a JSON blob with keys `name`, `description`, `suggestions`, `prompt`.
5. Finally, provide a detailed execution plan as a numbered list; each step begins with the list of expert roles involved.

# Suggestions
{suggestions}

# Attention
1. DO NOT answer the task directly. Focus on generating high-performance roles and a detailed plan.
2. If you receive suggestions, evaluate how to enhance the previous role list and plan accordingly.
-----
'''

_FORMAT_EXAMPLE = '''
---
## Selected Roles List:
```
JSON BLOB 1,
JSON BLOB 2
```

## Created Roles List:
```
JSON BLOB 1,
JSON BLOB 2
```

## Execution Plan:
1. [ROLE 1, ROLE 2, ...]: STEP 1
2. [ROLE 1, ROLE 2, ...]: STEP 2

## RoleFeedback
feedback on the historical Role suggestions

## PlanFeedback
feedback on the historical Plan suggestions
---
'''

_FORMATTER_PROMPT = '''
-----
You are a formatting expert. I will provide an agent planner's task execution plan; extract the information and present it EXACTLY in the specified format.
# Content to Format:
{raw_content}

# Format Requirements
1. Organize content into these sections:
   - Selected Roles List (JSON blobs)
   - Created Roles List (JSON blobs)
   - Execution Plan (numbered list; each step begins with the expert roles involved)
   - RoleFeedback (feedback on the historical Role suggestions)
   - PlanFeedback (feedback on the historical Plan suggestions)
2. Your final output should ALWAYS be in the following format:
{format_example}
3. Use '##' for section headers.
4. Each JSON blob contains exactly one expert role with keys "name", "description", "suggestions", "prompt". The prompt field starts with "You are ".
-----
'''

_CHECK_ROLES_PROMPT = '''
-----
You are a ChatGPT executive observer expert skilled in identifying errors in problem-solving plans. Your goal is to check whether the created Expert Roles follow the requirements and give improvement suggestions. Refer to History but do not repeat it.

# Question or Task
{question}

# Selected Roles List
{selected_roles}

# Created Roles List
{created_roles}

# History
{history}

# Steps
Check that each role has a clear, non-duplicate responsibility, a name, a detailed description, suggestions, and a prompt template. Output a summary. If you find errors or have suggestions, state them in the Suggestions section. If there are none, you MUST write 'No Suggestions'.

# Format example
{format_example}

# Attention
DO NOT repeat historical suggestions. DO NOT ask the user any questions.
-----
'''

_CHECK_PLANS_PROMPT = '''
-----
You are a ChatGPT executive observer expert skilled in identifying errors in execution plans. Your goal is to check whether the Execution Plan follows the requirements and give improvement suggestions. Refer to History but do not repeat it.

# Question or Task
{context}

# Role List
{roles}

# Execution Plan
{plan}

# History
{history}

# Steps
Check that the plan solves the problem progressively, each step assigns at least one expert role, and each step states its expected output and the input needed by the next step. Output a summary. If you find errors or have suggestions, state them in the Suggestions section. If there are none, you MUST write 'No Suggestions'.

# Format example
{format_example}

# Attention
DO NOT repeat historical suggestions. DO NOT ask the user any questions.
-----
'''

_CHECK_FORMAT_EXAMPLE = '''
---
## Thought
think about whether there are any errors or suggestions.

## Suggestions
1. ERROR1/SUGGESTION1
2. ERROR2/SUGGESTION2
---
'''

_SELECT_GROUP_PROMPT = '''
You are tasked with selecting the best group of experts to help answer a given question. Consider the relevance and effectiveness of each group's composition.

# Question or Task
{context}

# Groups
{groups}

Follow these steps:
1. Analyze the Question: identify its scope and complexity.
2. Evaluate Each Group: assess the relevance of the roles in each group to the question.
3. Make a Justified Choice: select the most suitable group.

# Attention
1. A larger group is not necessarily better—irrelevant roles can hurt. A diverse group helps when the question needs broad knowledge.
2. The last line of your response MUST be 'Choice: Group X' where X is the number of the selected group.
'''


# ============================ 解析助手 ============================
def _parse_blocks(text: str) -> dict[str, str]:
    """按 '##' 切分成 {标题: 内容}（移植官方 OutputParser.parse_blocks）。"""
    blocks: dict[str, str] = {}
    for block in str(text).split("##"):
        if block.strip() == "":
            continue
        parts = block.split("\n", 1)
        if len(parts) != 2:
            continue
        title, content = parts
        title = title.strip()
        if title.endswith(":"):
            title = title[:-1]
        blocks[title.strip()] = content.strip()
    return blocks


def _strip_code_fence(text: str) -> str:
    m = re.findall(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    return "\n".join(m) if m else text


def _extract_role(blob: str) -> Optional[tuple[str, str]]:
    """从一个 JSON blob 抠 (name, prompt)。先试 json，失败退正则（容错 LLM 花式引号）。"""
    cleaned = blob.replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        data = json.loads(cleaned)
        if data.get("name") and data.get("prompt"):
            return str(data["name"]).strip(), str(data["prompt"]).strip()
    except Exception:
        pass
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', cleaned)
    prompt_m = re.search(r'"prompt"\s*:\s*"([\s\S]*?)"\s*[,}]', cleaned)
    if name_m and prompt_m:
        return name_m.group(1).strip(), prompt_m.group(1).strip()
    return None


def _extract_roles_and_plan(formatted: str) -> dict[str, str]:
    """从格式化响应抽 {name: prompt}，按 Execution Plan 出现顺序排列（对齐官方 manager）。"""
    blocks = _parse_blocks(formatted)
    roles_text = _strip_code_fence(blocks.get("Selected Roles List", "")) + "\n" + \
        _strip_code_fence(blocks.get("Created Roles List", ""))
    roles: dict[str, str] = {}
    for blob in re.findall(r"\{[\s\S]*?\}", roles_text):
        got = _extract_role(blob)
        if got:
            roles[got[0]] = got[1]
    if not roles:
        return {}

    # 按执行计划里角色首次出现的顺序排序（计划解析失败则用原顺序）
    plan_text = blocks.get("Execution Plan", "")
    ordered: dict[str, str] = {}
    for bracket in re.findall(r"\d+\.\s*\**\[(.*?)\]", plan_text):
        for name in roles:
            if name not in ordered and name in bracket:
                ordered[name] = roles[name]
    for name, prompt in roles.items():  # 补上计划没覆盖到的
        ordered.setdefault(name, prompt)
    return ordered


def _get_suggestions(text: str) -> str:
    """从批判响应抽 '## Suggestions' 段；抽不到则用全文（保守，倾向不早停）。"""
    blocks = _parse_blocks(text)
    return blocks.get("Suggestions", str(text)).strip()


def _sanitize(name: str) -> str:
    s = re.sub(r"\W+", "_", name.strip()).strip("_")
    return s or "agent"


# ============================ 状态机各步 ============================
def _create_roles(ask: Callable, context: str, history: str, suggestions: str) -> str:
    raw = ask([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _CREATE_PROMPT.format(
            context=context, existing_roles="[]", history=history, suggestions=suggestions)},
    ])
    fmt = ask([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _FORMATTER_PROMPT.format(
            format_example=_FORMAT_EXAMPLE, raw_content=raw)},
    ])
    return fmt


def _check_roles(ask: Callable, formatted: str, question: str, history: str) -> str:
    blocks = _parse_blocks(formatted)
    prompt = _CHECK_ROLES_PROMPT.format(
        question=question, history=history,
        selected_roles=blocks.get("Selected Roles List", ""),
        created_roles=blocks.get("Created Roles List", ""),
        format_example=_CHECK_FORMAT_EXAMPLE)
    return _get_suggestions(ask([
        {"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]))


def _check_plans(ask: Callable, formatted: str, question: str, history: str) -> str:
    blocks = _parse_blocks(formatted)
    roles = blocks.get("Selected Roles List", "") + blocks.get("Created Roles List", "")
    prompt = _CHECK_PLANS_PROMPT.format(
        context=question, roles=roles, plan=blocks.get("Execution Plan", ""),
        history=history, format_example=_CHECK_FORMAT_EXAMPLE)
    return _get_suggestions(ask([
        {"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]))


def _select_group(ask: Callable, context: str, groups_text: str) -> int:
    rsp = ask([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _SELECT_GROUP_PROMPT.format(context=context, groups=groups_text)}])
    m = re.search(r"Choice:\s*Group\s*(\d+)", rsp)
    return int(m.group(1)) if m else 1


# ============================ 主编排 ============================
def generate_and_select(
    selector: "AgentInitSelector", query: TaskQuery, budget: Optional[Budget]
) -> list[AgentSpec]:
    from ._pareto import (
        cosine_matrix,
        cosine_to_query,
        fast_non_dominated_sort,
        objective_diversity,
        objective_relevance,
    )

    question = query.question or ""
    usage_total = ChatUsage()

    # 解析注入点（测试用 fake；默认走 env 驱动的真实 LLM / embedder）
    chat_fn = selector.cfg.get("chat_fn")
    embed_fn = selector.cfg.get("embed_fn")
    if chat_fn is None:
        from ._llm import chat as _chat
        chat_fn = _chat
    if embed_fn is None:
        from .embedder import HFEmbedder
        embed_fn = HFEmbedder(selector.cfg.get("embedder_model")).embed

    def ask(messages: list[dict]) -> str:
        out = chat_fn(messages)
        if isinstance(out, tuple):  # (text, ChatUsage)
            text, u = out
            usage_total.prompt_tokens += u.prompt_tokens
            usage_total.completion_tokens += u.completion_tokens
            return text
        return out  # fake chat_fn 可直接返回 str

    def _finish(specs: list[AgentSpec]) -> list[AgentSpec]:
        # #2：整轮生成开销只记一次到 selector（不逐 spec 重复），供实验取用。
        selector.last_gen_tokens = usage_total.total
        return specs

    # #1：token 预算——仅认 token 单位，给「角色生成的多轮迭代」封顶（calls/usd/None 不限）。
    token_budget = None
    if budget is not None and budget.unit == BudgetUnit.TOKENS and budget.limit > 0:
        token_budget = int(budget.limit)

    # --- 共识循环（对齐 manager._act：含 RoleFeedback/PlanFeedback 双向反馈，#7）---
    num_steps = int(selector.cfg.get("critique_rounds", 3))
    roles_plan, suggestions = "", ""
    suggestions_roles, suggestions_plan = "", ""
    consensus, steps = False, 0
    while not consensus and steps < num_steps:
        # #1：预算耗尽则不再开新一轮生成（第 0 轮总会先跑，保证至少生成一次角色）。
        if steps > 0 and token_budget is not None and usage_total.total >= token_budget:
            break
        roles_plan = _create_roles(ask, question, history=roles_plan, suggestions=suggestions)
        if "\n" not in roles_plan:
            return _finish([_as_spec("Normal", _DEFAULT_ROLE["Normal"], None, set())])
        if steps == num_steps - 1:
            break
        # #7：把创建者对上一轮批评的反馈（官方 RoleFeedback/PlanFeedback）注入批判者 history。
        fb = _parse_blocks(roles_plan)
        history_roles = (f"## Role Suggestions\n{suggestions_roles}\n\n"
                         f"## Feedback\n{fb.get('RoleFeedback', '')}")
        sr = _check_roles(ask, roles_plan, question, history=history_roles)
        suggestions_roles += "\n" + sr
        # #7 bugfix：官方此处误用 suggestions_roles，这里改回 suggestions_plan（计划批判用计划历史）。
        history_plan = (f"## Plan Suggestions\n{suggestions_plan}\n\n"
                        f"## Feedback\n{fb.get('PlanFeedback', '')}")
        sp = _check_plans(ask, roles_plan, question, history=history_plan)
        suggestions_plan += "\n" + sp
        suggestions = f"## Role Suggestions\n{sr}\n\n## Plan Suggestions\n{sp}"
        if "No Suggestions" in suggestions_roles and "No Suggestions" in suggestions_plan:
            consensus = True
        steps += 1

    roles = _extract_roles_and_plan(roles_plan)
    if not roles:
        return _finish([_as_spec("Normal", _DEFAULT_ROLE["Normal"], None, set())])

    # --- 嵌入 + Pareto 选择（与 pool 模式共享 _pareto）---
    names = list(roles.keys())
    prompts = [roles[n] for n in names]
    embeddings = embed_fn(prompts)
    sim_matrix = cosine_matrix(embeddings)
    query_sims = cosine_to_query(embed_fn([question])[0], embeddings)

    n = len(names)
    hi = min(selector.max_roles, n)
    lo = max(1, min(selector.min_roles, n))
    from itertools import combinations
    groups = [g for k in range(lo, hi + 1) for g in combinations(range(n), k)]
    objectives = [
        (objective_relevance(g, query_sims), objective_diversity(g, sim_matrix)) for g in groups
    ]
    front = fast_non_dominated_sort(objectives)[0]

    # --- SelectGroup：LLM 从前沿挑一组 ---
    groups_text = ""
    for i, gi in enumerate(front):
        groups_text += f"Group {i + 1}:\n"
        for idx in groups[gi]:
            groups_text += f"Role: {names[idx]}, Prompt: {prompts[idx]}\n"
    choice = _select_group(ask, question, groups_text)
    gi = front[max(0, min(choice - 1, len(front) - 1))]  # 越界钳制

    chosen = groups[gi]
    seen: set[str] = set()
    return _finish([_as_spec(names[i], prompts[i], embeddings[i], seen) for i in chosen])


def _as_spec(name: str, prompt: str, vec, seen: set[str]) -> AgentSpec:
    # #4：sanitize 后若与已选名冲突，加 _2/_3… 后缀保证 AgentSpec.name 队内唯一（runtime 按 name 路由）。
    base = _sanitize(name)
    final, k = base, 2
    while final in seen:
        final, k = f"{base}_{k}", k + 1
    seen.add(final)
    profile = {"description": prompt}
    if vec is not None:
        profile["vec"] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return AgentSpec(
        name=final, role=final, system_prompt=prompt, profile=profile,
        meta={"role_name": name, "source": "agentinit_generate"})
