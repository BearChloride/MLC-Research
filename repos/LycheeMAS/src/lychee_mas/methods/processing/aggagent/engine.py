"""聚合 agent 循环 —— 移植自 princeton-pli/AggAgent（Apache-2.0）agent.py 的 ``_run``。

出处: https://github.com/princeton-pli/AggAgent（commit 9638f7d, 2026-04-29）
原文件: aggagent/agent.py。循环结构与原版 `_run` 逐段对应（可对照审计），唯一改动：
``call_server``（litellm 完成调用）替换为注入的 **client 接缝**（`client.complete`
协议见 client.py）——mock 测试注入脚本化客户端，真实实验注入 HttpAggClient（或
同协议的自研客户端）。语义全部保留：≤1 个 tool_call / 轮、错误哨兵字符串处理、
finish 命中即返回、上下文预算超限走 FINAL_MESSAGE 强制 finish、统计字段命名
与原版 stats 一致（iterations / server_errors / tool_call_errors / tool_calls /
context_limit_reached / token_usage_each_step）。

``run_aggregation`` 返回 ``{"result", "error", "stats", "messages"}``：
- result: ``{"solution": str, "reason": str}``（finish 成功）或 None；
- error: 失败诊断（None=成功），调用方按显式报错规则决定 raise。
许可证与修改说明见 NOTICE.md。
"""
from __future__ import annotations

import json
from typing import Any

from . import prompts
from .client import sanitize_tool_name
from .tools import (
    FinishTool,
    GetSegmentTool,
    GetSolutionTool,
    SearchTrajectoriesTool,
    _count_tokens_approx,
    format_metadata,
)

# 长报告式任务（原版 LONG_FORM_TASKS 逐字保留：healthbench / researchrubrics 走 report 变体）
LONG_FORM_TASKS = {"healthbench", "researchrubrics"}

DEFAULT_MAX_CONTEXT_TOKENS = 100 * 1024  # 原版默认 100K 上下文字符预算
DEFAULT_MAX_ITERATIONS = 100             # 原版 MAX_ITERATIONS


def variant_for_task(task: str) -> str:
    """finish/输出变体：长报告任务 → "long_form"，否则 ""（短答 XML）。"""
    return "long_form" if task in LONG_FORM_TASKS else ""


def _build_tools(task: str, model: str) -> tuple[dict[str, Any], list[dict]]:
    """构造四工具 + 其 OpenAI schema 定义（顺序与原版一致，finish 附 variant/model）。"""
    variant = variant_for_task(task)
    tools = [
        GetSolutionTool(),
        SearchTrajectoriesTool(),
        GetSegmentTool(),
        FinishTool(variant=variant, model=model),
    ]
    tool_map = {t.name: t for t in tools}
    tool_description = [tool_map[t].get_tool_definitions() for t in tool_map]
    return tool_map, tool_description


def run_aggregation(
    question: str,
    trajectories: list[list[dict]],
    client: Any,
    *,
    task: str = "",
    model: str = "",
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict:
    """跑一次聚合 agent 会话：读 K 条轨迹（消息步列表）→ 检索工具 → finish。

    参数对齐原版 AggAgent.run/_run：question=任务题面；trajectories=每元素为一条
    轨迹的「消息步 dict 列表」（本框架由 Trajectory 适配而来）；client=满足
    AggClient 协议的 complete(messages, tools)。task/model 决定 prompt 与 finish
    变体（qwen → "Explanation:/Exact Answer:"；long_form 任务 → report 格式）。

    返回 ``{"result", "error", "stats", "messages"}``，见模块 docstring。
    """
    tool_map, tool_description = _build_tools(task, model)
    metadata = format_metadata(trajectories)
    n_traj = len(trajectories)
    variant = variant_for_task(task)

    if variant == "long_form":
        system_prompt = prompts.SYSTEM_PROMPT_AGGAGENT_REPORT
        user_prompt = prompts.USER_PROMPT_AGGAGENT_REPORT.format(
            question=question, metadata=metadata, traj_N=n_traj)
    elif "qwen" in model.lower():
        system_prompt = prompts.SYSTEM_PROMPT_AGGAGENT_QWEN
        user_prompt = prompts.USER_PROMPT_AGGAGENT.format(
            question=question, metadata=metadata, traj_N=n_traj)
    else:
        system_prompt = prompts.SYSTEM_PROMPT_AGGAGENT
        user_prompt = prompts.USER_PROMPT_AGGAGENT.format(
            question=question, metadata=metadata, traj_N=n_traj)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    full_messages: list[dict] = list(messages)

    stats: dict[str, Any] = {
        "iterations": 0,
        "server_errors": 0,
        "tool_call_errors": 0,
        "tool_calls": {},
        "context_limit_reached": False,
        "token_usage_each_step": [],
    }
    error: str | None = None

    def count_tokens(messages_: list[dict]) -> int:
        # 原版 AggAgent.count_tokens_approx = 消息 token + 工具 schema token
        tool_chars = len(json.dumps(tool_description, ensure_ascii=False))
        return _count_tokens_approx(messages_) + int(tool_chars / 4.0)

    def store_token_usage(response, iteration: int) -> None:
        if isinstance(response, dict) and response.get("usage"):
            usage = response["usage"]
            entry = {
                "iteration": iteration,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            }
        else:
            entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
        stats["token_usage_each_step"].append(entry)

    def run_tool_calls(response: dict) -> dict | None:
        """执行 assistant 步里的（首个）tool_call；finish 成功返回 result dict。

        未知工具名 → 错误串作为 tool 观测（原版 custom_call_tool 行为，不计异常）；
        arguments 非法 JSON → 报错串并计入 tool_call_errors（原版逐字保留）。
        """
        for tool_call in response.get("tool_calls", [])[:1]:
            tool_name = sanitize_tool_name(tool_call["function"]["name"])
            stats["tool_calls"][tool_name] = stats["tool_calls"].get(tool_name, 0) + 1
            tool = tool_map.get(tool_name)
            tool_result = None
            if tool is None:
                tool_result_str = f"Error: Tool {tool_name} not found"
            else:
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    tool_result = tool.call(tool_args, trajectories=trajectories)
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                except Exception:
                    stats["tool_call_errors"] += 1
                    tool_result_str = (
                        'Error: Tool call is not a valid JSON. Tool call must contain '
                        'a valid "name" and "arguments" field')
            if tool_name == "finish" and isinstance(tool_result, dict):
                return tool_result
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": tool_name,
                "content": tool_result_str,
            }
            messages.append(tool_msg)
            full_messages.append(tool_msg)
        return None

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        stats["iterations"] = iteration

        response = client.complete(messages, tool_description)
        store_token_usage(response, iteration)

        if response == "ContextLengthError":
            stats["context_limit_reached"] = True
            error = "上下文超限且服务端拒绝本次请求（ContextLengthError）"
            break
        if isinstance(response, str):
            stats["server_errors"] += 1
            err_msg = {"role": "assistant", "content": response}
            messages.append(err_msg)
            full_messages.append(err_msg)
            continue

        messages.append(response)
        full_messages.append(response)

        result = run_tool_calls(response)
        if result is not None:
            return {"result": result, "error": None, "stats": stats, "messages": full_messages}

        # 上下文预算（原版：每轮工具执行后检查，超限则 FINAL_MESSAGE 强制 finish）
        if count_tokens(messages) > max_context_tokens:
            stats["context_limit_reached"] = True
            final_msg = {"role": "user", "content": prompts.FINAL_MESSAGE}
            messages.append(final_msg)
            full_messages.append(final_msg)
            finish_only = [tool_map["finish"].get_tool_definitions()]

            response = client.complete(messages, finish_only)
            store_token_usage(response, iteration)
            if isinstance(response, str):
                stats["server_errors"] += 1
                err_msg = {"role": "assistant", "content": response}
                messages.append(err_msg)
                full_messages.append(err_msg)
                error = "上下文超限后的强制 finish 也失败"
                break

            messages.append(response)
            full_messages.append(response)
            result = run_tool_calls(response)
            if result is not None:
                return {"result": result, "error": None, "stats": stats, "messages": full_messages}
            error = "上下文超限后模型未调用 finish"
            break

    if error is None:
        error = f"聚合 agent 迭代预算 {max_iterations} 耗尽仍未 finish"
    return {"result": None, "error": error, "stats": stats, "messages": full_messages}


__all__ = [
    "run_aggregation",
    "variant_for_task",
    "LONG_FORM_TASKS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_MAX_ITERATIONS",
]
