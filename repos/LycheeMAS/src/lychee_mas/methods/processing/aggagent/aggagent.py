"""aggregator/aggagent —— agentic 聚合（接入 princeton-pli/AggAgent, Apache-2.0）。

AggAgent（arXiv:2604.11753, COLM 2026）把「K 条并行轨迹 → 一个答案」的归约本身做成
agentic 任务：一个聚合 agent 把 K 条轨迹当作可检索环境，用四个轻量工具跨轨迹推理：

    get_solution(trajectory_id?)  取轨迹最后一步的最终解（不传 = 全部 K 个解）
    search_trajectory(traj, q)    单条轨迹内关键词检索 top-k 步（ROUGE-L 排序，
                                  可 role='tool' 过滤到真实工具观测）
    get_segment(traj, s, e)       细读连续段（≤5 步）验证上下文
    finish(solution, reason)      提交 <explanation>/<answer> 或长报告格式终答

核心原则（写进 system prompt）：**数证据不数轨迹数**——单条带工具观测支撑的证据
强于多条仅靠推理的多数一致；冲突时信工具观测而非 agent 推理。

接入形态（对齐本框架聚合器类别）:
- 协议: ``TrajectoryAggregator.aggregate(list[Trajectory]) -> Answer``，是
  ``processor/parallel`` 的可插拔归约策略——配置 ``aggregator: aggagent`` 即替换
  self_consistency，同实验设置直接对拍；
- 轨迹适配: 原版消费「消息步 dict 列表」；本框架 ``Trajectory`` 按 (round, sender)
  稳定排序折成步列表（`trajectory_to_steps`），role 直通 ``Message.role``——工具观测
  消息请标 role="tool"（运行时录制的约定），聚合检索的 role 过滤语义即对齐；
- 聚合模型: 需支持 tool-calling（OpenAI 兼容端点；vLLM 开 --enable-auto-tool-choice +
  --tool-call-parser）。LLM 走构造注入的 ``client``（AggClient 协议）或
  model/api_base 走内置 HttpAggClient（client.py）；未配置任一 → 显式 ValueError。
- 失败语义: 不静默兜底——空输入/缺题面/缺 LLM 配置/聚合未产出合法解均显式报错。

子模块: tools.py（四工具纯 stdlib 移植）、prompts.py（原版模板 verbatim）、
engine.py（原版循环移植）、client.py（客户端接缝 + HttpAggClient）。均为纯标准库，
离线 import 零重依赖。许可证与修改说明见 NOTICE.md。
"""

import re
from typing import Any

from ....core.registry import REGISTRY
from ....core.types import Answer, Trajectory
from .client import AggClient, HttpAggClient
from .engine import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_ITERATIONS,
    LONG_FORM_TASKS,
    run_aggregation,
)

_XML_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_QWEN_ANSWER_RE = re.compile(r"Exact Answer:\s*(\S.*)", re.DOTALL)

@REGISTRY.register("aggregator", "aggagent")
class AggAgentAggregator:
    """agentic 聚合（AggAgent）：K 条轨迹 → 检索式推理 → 一个 Answer。

    协议: ``TrajectoryAggregator.aggregate(list[Trajectory]) -> Answer``。
    构造参数: LLM 来源二选一——注入 ``client``（AggClient 协议，mock/脚本化用）
    或给 ``model`` + ``api_base``（走内置 HttpAggClient，真实实验用；api_key 留空
    = vLLM 惯例 Bearer EMPTY）。``question`` 缺省取 ``trajectories[0].task_id``。
    其余超参语义对齐原版 AggAgent（task/model 决定 finish 输出变体）。
    """

    name = "aggagent"

    def __init__(
        self,
        client: AggClient | None = None,
        question: str | None = None,
        task: str = "",
        model: str = "",
        api_base: str | None = None,
        api_key: str = "",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tries: int = 5,
        timeout: float = 60.0,
    ):
        self.client = client
        self.question = question
        self.task = task
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_context_tokens = int(max_context_tokens)
        self.max_iterations = int(max_iterations)
        self.max_tries = int(max_tries)
        self.timeout = float(timeout)

    # -- 组件协议 ------------------------------------------------------------

    def aggregate(self, trajectories: list[Trajectory]) -> Answer:
        """聚合 K 条并行轨迹为一个 Answer（同步；见处理层 base.py 的协议约定）。"""
        if not trajectories:
            raise ValueError("aggagent: 需要 ≥1 条轨迹参与聚合，得到空列表")
        if not all(isinstance(t, Trajectory) for t in trajectories):
            got = type(trajectories[0]).__name__
            raise TypeError(f"aggagent: 输入必须是 list[Trajectory]，得到含 {got} 的列表")

        question = self.question if self.question is not None else trajectories[0].task_id
        if not (question or "").strip():
            raise ValueError(
                "aggagent: 缺少题面——trajectories[0].task_id 为空且构造时未显式给 question=")

        variant = "long_form" if self.task in LONG_FORM_TASKS else ""
        client = self._resolve_client()
        traj_steps = [trajectory_to_steps(t) for t in trajectories]

        out = run_aggregation(
            question,
            traj_steps,
            client,
            task=self.task,
            model=self.model,
            max_context_tokens=self.max_context_tokens,
            max_iterations=self.max_iterations,
        )

        result = out["result"]
        if result is None or not isinstance(result, dict) or "solution" not in result:
            stats = out.get("stats", {})
            raise RuntimeError(
                f"aggagent 聚合失败: {out.get('error') or '模型未产出合法 solution'} "
                f"(iterations={stats.get('iterations')}, "
                f"tool_calls={stats.get('tool_calls')})")

        # finish 工具已按 variant 校验过格式，抽不到 = 上游校验与抽取正则漂移 → 显式 raise
        content = extract_answer_text(result["solution"], variant=variant, model=self.model)

        return Answer(
            content=content,
            source=self.name,
            meta={
                "variant": variant,
                "reason": result.get("reason", ""),
                "solution": result["solution"],
                "stats": out.get("stats", {}),
                "n_trajectories": len(trajectories),
                "question": question,
            },
        )

    # -- 内部 ----------------------------------------------------------------

    def _resolve_client(self) -> AggClient:
        if self.client is not None:
            if not hasattr(self.client, "complete"):
                raise TypeError(
                    f"aggagent: 注入的 client 须实现 complete(messages, tools)->dict|str，"
                    f"得到 {type(self.client).__name__}")
            return self.client
        if self.model and self.api_base:
            return HttpAggClient(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                max_tries=self.max_tries,
                timeout=self.timeout,
            )
        raise ValueError(
            "aggagent: 未配置聚合模型——二选一：注入 client=（AggClient 协议，mock 用），"
            "或构造给 model+api_base（真实实验，configs/aggregator/aggagent.yaml）")

def trajectory_to_steps(t: Trajectory) -> list[dict]:
    """Trajectory → 原版「消息步」列表（聚合 agent 的检索粒度 = 每条消息一步）。

    确定性映射（与 self_consistency 的输入宽容无关，本组件要求明确结构）：
    1. 按 (round, sender) 升序稳定排序（sender 保证同轮多条消息的分组确定性；
       无 messages 的轨迹 → 空步列表，get_solution 对其返回空 content）；
    2. ``role`` 直通 ``Message.role``（"tool" = 工具观测，其余默认 "assistant"）；
    3. ``meta["reasoning"]/["reasoning_content"]`` → reasoning_content；
       ``meta["tool_calls"]``（OpenAI 形状）原样透传。
    末条消息即「最终解」（对齐原版 get_solution 取 last step 的语义），因此实验
    回放应保证终答落在最后一条 message（框架 Trajectory.final_answer 不被消费）。
    """
    steps: list[dict] = []
    for m in sorted(t.messages, key=lambda x: (x.round, x.sender)):
        step: dict[str, Any] = {"role": m.role or "assistant", "content": m.content or ""}
        reasoning = m.meta.get("reasoning") or m.meta.get("reasoning_content")
        if reasoning:
            step["reasoning_content"] = str(reasoning)
        if m.meta.get("tool_calls"):
            step["tool_calls"] = m.meta["tool_calls"]
        steps.append(step)
    return steps


def extract_answer_text(solution: str, *, variant: str, model: str) -> str:
    """从 finish 的 solution 里抽 Answer.content（与 finish 工具校验用的同款正则）。

    variant="long_form" → 整篇报告即答案；model 名含 qwen → "Exact Answer:" 行；
    否则取 <answer>...</answer> 内文。抽不到 = 上游校验不该漏过的异常 → 显式 raise。
    """
    if variant == "long_form":
        return solution.strip()
    if "qwen" in model.lower():
        match = _QWEN_ANSWER_RE.search(solution)
    else:
        match = _XML_ANSWER_RE.search(solution)
    if match is None or not match.group(1).strip():
        raise RuntimeError(f"aggagent: 无法从 solution 抽取答案文本（variant={variant!r}）")
    return match.group(1).strip()

