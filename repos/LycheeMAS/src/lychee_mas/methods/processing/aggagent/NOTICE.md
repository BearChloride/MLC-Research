# NOTICE —— AggAgent 第三方代码移植声明

本目录代码移植自 **princeton-pli/AggAgent**（"Agentic Aggregation for Parallel Scaling
of Long-Horizon Agentic Tasks", arXiv:2604.11753, COLM 2026）：

- 出处: https://github.com/princeton-pli/AggAgent
- 移植基准: commit `9638f7d88aee01eb636c02841e13a05bb2e3c449`（2026-04-29）
- 原许可证: **Apache License 2.0**，全文副本见同目录 `LICENSE-APACHE-2.0.txt`
  （取自参考仓库根目录 `LICENSE`）

## 移植文件与原文件对应关系

| 本仓库文件 | 原仓库文件 | 改动 |
|---|---|---|
| `prompts.py` | `aggagent/prompts.py` | 逐字复制，仅加文件头注释（模板原文不可动） |
| `tools.py` | `aggagent/tools.py` | 结构保留；qwen_agent `BaseTool` 继承 → 纯 Python 类；`rouge_score` → 本地 LCS 版 ROUGE-L recall（详见文件头注释） |
| `engine.py` | `aggagent/agent.py` (`_run`) | 循环结构逐段对应；litellm `call_server` → `client.complete` 注入接缝；返回失败详情替代裸 None |
| `client.py` | `aggagent/agent.py` (`call_server`) | litellm 多后端路由收窄为 OpenAI 兼容 chat/completions（urllib）；重试去掉随机 jitter |
| `__init__.py` | —（接入层） | 适配本框架 `TrajectoryAggregator` 协议 + `Trajectory`→消息步映射，非原仓库内容 |

## 行为差异摘要（实验可复现性注意）

1. 原版聚合模型经 litellm 支持 hosted_vllm / gemini / openai 多路由；本移植仅支持
   OpenAI 兼容 tool-calling 端点（vLLM 需 `--enable-auto-tool-choice` + 对应
   `--tool-call-parser`），且未做 gemini / minimax 等厂商特判。
2. `search_trajectory` 的排序分改为本地 LCS 实现的 ROUGE-L recall（分词≈原版
   默认 tokenizer，无词干化；数值与原版 rouge_score 可能有微小差异）。
3. 重试退避去掉随机 jitter（确定性）；`run()` 的「模型未产出合法 solution」错误
   路径在本框架变为显式 `RuntimeError`（黄金法则 6：无静默兜底）。
4. 轨迹输入为本框架 `Trajectory` 适配层（消息按 (round, sender) 排序为步）；题面
   取 `task_id`。其余超参（max_context_tokens / max_iterations / task / model 变体
   语义）与原版一致。
