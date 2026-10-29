# MLC 角色分化诊断（阶段一）

定位：**MLC 原型和端到端实验基线，不是已复现论文效果。** 本报告只回答训练后是否观察到角色/输出分化，不据此判断准确率优劣。

## 设置

- GSM8K 训练样本：32 条，固定 seed 0；两轮协作。
- 诊断评测：与 baseline 相同的前 10 个 case IDs，10 条、两轮。
- 基座：Qwen3-0.6B 冻结；每个 agent 训练 rank-8 输出 adapter。
- 原始逐 case / 逐轮诊断：`runs/benchmarks/Qwen3-0.6B/mlc_diag32_10/gsm8k/role_diagnostics.json`。
- 自动汇总：`runs/benchmarks/Qwen3-0.6B/mlc_diag32_10/gsm8k/role_diagnostics.md`。

## 观察

| 诊断项 | 结果 |
|---|---|
| Round 1 角色选择 | agent 0: alternative_checker 10/10；agent 1、2: consensus_resolver 各 10/10 |
| Round 2 角色选择 | 与 Round 1 相同，角色没有随题目改变 |
| Official agent accuracy | agent 0/1/2: 70% / 70% / 70%（各 20 次回答） |
| Strict numeric agent accuracy | 55% / 60% / 60%（各 20 次回答） |
| 平均轮内两两输出距离 | 0.286（token 序列归一化编辑距离） |
| Adapter 参数更新范数 | agent 0/1/2: 0.1623 / 0.1523 / 0.1164；相对更新 10.31% / 9.65% / 7.19% |

## 结论

有有限的 agent 间分化：不同 agent 稳定选择了不同角色，输出文本不完全相同，且三个 adapter 均发生非零更新。但角色选择完全固定在 agent 身份上，没有观察到输入条件下的动态分配；agent 准确率也没有形成清晰差异。因此只能说**观察到角色分化信号**，不能说模型学会了按题目自动分工或有效协作，更不能等同于复现论文效果。

10 个 case 的准确率仅是诊断辅助，不用于方法比较结论。扩大到 baseline 相同的 100 个 case 后再报告初步对比。

## 100-case 阶段状态

已确认 `langgraph_baseline_single` 与 `langgraph_baseline_chain3` 都包含同一组 100 个唯一 case ID。随后使用本 checkpoint 启动了 100-case MLC 推理，但由于当前 runner 对 100 个 case 执行 600 次顺序生成，运行时间过长而被中断；`mlc_diag32_100` 没有写出完整 `outputs.jsonl`、`metrics.json` 或比较表。因此本报告不包含 100-case 效果结论，旧的 10-case 结果也不能替代它。
