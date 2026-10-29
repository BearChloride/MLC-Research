# Qwen3.5-2B GSM8K 基线结果

日期：2026-10-26

## 结论

在相同的 GSM8K 100 题子集（`seed=0`）和相同的 `run_langgraph_baseline.py` 配置下，Qwen3.5-2B 的初步基线为：

- `single`：官方准确率 88/100，严格数值准确率 88/100。
- `chain3`：官方准确率 88/100，严格数值准确率 84/100。

三个模型的对照如下。

## 实验设置

- 数据：GSM8K，`--n 100`，`--sample-seed 0`。
- 三组模型使用同一组 100 个 case index，已核对完全一致。
- 任务参数：`--max-rounds 2`、`--max-new-tokens 1024`、`do_sample=false`、`bfloat16`。
- 设备：GPU 0，NVIDIA L20。
- Qwen3.5-2B 环境：`qwen35`（克隆自 `nhzk`，`transformers 5.2.0`）。
- Qwen3-0.6B / Qwen3-8B 历史结果环境：`nhzk`，`transformers 4.57.1`。

Qwen3.5-2B 的模型类型为 `qwen3_5`，当前 `nhzk` 环境的 `transformers 4.57.1` 不识别该架构，因此本轮未改动历史环境，而是新建 `qwen35` 环境并升级到 `transformers 5.2.0`。

## 准确率对照

| 模型 | single 官方 | single 严格 | chain3 官方 | chain3 严格 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 66% | 65% | 63% | 60% |
| Qwen3.5-2B | 88% | 88% | 88% | 84% |
| Qwen3-8B | 91% | 91% | 92% | 90% |

## 成本指标

### single

| 模型 | 每题调用次数 | 每题输出 token | 每题模型时延 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1.45 | 241.8 | 6.635s |
| Qwen3.5-2B | 1.11 | 467.5 | 13.903s |
| Qwen3-8B | 1.00 | 255.9 | 8.799s |

### chain3

| 模型 | 每题调用次数 | 每题输出 token | 每题模型时延 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 3.00 | 353.7 | 14.212s |
| Qwen3.5-2B | 5.01 | 1384.7 | 41.982s |
| Qwen3-8B | 3.00 | 448.6 | 15.362s |

## 严格口径差异

Qwen3.5-2B 的 `single` 严格数值结果与官方结果一致。

Qwen3.5-2B 的 `chain3` 有 4 题被官方 `exact` 口径判对、严格数值口径判错：

- case 191：gold=5，预测 `0.5`
- case 564：gold=240，预测未抽取到直接数值
- case 635：gold=4，预测未抽取到直接数值
- case 1067：gold=245，预测 `2.45`

因此 `chain3` 的严格数值准确率为 84%，官方口径为 88%。

## 产物

- `runs/benchmarks/Qwen3.5-2B/langgraph_baseline_single/gsm8k/`
- `runs/benchmarks/Qwen3.5-2B/langgraph_baseline_chain3/gsm8k/`
- `runs/benchmarks/Qwen3.5-2B/logs/baseline_single.log`
- `runs/benchmarks/Qwen3.5-2B/logs/baseline_chain3.log`
- `runs/benchmarks/Qwen3.5-2B/audit_exact_scoring.json`
