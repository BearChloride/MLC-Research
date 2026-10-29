# Qwen3.5-2B MLC 三种子结果

日期：2026-10-26

## 结论

Qwen3.5-2B 的 full MLC 三种子运行已完成，官方准确率分别为 91%、85%、86%，三种子均值为 87.3%。与 Qwen3.5-2B 自身基线相比，MLC 没有形成稳定提升：

- 相对 `single` 基线，三种子分别 +3、-3、-2 个百分点。
- 相对 `chain3` 基线，三种子分别 +3、-3、-2 个百分点。
- 所有配对的官方/严格 McNemar 检验 p 值都大于 0.05，不显著。

这与 Qwen3-8B 的 full MLC 结果不同。8B 在相同协议下三种子官方准确率为 97%、96%、97%，相对自身 `single` 基线 +6、+5、+6 个百分点。

## 运行协议

| 项目 | 值 |
|---|---|
| 基座模型 | `/home/med_code/models/Qwen3.5-2B` |
| 训练样本 | GSM8K，每种子 32 题 |
| 评测样本 | 与 8B/0.6B 基线相同的 GSM8K 100 题 |
| 种子 | 101、202、303 |
| 轮次 | 2 |
| 生成上限 | 1024 token |
| 输入截断 | 关闭 |
| 设备 | 物理 GPU 1，进程内 `cuda:0`，bfloat16 |
| 环境 | `qwen35`，`transformers 5.2.0` |

GPU 0 当时被 vLLM 服务占用，因此本轮使用空闲的物理 GPU 1。启动器通过 `tmux` 后台运行，训练 checkpoint 在每个 episode 后原子写入，评测进度按题追加。

## 准确率对照

| 模型与方法 | 官方准确率 | 严格数值准确率 |
|---|---:|---:|
| Qwen3.5-2B `single` 基线 | 88% | 88% |
| Qwen3.5-2B `chain3` 基线 | 88% | 84% |
| Qwen3.5-2B full MLC seed 101 | 91% | 85% |
| Qwen3.5-2B full MLC seed 202 | 85% | 83% |
| Qwen3.5-2B full MLC seed 303 | 86% | 82% |
| Qwen3.5-2B full MLC 三种子均值 | 87.3% | 83.3% |
| Qwen3-8B `single` 基线 | 91% | 91% |
| Qwen3-8B `chain3` 基线 | 92% | 90% |
| Qwen3-8B full MLC seed 101 | 97% | 96% |
| Qwen3-8B full MLC seed 202 | 96% | 95% |
| Qwen3-8B full MLC seed 303 | 97% | 96% |
| Qwen3-8B full MLC 三种子均值 | 96.7% | 95.7% |
| Qwen3-0.6B `single` 基线 | 66% | 65% |
| Qwen3-0.6B `chain3` 基线 | 63% | 60% |

0.6B 的历史 full MLC 使用不同题目子集、三轮和 48 token 上限，不与本协议直接合并，因此只保留其同协议基线作参考。

## 配对检验

Qwen3.5-2B full MLC 与 Qwen3.5-2B 基线的 McNemar 双侧精确检验：

| Seed | vs single 官方 | vs single 严格 | vs chain3 官方 | vs chain3 严格 |
|---:|---:|---:|---:|---:|
| 101 | 0.5488 | 0.5811 | 0.5811 | 1.0000 |
| 202 | 0.5811 | 0.3018 | 0.6072 | 1.0000 |
| 303 | 0.7744 | 0.2101 | 0.7905 | 0.8036 |

所有 p 值均大于 0.05。

## 训练检查

| Seed | Episodes | 平均 team reward | 最后 TD loss | 最后 role grad | 最后 prefix grad |
|---:|---:|---:|---:|---:|---:|
| 101 | 32/32 | 0.875 | 0.1383 | 0.2853 | 0.1699 |
| 202 | 32/32 | 0.844 | 0.1005 | 0.2813 | 0.1808 |
| 303 | 32/32 | 0.812 | 0.1145 | 0.2684 | 0.3661 |

三个 run 均保留 `mlc_state.pt`、`train_log.json`、`case_progress.jsonl`、`comparison.json`、`role_diagnostics.json` 和完整评测产物。

## 成本指标

| Run | 每题调用次数 | 每题输出 token | 每题模型时延 |
|---|---:|---:|---:|
| full MLC seed 101 | 6.0 | 1974.6 | 61.721s |
| full MLC seed 202 | 6.0 | 2105.8 | 66.392s |
| full MLC seed 303 | 6.0 | 1904.4 | 60.913s |

## 产物

- 三种子目录：
  `runs/benchmarks/Qwen3.5-2B/mlc_aligned1024_seed101/gsm8k/`
  `runs/benchmarks/Qwen3.5-2B/mlc_aligned1024_seed202/gsm8k/`
  `runs/benchmarks/Qwen3.5-2B/mlc_aligned1024_seed303/gsm8k/`
- 日志：
  `runs/benchmarks/Qwen3.5-2B/mlc_3seed_logs/`
- 严格评分审计：
  `runs/benchmarks/Qwen3.5-2B/mlc_audit_exact_scoring.json`
- 启动器：
  `scripts/run_mlc_qwen35_2b_3seeds.sh`

若任务意外中断，重新执行 `bash scripts/run_mlc_qwen35_2b_3seeds.sh` 即可：

- 已完成的 seed 会因 `metrics.json` 存在而跳过。
- 未完成的 seed 会因 `mlc_state.pt` 和 `case_progress.jsonl` 存在而自动续跑。
