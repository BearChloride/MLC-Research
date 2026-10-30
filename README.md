# MLC-Research

本仓库用于复现与适配论文 *Advancing Collaborative Debates with Role Differentiation through Multi-agent Reinforcement Learning*（ACL 2025 Long Papers，编号 2025.acl-long.1105）。

项目基于 LycheeMAS 框架，在 GSM8K 上完成了 MLC 的角色编码、soft prefix、Q-mixing、角色分化损失等核心机制，并对比了 Qwen3-8B、Qwen3.5-2B 与 Qwen3-0.6B 的 baseline、三随机种子和消融实验。

## 轻量化设计声明

为在有限显存条件下实现和测试 MLC 的主要机制，本实验采用了轻量化设计：

- 冻结 Qwen 基座模型参数，只训练外部 MLC 网络模块，包括各 agent 的角色网络、soft-prefix 投影、个体 Q 与 role-value heads、mixing network 和 latent heads。
- 使用每个 agent 独立的 GRUCell，根据共享状态和 recurrent state 产生连续高斯角色表示；角色向量经线性映射生成 soft prefix，并拼接到模型输入 embedding 中。
- 状态特征使用冻结模型输入 token embedding 的均值池化，以控制状态编码的计算与显存开销。该表示不保留 token 顺序及 token 间交互。
- 角色分化目标中的互信息和成对差异使用有限样本代理估计；互信息代理由单个状态内三个 agent 的相似度矩阵计算，并非跨样本 batch 估计，也不是论文估计方法的完整复刻。
- 训练使用角色策略、token 策略、团队 TD 价值学习和角色分化目标；通过 teacher-forced token log-prob 计算生成动作的策略梯度。

## 目录结构

```text
MLC-Research/
├── model/                                  # 实验使用的数学推理数据集及原始数据
├── references/                             # 论文及相关参考资料
├── repos/
│   └── LycheeMAS/                          # LycheeMAS 框架及本实验代码
│       ├── src/lychee_mas/
│       │   ├── methods/prerun/mlc/
│       │   │   ├── model.py                # MLC 网络：角色编码、Q/mixing、prefix、latent 模块
│       │   │   └── __init__.py             # MLC 网络配置与模块导出
│       │   └── eval/
│       │       ├── benchmarks/gsm8k.py     # GSM8K 数据加载
│       │       ├── sampling.py             # 固定随机种子抽样
│       │       ├── metrics.py              # 官方评分与结果汇总
│       │       └── audit.py                # 严格数值评分辅助逻辑
│       ├── scripts/
│       │   ├── run_mlc_gsm8k.py            # MLC 训练、评测、消融与结果落盘主程序
│       │   ├── run_langgraph_baseline.py   # single/chain3 baseline runner
│       │   ├── run_mlc_8b_3seeds.sh        # Qwen3-8B 多随机种子实验启动脚本
│       │   ├── run_mlc_8b_ablations_3seeds.sh
│       │   │                               # Qwen3-8B 消融实验启动脚本
│       │   ├── run_mlc_qwen35_2b_3seeds.sh # Qwen3.5-2B 多随机种子实验启动脚本
│       │   ├── run_mlc_qwen35_2b_ablations_3seeds.sh
│       │   │                               # Qwen3.5-2B 消融实验启动脚本
│       │   ├── audit_exact_scoring.py       # 官方与严格评分口径审计
│       │   ├── compare_runs.py              # 逐题配对比较与 McNemar 检验
│       │   └── selfcheck_mlc_math.py        # 数学 benchmark 与评分自检
│       ├── tests/                           # MLC 网络、runner、抽样与评分测试
│       └── docs/reports/                    # 实验阶段报告、结果与诊断文档
└── README.md
```
- `repos/LycheeMAS/`：代码仓库，核心实现位于 `src/lychee_mas/methods/prerun/mlc/`，实验入口位于 `scripts/run_mlc_gsm8k.py`。
- `model/`：实验使用的数学推理原始数据。
- `references/`：论文 PDF。

## 不包含的内容

- `models/`：模型权重（ Qwen3-8B、Qwen3.5-2B 与 Qwen3-0.6B ）不上传，请按环境文档从 ModelScope 或 Hugging Face 获取。
- `downloads/`：CUDA PyTorch wheel 等大文件不上传。
- `runs/`：训练与评测产物不上传，可由仓库脚本重新生成。

## 常用命令

```bash
cd repos/LycheeMAS
export PYTHONPATH=src

python -m pytest -q
python scripts/selfcheck_mlc_math.py
bash scripts/run_mlc_8b_3seeds.sh
bash scripts/run_mlc_qwen35_2b_3seeds.sh
```
