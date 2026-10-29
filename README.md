# MLC-Research

本仓库用于复现与适配论文 *Advancing Collaborative Debates with Role Differentiation through Multi-agent Reinforcement Learning*（ACL 2025 Long Papers，编号 2025.acl-long.1105）。

项目基于 LycheeMAS 框架，在 GSM8K 上完成了 MLC 的角色编码、soft prefix、Q-mixing、角色分化损失等核心机制，并对比了 Qwen3-8B、Qwen3.5-2B 与 Qwen3-0.6B 的 baseline、三随机种子和消融实验。

## 目录结构

- `repos/LycheeMAS/`：代码仓库，核心实现位于 `src/lychee_mas/methods/prerun/mlc/`，实验入口位于 `scripts/run_mlc_gsm8k.py`。
- `model/`：实验使用的数学推理原始数据。
- `references/`：论文 PDF。

## 不包含的内容

- `models/`：模型权重不上传，请按环境文档从 ModelScope 或 Hugging Face 获取。
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
