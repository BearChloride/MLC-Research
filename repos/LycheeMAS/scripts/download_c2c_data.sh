#!/usr/bin/env bash
# 下载 Cache-to-Cache (C2C) 训练/评测用的 HuggingFace 数据集。
#   主力训练集 = teknium/OpenHermes-2.5（C2C fuser 训练用，~数 GB，较大）。其余为消融/可选。
#
# 用法：
#   bash scripts/download_c2c_data.sh              # 只下主力 OpenHermes-2.5
#   bash scripts/download_c2c_data.sh all          # 下全部（主力 + 可选）
#   bash scripts/download_c2c_data.sh cais/mmlu allenai/openbookqa   # 指定 repo id
#   DEST=/your/dir HF_ENDPOINT=https://huggingface.co bash scripts/download_c2c_data.sh all
#
# 依赖：huggingface_hub（pip install -U huggingface_hub）。国内慢默认走 hf-mirror 镜像。
set -euo pipefail

DEST="${DEST:-/data/mxy/Project/CDM/Data/raw/c2c}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # 官方源：
export HF_ENDPOINT=https://huggingface.co

# 主力训练集（C2C fuser 训练，recipe/train_recipe/C2C_*.json 用）
TRAIN_MAIN=( "teknium/OpenHermes-2.5" )
# 消融/其它配置 + 评测用到的（dataset_adapters.py 里的 load_dataset 目标）
OPTIONAL=(
  "cais/mmlu"                       # MMLUChatDataset（baseline/oracle）
  "Brench/MMLU-Pro-CoT-Train-43K"   # MMLUCotChatDataset
  "allenai/openbookqa"              # OpenBookChatDataset
  "THUDM/LongBench"                 # LongBenchChatDataset
)

# CLI：新版 huggingface_hub 改名为 `hf`，旧版是 `huggingface-cli`
HF="huggingface-cli"
command -v hf >/dev/null 2>&1 && HF="hf"
command -v "$HF" >/dev/null 2>&1 || { echo "缺 huggingface CLI：pip install -U huggingface_hub"; exit 1; }

dl() {  # $1 = HF dataset repo id
  local name="$1"
  local out="$DEST/${name//\//__}"
  echo "[c2c-data] $name -> $out  (HF_ENDPOINT=$HF_ENDPOINT)"
  "$HF" download --repo-type dataset "$name" --local-dir "$out" \
    || { echo "[c2c-data] FAILED: $name（检查网络/镜像/HF_TOKEN）"; return 1; }
}

case "${1:-main}" in
  main) sets=( "${TRAIN_MAIN[@]}" ) ;;
  all)  sets=( "${TRAIN_MAIN[@]}" "${OPTIONAL[@]}" ) ;;
  *)    sets=( "$@" ) ;;   # 直接传一个或多个 repo id
esac

mkdir -p "$DEST"
for s in "${sets[@]}"; do dl "$s"; done
echo "[c2c-data] done -> $DEST"
