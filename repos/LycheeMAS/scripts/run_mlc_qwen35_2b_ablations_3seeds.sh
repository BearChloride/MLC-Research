#!/usr/bin/env bash
set -u

ROOT_DIR="/home/med_code/nhzk/yjw/MAS-Research/repos/LycheeMAS"
PY="/home/med_code/miniconda3/envs/qwen35/bin/python"
MODEL="/home/med_code/models/Qwen3.5-2B"
OUT_ROOT="$ROOT_DIR/runs/benchmarks"
LOG_DIR="$OUT_ROOT/Qwen3.5-2B/mlc_ablation_logs"
GPU_INDEX="${MLC_GPU_INDEX:-1}"

mkdir -p "$LOG_DIR"

run_one() {
  local variant="$1"
  local seed="$2"
  local method="mlc_aligned1024_${variant}_seed${seed}"
  local dir="$OUT_ROOT/Qwen3.5-2B/${method}/gsm8k"
  local ckpt="$dir/mlc_state.pt"
  local log="$LOG_DIR/ablation_${variant}_seed_${seed}.log"
  local resume_args=()

  if [ -f "$dir/metrics.json" ]; then
    printf '%s variant=%s seed=%s already-complete skip\n' \
      "$(date '+%F %T')" "$variant" "$seed" \
      | tee -a "$LOG_DIR/launcher.log"
    return 0
  fi

  if [ -f "$ckpt" ]; then
    resume_args=(--resume-checkpoint "$ckpt")
  fi

  printf '%s variant=%s seed=%s start method=%s\n' \
    "$(date '+%F %T')" "$variant" "$seed" "$method" \
    | tee -a "$LOG_DIR/launcher.log"

  cd "$ROOT_DIR" || return 1
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/run_mlc_gsm8k.py \
      --model-path "$MODEL" \
      --train-n 32 --eval-n 100 \
      --sample-seed "$seed" --eval-seed "$seed" \
      --rounds 2 --max-new-tokens 1024 --max-input-tokens 0 \
      --device cuda:0 --dtype bfloat16 \
      --gradient-checkpointing \
      --ablation "$variant" \
      --method "$method" --out-root "$OUT_ROOT" \
      "${resume_args[@]}" >>"$log" 2>&1

  local status=$?
  printf '%s variant=%s seed=%s exit=%s\n' \
    "$(date '+%F %T')" "$variant" "$seed" "$status" \
    | tee -a "$LOG_DIR/launcher.log"
  return "$status"
}

status=0
for variant in no_role_embedding no_mixing no_l_dis; do
  for seed in 101 202 303; do
    run_one "$variant" "$seed" || status=$?
  done
done

printf '%s all-ablations-done exit=%s\n' "$(date '+%F %T')" "$status"
exit "$status"
