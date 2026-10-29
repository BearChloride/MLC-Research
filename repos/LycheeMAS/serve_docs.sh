#!/usr/bin/env bash
# LycheeMAS 文档站启动脚本 —— 本地/远程在线预览（MkDocs Material + 自动 API Reference）。
#
# 用法：
#   ./serve_docs.sh                # 启动开发服务器（默认 0.0.0.0:8000，改 docstring/页面即热更新）
#   ./serve_docs.sh -p 8080        # 指定端口
#   ./serve_docs.sh -H 127.0.0.1   # 只绑本机（不对外网暴露）
#   ./serve_docs.sh build          # 只生成静态站点到 site/（不启动服务器）
#
# 环境变量覆盖：HOST / PORT / VENV。
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODE="serve"

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    build)      MODE="build"; shift ;;
    serve)      MODE="serve"; shift ;;
    -p|--port)  PORT="$2"; shift 2 ;;
    -H|--host)  HOST="$2"; shift 2 ;;
    -h|--help)  awk 'NR==1{next} /^#/{sub(/^#\s?/,"");print;next} {exit}' "$0"; exit 0 ;;
    *)          echo "未知参数：$1（用 -h 看用法）" >&2; exit 1 ;;
  esac
done

# ---- 激活虚拟环境（若当前 shell 尚未激活）----
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for cand in "${VENV:-}" "$REPO_DIR/.venv" "$REPO_DIR/../.venv"; do
    if [[ -n "$cand" && -f "$cand/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "$cand/bin/activate"
      echo "[docs] 已激活 venv：$cand"
      break
    fi
  done
fi

# ---- 确认文档工具已装；缺则自动安装 docs extra ----
if ! command -v mkdocs >/dev/null 2>&1; then
  echo "[docs] 未检测到 mkdocs，安装文档依赖（.[docs]）……"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e ".[docs]"
  else
    python -m pip install -e ".[docs]"
  fi
fi

# 屏蔽 gen-files/literate-nav/section-index 依赖的 properdocs 推广横幅
export DISABLE_MKDOCS_2_WARNING=true

# ---- build 模式：只生成静态站点 ----
if [[ "$MODE" == "build" ]]; then
  echo "[docs] 生成静态站点 -> $REPO_DIR/site/"
  exec mkdocs build --strict
fi

# ---- serve 模式：启动开发服务器 ----
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "============================================================"
echo "  LycheeMAS 文档站已启动（Ctrl-C 停止）"
echo "  本机浏览器 : http://127.0.0.1:${PORT}"
if [[ -n "$IP" && "$HOST" == "0.0.0.0" ]]; then
  echo "  远程访问   : http://${IP}:${PORT}   （从你的电脑浏览器打开）"
fi
echo "============================================================"
exec mkdocs serve -a "${HOST}:${PORT}"
