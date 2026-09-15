#!/usr/bin/env bash
#
# 用 llama-cli 跑一轮 fp16 推理 (或它的对照组)
#
# 用法:
#   ggml/src/ggml-ocl/fp16-cli.sh [模式] [模型.gguf]
#
#   模式:
#     off      GGML_OCL_FP16=0                 基线
#     on       GGML_OCL_FP16=1                 fp16 存储
#     on-norope GGML_OCL_FP16=1 + ROPE 走 CPU   新启用路径的对照组
#
# 环境变量: PROMPT / N_PREDICT / NGL
#
# 为什么用 stdin 喂提示词:
#   本仓库的 llama-cli 是会话式实现, -no-cnv / -st 都不生效; stdin 为
#   /dev/null 时会 EOF 空转刷屏。把提示词和 /exit 一起喂进去才能非交互跑完。

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODE="${1:-on}"
MODEL="${2:-}"

if [ -z "$MODEL" ]; then
    for c in "$HOME/PROJECTS/MODELS/Qwen/Qwen3-0.6B/qwen3_0.6b_q4_1.gguf" \
             "$HOME/PROJECTS/MODELS/Qwen/Qwen3-0.6B/ggml-model-Q4_1.gguf" \
             "$HOME/PROJECTS/MODELS/Qwen/qwen3_1.7B_q4_1.gguf"; do
        [ -f "$c" ] && { MODEL="$c"; break; }
    done
fi

PROMPT="${PROMPT:-The capital of France is}"
N_PREDICT="${N_PREDICT:-32}"
NGL="${NGL:-999}"
CLI="$REPO/build/bin/llama-cli"

fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; }
info() { printf '\033[36m[info]\033[0m %s\n' "$*"; }

case "$MODE" in
    off)       FP16=0; DISABLE="" ;;
    on)        FP16=1; DISABLE="" ;;
    on-norope) FP16=1; DISABLE="ROPE" ;;
    *) fail "未知模式 '$MODE' (可用: off / on / on-norope)"; exit 1 ;;
esac

[ -x "$CLI" ]   || { fail "找不到 $CLI"; exit 1; }
[ -f "$MODEL" ] || { fail "找不到模型 $MODEL (用法: $0 $MODE /path/to/model.gguf)"; exit 1; }

echo "======================================================================"
echo " llama-cli fp16 推理   mode=$MODE  FP16=$FP16  DISABLE_OPS=${DISABLE:-无}"
echo "======================================================================"
info "model : $MODEL"
info "prompt: $PROMPT   n_predict=$N_PREDICT   ngl=$NGL"
echo

OUT="${OUT:-$(mktemp -d /tmp/ocl-fp16-cli.XXXXXX)}"
mkdir -p "$OUT"

printf '%s\n/exit\n' "$PROMPT" | \
    env GGML_OCL_FP16="$FP16" ${DISABLE:+GGML_OCL_DISABLE_OPS="$DISABLE"} \
        timeout -k 10 "${TIMEOUT:-1800}" "$CLI" \
        -m "$MODEL" -ngl "$NGL" -fa off --seed 42 --temp 0 -n "$N_PREDICT" \
        > "$OUT/out.txt" 2> "$OUT/err.txt"
RC=$?

echo "--- 生成内容 (尾部) ---"
sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$OUT/out.txt" | grep -v '^\[ Prompt: ' | tail -20
echo
echo "--- 统计 ---"
printf '  退出码        : %s\n' "$RC"
printf '  OCL 算子派发  : %s\n' "$(grep -c 'ggml-ocl: GPU op' "$OUT/err.txt")"
printf '  fp16 存储开关 : %s\n' "$(grep -m1 -o 'fp16 storage: [a-z]*' "$OUT/err.txt" || echo '<未打印>')"
printf '  no usable GPU : %s\n' "$(grep -c 'no usable GPU found' "$OUT/err.txt")"
printf '  Generation    : %s\n' "$(grep -oE 'Generation: *[0-9.]+ t/s' "$OUT/out.txt" | tail -1 || echo 'n/a')"
echo
info "产物: $OUT/"
echo "======================================================================"
[ "$RC" = "0" ] || exit "$RC"
