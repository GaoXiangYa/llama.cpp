#!/usr/bin/env bash
#
# ggml-ocl fp16 存储模式: 端到端验证
#
# 用法:
#   ggml/src/ggml-ocl/fp16-e2e.sh [模型.gguf] [build目录]
#
# 环境变量:
#   NGL        offload 层数  (默认 999)
#   OUT        输出目录      (默认自动建临时目录)
#   PPL_CTX    perplexity 上下文 (默认 128)
#   TIMEOUT    每步超时秒数  (默认 3600)
#
# 设计要点: 判据全部是数字, 不依赖任何日志行。
#   本仓库的 llama-perplexity / llama-bench 日志里既没有后端 INFO 行也没有
#   offload 记录, 靠 grep 日志判断"跑在哪", 会得出错误结论 (踩过)。
#   llama-cli 的 stderr 有日志但它的会话式 UI 让文本对比很脆。
#   所以: 跑三组配置, 用 CPU 参考组当锚点。
#
#      A  -ngl 0                    纯 CPU 参考
#      B  -ngl NGL, GGML_OCL_FP16=0 GPU, f32 路径
#      C  -ngl NGL, GGML_OCL_FP16=1 GPU, fp16 存储
#
#   PPL(A) 与 PPL(B) 不同 -> GPU 真的参与了计算 (这就是守卫)
#   PPL(C) - PPL(B)       -> fp16 存储的精度代价, 同硬件同 kernel, 干净地隔离出来
#   PPL(B) - PPL(A)       -> 你 kernel 自身相对 CPU 的偏差, 和 fp16 无关, 但要心里有数
#
# 注意: 目前只有 MUL_MAT / SET_ROWS 适配了 fp16, 其余 op 回退 CPU,
# 所以 C 组会明显更慢, 且当前的耗时数字不代表 fp16 的最终性能。

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${1:-}"
BUILD="${2:-$REPO/build}"

if [ -z "$MODEL" ]; then
    for c in "$HOME/PROJECTS/MODELS/Qwen/Qwen3-0.6B/qwen3_0.6b_q4_1.gguf" \
             "$HOME/PROJECTS/MODELS/Qwen/Qwen3-0.6B/ggml-model-Q4_1.gguf" \
             "$HOME/PROJECTS/MODELS/Qwen/qwen3_1.7B_q4_1.gguf"; do
        [ -f "$c" ] && { MODEL="$c"; break; }
    done
fi

NGL="${NGL:-999}"
PPL_CTX="${PPL_CTX:-128}"
TIMEOUT="${TIMEOUT:-3600}"
OUT="${OUT:-$(mktemp -d /tmp/ocl-fp16-e2e.XXXXXX)}"
mkdir -p "$OUT" || { echo "无法创建输出目录 $OUT"; exit 1; }

PPL_BIN="$BUILD/bin/llama-perplexity"

fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
info() { printf '\033[36m[info]\033[0m %s\n' "$*"; }

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    fail "找不到模型。请显式指定: $0 /path/to/model.gguf"
    find "$HOME/PROJECTS/MODELS" -maxdepth 4 -iname "*q4_1*.gguf" 2>/dev/null | head -10 | sed 's/^/    /'
    exit 1
fi
[ -x "$PPL_BIN" ] || { fail "找不到 $PPL_BIN"; exit 1; }

echo "======================================================================"
echo " ggml-ocl fp16 存储端到端验证"
echo "======================================================================"
info "model  : $MODEL"
info "build  : $BUILD"
info "output : $OUT"
info "ngl=$NGL  ppl_ctx=$PPL_CTX  timeout=${TIMEOUT}s"
echo

# 固定语料, 保证三组完全一致
CORPUS="$OUT/corpus.txt"
python3 - "$CORPUS" <<'PY'
import sys
para = ("The capital of France is Paris. It is the largest city in the country and one of "
        "the most visited cities in the world. The city is known for its museums, its "
        "architecture, and its cultural influence. Paris sits on the river Seine, in the "
        "north of France. The Eiffel Tower was completed in 1889 and has become a global "
        "symbol of the city. Millions of tourists travel there every year to see its "
        "landmarks and to eat French food. The Louvre is one of the largest museums in the "
        "world and holds a huge collection of art. Notre Dame cathedral stands on an island "
        "in the middle of the river. The city has a long history that reaches back more than "
        "two thousand years. Today it is a centre for finance, fashion, and education. Many "
        "people travel by train and by underground to move around the city each day. ")
open(sys.argv[1], "w").write((para * 3).strip() + "\n")
PY

run_one() {
    local name="$1" fp16="$2" ngl="$3"
    info "跑 $name (FP16=$fp16 ngl=$ngl) ..."
    local t0 t1
    t0=$(date +%s)
    GGML_OCL_FP16="$fp16" timeout -k 10 "$TIMEOUT" "$PPL_BIN" \
        -m "$MODEL" -ngl "$ngl" -f "$CORPUS" -c "$PPL_CTX" \
        --seed 42 --no-warmup -fa off \
        </dev/null > "$OUT/$name.out" 2> "$OUT/$name.err"
    echo "$?" > "$OUT/$name.rc"
    t1=$(date +%s)
    echo "$((t1 - t0))" > "$OUT/$name.secs"
    info "  退出码 $(cat "$OUT/$name.rc"), 墙钟 $((t1 - t0))s"
}

run_one A 0 0
run_one B 0 "$NGL"
run_one C 1 "$NGL"
echo

get_ppl() {
    grep -hoE "Final estimate: PPL = [0-9.]+" "$OUT/$1.out" "$OUT/$1.err" 2>/dev/null \
        | tail -1 | grep -oE "[0-9.]+$"
}

PA=$(get_ppl A); PB=$(get_ppl B); PC=$(get_ppl C)

for x in A B C; do
    rc=$(cat "$OUT/$x.rc")
    [ "$rc" = "0" ] || { fail "$x 退出码 $rc"; tail -20 "$OUT/$x.err" | sed 's/^/    /'; exit 3; }
done

echo "--- 结果 ---"
printf '  %-4s %-34s PPL = %-9s 墙钟 %ss\n' "A" "ngl=0        (纯 CPU 参考)" "${PA:-n/a}" "$(cat "$OUT/A.secs")"
printf '  %-4s %-34s PPL = %-9s 墙钟 %ss\n' "B" "ngl=$NGL fp16=0 (GPU, f32)" "${PB:-n/a}" "$(cat "$OUT/B.secs")"
printf '  %-4s %-34s PPL = %-9s 墙钟 %ss\n' "C" "ngl=$NGL fp16=1 (GPU, fp16存储)" "${PC:-n/a}" "$(cat "$OUT/C.secs")"
echo

if [ -z "$PA" ] || [ -z "$PB" ] || [ -z "$PC" ]; then
    fail "有配置拿不到 PPL (A='${PA:-无}' B='${PB:-无}' C='${PC:-无}')"
    exit 3
fi

python3 - "$PA" "$PB" "$PC" <<'PY'
import sys
a, b, c = (float(x) for x in sys.argv[1:4])
print(f"  B-A (GPU f32 相对 CPU)   : {(b-a)/a*100:+.3f}%")
print(f"  C-B (fp16 存储的代价)    : {(c-b)/b*100:+.4f}%")
print(f"  C-A (GPU fp16 相对 CPU)  : {(c-a)/a*100:+.3f}%")
print()
if b == a:
    print("\033[31m[FAIL]\033[0m PPL(A) == PPL(B): GPU 没有参与计算, 本轮无效")
    sys.exit(2)
print("\033[32m[ ok ]\033[0m 守卫通过: PPL(A) != PPL(B), GPU 确实参与了计算")
d = abs((c-b)/b*100)
if d < 0.01:
    print(f"\033[32m[ ok ]\033[0m fp16 存储代价 {d:.4f}% —— 可忽略")
elif d < 0.5:
    print(f"\033[33m[warn]\033[0m fp16 存储代价 {d:.4f}% —— 可接受")
else:
    print(f"\033[31m[FAIL]\033[0m fp16 存储代价 {d:.4f}% —— 超出预期")
    sys.exit(1)
PY
RC=$?

echo
echo "--- 性能 (仅供参考) ---"
warn "只有 MUL_MAT / SET_ROWS 在设备上, C 组其余 op 回退 CPU, 慢是预期内的。"
echo
echo "======================================================================"
if [ "$RC" = "0" ]; then
    ok "端到端通过: fp16 存储机制自洽"
    echo "     下一步: 按清单转换剩余 op (rmsnorm -> add -> rope -> softmax/glu/mul -> get_rows),"
    echo "             每转一个重跑本脚本, 看 C-B 的变化。"
else
    fail "未通过, 详见上面的数字"
    echo "     排查: GGML_OCL_DUMP_NODE=1 GGML_OCL_CHECK_NAN=1 逐节点比 min/max"
    echo "           GGML_OCL_DISABLE_OPS=NAME 二分定位"
fi
echo "     产物: $OUT/"
echo "======================================================================"
exit $RC
