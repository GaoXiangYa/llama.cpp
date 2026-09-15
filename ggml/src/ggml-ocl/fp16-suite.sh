#!/usr/bin/env bash
#
# 全量 test-backend-ops 的 fp16 A/B 对比
#
# 用法:
#   ggml/src/ggml-ocl/fp16-suite.sh [build目录] [设备名]
#
# 默认: build 目录 = 仓库下的 build, 设备名 = GPUOCL
#
# 这个脚本回答的问题是: "全量测试里的失败, 哪些是 fp16 引入的, 哪些本来就存在"
# 判据: 两份失败列表的 diff。fp16=0 那轮我所有 helper 都是恒等 (ocl_f16_packed
# 全假, kernel 选择不变, 门禁不参与), 所以两轮的差异只可能来自 fp16 存储路径。
#
#   差异为空            -> fp16 干净, 失败全是既有的
#   fp16=1 多出若干条   -> 那些就是 fp16 转换引入的

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD="${1:-$REPO/build}"
DEV="${2:-GPUOCL}"
OUT="${OUT:-$(mktemp -d /tmp/ocl-fp16-suite.XXXXXX)}"
BIN="$BUILD/bin/test-backend-ops"

mkdir -p "$OUT" || { echo "无法创建 $OUT"; exit 1; }

fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
info() { printf '\033[36m[info]\033[0m %s\n' "$*"; }

[ -x "$BIN" ] || { fail "找不到 $BIN"; exit 1; }

echo "======================================================================"
echo " test-backend-ops fp16 A/B (设备 $DEV)"
echo "======================================================================"
info "产物: $OUT"
echo

for F in 0 1; do
    info "跑 GGML_OCL_FP16=$F ..."
    GGML_OCL_FP16=$F timeout -k 10 7200 "$BIN" -b "$DEV" > "$OUT/fp16_$F.txt" 2>&1
    echo "$?" > "$OUT/fp16_$F.rc"
    # 失败列表: "Failing tests:" 之后所有以两个空格开头的行
    # 注意: 不能用 f{exit} 提前退出 —— GGML_OCL_PROFILING 的汇总行会插进列表中间,
    # 它不以空格开头, 会让提取提前截断。改成遇到 "  Backend X: ..." 才停止,
    # 并且滤掉 profiling 那些也以两个空格开头的行。
    awk '/^Failing tests:/{f=1;next} f && /^  Backend /{f=0} f && /^  /{print}' \
        "$OUT/fp16_$F.txt" \
        | sed 's/^  //' \
        | grep -vE "^profiling summary| calls +[0-9]+ +ms total| us/call" \
        | sort -u > "$OUT/fail_$F.txt"
    ns=$(grep -c "not supported" "$OUT/fp16_$F.txt")
    info "  退出码 $(cat "$OUT/fp16_$F.rc"), 日志 $OUT/fp16_$F.txt, not-supported $ns 次"
done
echo

echo "--- 通过数 ---"
for F in 0 1; do
    r=$(grep -oE "[0-9]+/[0-9]+ tests passed" "$OUT/fp16_$F.txt" | tail -1)
    n=$(wc -l < "$OUT/fail_$F.txt")
    printf '  fp16=%s   %-24s 失败 %s 条\n' "$F" "${r:-<未打印>}" "$n"
done
echo

echo "--- 失败列表差异 ---"
if diff -q "$OUT/fail_0.txt" "$OUT/fail_1.txt" >/dev/null 2>&1; then
    ok "两份失败列表完全相同 -> fp16 没有引入任何新失败"
    echo
    info "fp16=0 的失败清单 (既有问题, 与 fp16 无关):"
    if [ -s "$OUT/fail_0.txt" ]; then
        sed 's/^/    /' "$OUT/fail_0.txt"
    else
        echo "    (无)"
    fi
else
    echo "  fp16=1 相对 fp16=0:"
    diff "$OUT/fail_0.txt" "$OUT/fail_1.txt" | sed 's/^/    /'
    echo
    n_new=$(diff "$OUT/fail_0.txt" "$OUT/fail_1.txt" | grep -c '^>' || true)
    if [ "${n_new:-0}" -gt 0 ]; then
        fail "fp16 引入了 $n_new 条新失败 —— 这些是需要排查的"
        echo "    只看新增的那部分 (上面 diff 里 '>' 开头的行)"
    else
        ok "fp16=1 没有新增失败 (差异只是 fp16=0 多出的条目)"
    fi
fi
echo
echo "     产物: $OUT/"
echo "======================================================================"
