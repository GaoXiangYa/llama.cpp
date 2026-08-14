// ggml-ocl backend: operator registry (DESIGN.md section 19.1)
// 各算子族文件 (ops/*.cpp) 贡献条目数组 (哨兵终止), 本文件统一遍历.
// 新增算子: 在对应 ops/op-*.cpp 中加条目, 并在 g_op_groups 中登记该文件.

#include "ggml-ocl-internal.h"

extern const ocl_op ocl_ops_op_add[];

static const ocl_op * g_op_groups[] = {
    ocl_ops_op_add,
};

bool ocl_op_dispatch(ggml_ocl_backend * b, ggml_tensor * node) {
    for (const ocl_op * group : g_op_groups) {
        for (int i = 0; group[i].run != nullptr; i++) {
            const ocl_op & o = group[i];
            if (node->op != o.op) {
                continue;
            }
            if (o.supports != nullptr && !o.supports(&b->caps, node)) {
                continue;
            }
            return o.run(b, node->src[0], node->src[1], node);
        }
    }
    return false;
}

bool ocl_op_supports(const ggml_ocl_caps * caps, const ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }
    for (const ocl_op * group : g_op_groups) {
        for (int i = 0; group[i].run != nullptr; i++) {
            const ocl_op & o = group[i];
            if (node->op != o.op) {
                continue;
            }
            if (o.supports == nullptr) {
                return false;
            }
            return o.supports(caps, node);
        }
    }
    return false;
}
