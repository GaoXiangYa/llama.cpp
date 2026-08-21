#include "ggml-ocl-internal.h"

extern const ocl_op ocl_ops_op_add[];
extern const ocl_op ocl_ops_op_set_rows[];
extern const ocl_op ocl_ops_op_get_rows[];

static const ocl_op * g_op_groups[] = {
    ocl_ops_op_add,
    ocl_ops_op_set_rows,
    ocl_ops_op_get_rows,
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
