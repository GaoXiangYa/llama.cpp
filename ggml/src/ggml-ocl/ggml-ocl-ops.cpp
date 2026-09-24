#include "ggml-ocl-internal.h"

#include <cstdlib>
#include <string>

extern const ocl_op ocl_ops_op_add[];
extern const ocl_op ocl_ops_op_set_rows[];
extern const ocl_op ocl_ops_op_get_rows[];
extern const ocl_op ocl_ops_op_mul_mat[];
extern const ocl_op ocl_ops_op_softmax[];
extern const ocl_op ocl_ops_op_rmsnorm[];
extern const ocl_op ocl_ops_op_glu[];
extern const ocl_op ocl_ops_op_mul[];
extern const ocl_op ocl_ops_op_rope[];

static const ocl_op * g_op_groups[] = {
    ocl_ops_op_add,     ocl_ops_op_set_rows, ocl_ops_op_get_rows, ocl_ops_op_mul_mat, ocl_ops_op_softmax,
    ocl_ops_op_rmsnorm, ocl_ops_op_glu,      ocl_ops_op_mul,      ocl_ops_op_rope,
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
#ifdef GGML_OCL_OP_SHAPE
            GGML_LOG_INFO("ggml-ocl: GPU op %s (src0=%s src1=%s) ne=[%lld %lld %lld %lld]\n", ggml_op_name(node->op),
                          ggml_type_name(node->src[0]->type),
                          ggml_type_name(node->src[1] ? node->src[1]->type : GGML_TYPE_F32), (long long) node->ne[0],
                          (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
#endif
            return o.run(b, node->src[0], node->src[1], node);
        }
    }
    return false;
}

static bool ggml_ocl_debug_op_disabled(enum ggml_op op) {
    const char * env = getenv("GGML_OCL_DISABLE_OPS");
    if (env == nullptr || env[0] == 0) {
        return false;
    }

    const char *      op_name = ggml_op_name(op);
    const std::string list(env);

    size_t pos = 0;
    while (pos < list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) {
            end = list.size();
        }
        std::string item = list.substr(pos, end - pos);
        size_t      b    = item.find_first_not_of(" \t");
        size_t      e    = item.find_last_not_of(" \t");
        if (b != std::string::npos) {
            item = item.substr(b, e - b + 1);
        } else {
            item.clear();
        }
        if (!item.empty() && item == op_name) {
            return true;
        }
        pos = end + 1;
    }
    return false;
}

static bool ggml_ocl_fp16_op_ready(enum ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_SET_ROWS:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_GLU:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GET_ROWS:
        case GGML_OP_ROPE:
            return true;
        default:
            return false;
    }
}

bool ocl_op_supports(const ggml_ocl_caps * caps, const ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }

    if (ggml_ocl_fp16_storage() && !ggml_ocl_fp16_op_ready(node->op)) {
        GGML_LOG_DEBUG("ggml-ocl: %s not fp16-ready, fallback to CPU\n", ggml_op_name(node->op));
        return false;
    }

    if (ggml_ocl_debug_op_disabled(node->op)) {
        GGML_LOG_INFO("ggml-ocl: op %s disabled by GGML_OCL_DISABLE_OPS, fallback to CPU\n", ggml_op_name(node->op));
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
