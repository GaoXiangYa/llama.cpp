#include "ggml-ocl-internal.h"
#include "ops/ops.h"

#include <cstdlib>
#include <string>

// 所有 op 的唯一注册点: 新增算子只需在 ops/ 里实现并在下表加一行
static const ocl_op g_ops[] = {
    { GGML_OP_ADD,      ops::add_supports,      ops::add_run      },
    { GGML_OP_MUL,      ops::mul_supports,      ops::mul_run      },
    { GGML_OP_CONT,     ops::cpy_supports,      ops::cpy_run      },
    { GGML_OP_CPY,      ops::cpy_supports,      ops::cpy_run      },
    { GGML_OP_DUP,      ops::cpy_supports,      ops::cpy_run      },
    { GGML_OP_SET_ROWS, ops::set_rows_supports, ops::set_rows_run },
    { GGML_OP_GET_ROWS, ops::get_rows_supports, ops::get_rows_run },
    { GGML_OP_MUL_MAT,  ops::mul_mat_supports,  ops::mul_mat_run  },
    { GGML_OP_SOFT_MAX, ops::softmax_supports,  ops::softmax_run  },
    { GGML_OP_RMS_NORM, ops::rmsnorm_supports,  ops::rmsnorm_run  },
    { GGML_OP_GLU,      ops::glu_supports,      ops::glu_run      },
    { GGML_OP_ROPE,     ops::rope_supports,     ops::rope_run     },
};

bool ocl_op_dispatch(ggml_ocl_backend * b, ggml_tensor * node) {
    for (const ocl_op & o : g_ops) {
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

bool ocl_op_supports(const ggml_ocl_caps * caps, const ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }

    if (ggml_ocl_debug_op_disabled(node->op)) {
        GGML_LOG_INFO("ggml-ocl: op %s disabled by GGML_OCL_DISABLE_OPS, fallback to CPU\n", ggml_op_name(node->op));
        return false;
    }
    for (const ocl_op & o : g_ops) {
        if (node->op != o.op) {
            continue;
        }
        return o.supports != nullptr && o.supports(caps, node);
    }
    return false;
}
