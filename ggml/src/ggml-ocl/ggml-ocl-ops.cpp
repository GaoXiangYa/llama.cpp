#include <cstdlib>
#include <string>

#include "ggml-ocl-internal.h"

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
    ocl_ops_op_add,
    ocl_ops_op_set_rows,
    ocl_ops_op_get_rows,
    ocl_ops_op_mul_mat,
    ocl_ops_op_softmax,
    ocl_ops_op_rmsnorm,
    ocl_ops_op_glu,
    ocl_ops_op_mul,
    ocl_ops_op_rope,
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
            GGML_LOG_INFO("ggml-ocl: GPU op %s (src0=%s src1=%s) ne=[%lld %lld %lld %lld]\n",
                          ggml_op_name(node->op), ggml_type_name(node->src[0]->type),
                          ggml_type_name(node->src[1] ? node->src[1]->type : GGML_TYPE_F32),
                          (long long) node->ne[0], (long long) node->ne[1],
                          (long long) node->ne[2], (long long) node->ne[3]);
            return o.run(b, node->src[0], node->src[1], node);
        }
    }
    return false;
}

// 调试: GGML_OCL_DISABLE_OPS=MUL_MAT,ROPE,GLU
// 让指定 op 强制走 CPU，用于定位是哪个算子算错。
static bool ggml_ocl_debug_op_disabled(enum ggml_op op) {
    const char * env = getenv("GGML_OCL_DISABLE_OPS");
    if (env == nullptr || env[0] == 0) {
        return false;
    }

    const char * op_name = ggml_op_name(op);
    const std::string list(env);

    size_t pos = 0;
    while (pos < list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) {
            end = list.size();
        }
        std::string item = list.substr(pos, end - pos);
        // 去掉首尾空格
        size_t b = item.find_first_not_of(" \t");
        size_t e = item.find_last_not_of(" \t");
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

// fp16 存储模式下只有已经适配 half 存放的 op 能留在设备上。
// 其余回退 CPU: 边界拷贝走 buffer 的 get/set, 那里会做 half <-> float 转换。
// 每适配一个 op 就在这里加一个 case。
//
// 注意: 这个回退只对 scheduler 分配的张量有效。预分配张量 (KV cache) 不能被
// 搬走, ggml-backend.cpp:930 会直接 abort, 所以碰它们的 op 必须在这里返回 true。
static bool ggml_ocl_fp16_op_ready(enum ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:   // 权重 + 激活
        case GGML_OP_SET_ROWS:  // KV cache 写入 (预分配, 必须支持)
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
        GGML_LOG_INFO("ggml-ocl: op %s disabled by GGML_OCL_DISABLE_OPS, fallback to CPU\n",
                      ggml_op_name(node->op));
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
