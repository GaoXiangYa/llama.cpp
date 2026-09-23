
#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ggml.h"

#include "mul_mat.h"

#include <CL/cl.h>
#include <CL/cl_platform.h>

namespace ops {

static bool mul_mat_debug_disabled(const ggml_tensor * op) {
    const char * env = getenv("GGML_OCL_DISABLE_MUL_MAT");
    if (env == nullptr || env[0] == 0) {
        return false;
    }

    const bool has_q4_1 = strstr(env, "q4_1") != nullptr;
    const bool has_f16  = strstr(env, "f16") != nullptr;
    const bool has_f32  = strstr(env, "f32") != nullptr;
    const bool has_gemv = strstr(env, "gemv") != nullptr;
    const bool has_gemm = strstr(env, "gemm") != nullptr;

    const bool has_any_type   = has_q4_1 || has_f16 || has_f32;
    const bool has_any_kernel = has_gemv || has_gemm;

    const bool type_match = (op->src[0]->type == GGML_TYPE_Q4_1 && has_q4_1) ||
                            (op->src[0]->type == GGML_TYPE_F16 && has_f16) ||
                            (op->src[0]->type == GGML_TYPE_F32 && has_f32);

    const bool kernel_match = (op->ne[1] == 1 && has_gemv) || (op->ne[1] != 1 && has_gemm);

    if (has_any_type && has_any_kernel) {
        return type_match && kernel_match;
    }
    if (has_any_type) {
        return type_match;
    }
    if (has_any_kernel) {
        return kernel_match;
    }

    return false;
}

static bool mul_mat_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_MUL_MAT) {
        GGML_LOG_ERROR("op type is not mul mat!\n");
        return false;
    }

    if (mul_mat_debug_disabled(op)) {
        GGML_LOG_INFO("ggml-ocl: MUL_MAT src0=%s ne1=%lld disabled by GGML_OCL_DISABLE_MUL_MAT, fallback to CPU\n",
                      ggml_type_name(op->src[0]->type), (long long) op->ne[1]);
        return false;
    }
    if ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_Q4_1 &&
         op->src[0]->type != GGML_TYPE_F16) ||
        op->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("mul mat not support such type!");
        return false;
    }
    return true;
}

// 选 kernel: (src0 类型, 单token/多token, 激活是否 fp16 存储) -> mul_mat/<kernel>.cpp
static bool mul_mat_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool f16   = ocl_f16_packed(src1);
    const bool gemv  = (dst->ne[1] == 1);

    if (gemv) {
        switch (src0->type) {
            case GGML_TYPE_Q4_1: return f16 ? gemv_q4_1_f16_run(b, src0, src1, dst) : gemv_q4_1_f32_run(b, src0, src1, dst);
            case GGML_TYPE_F16:  return f16 ? gemv_f16_f16_run(b, src0, src1, dst)  : gemv_f16_f32_run(b, src0, src1, dst);
            case GGML_TYPE_F32:  return f16 ? gemv_f32_f16_run(b, src0, src1, dst)  : gemv_f32_f32_run(b, src0, src1, dst);
            default: break;
        }
    } else {
        switch (src0->type) {
            case GGML_TYPE_Q4_1: return f16 ? gemm_q4_1_f16_run(b, src0, src1, dst) : gemm_q4_1_f32_run(b, src0, src1, dst);
            case GGML_TYPE_F16:  return f16 ? gemm_f16_f16_run(b, src0, src1, dst)  : gemm_f16_f32_run(b, src0, src1, dst);
            case GGML_TYPE_F32:  return f16 ? gemm_f32_f16_run(b, src0, src1, dst)  : gemm_f32_f32_run(b, src0, src1, dst);
            default: break;
        }
    }

    GGML_LOG_ERROR("ggml-ocl: MUL_MAT no kernel for src0=%s gemv=%d f16=%d\n",
                   ggml_type_name(src0->type), (int) gemv, (int) f16);
    return false;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_mul_mat[] = {
    { GGML_OP_MUL_MAT, 0, ops::mul_mat_supports, ops::mul_mat_run },
    { GGML_OP_NONE,    0, nullptr,               nullptr          }, // 哨兵终止 (遍历依赖)
};
