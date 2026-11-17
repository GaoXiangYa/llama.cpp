#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ops/ops.h"

#include <cstdlib>
#include <cstring>

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

bool mul_mat_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
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
        GGML_LOG_ERROR("ggml-ocl: MUL_MAT unsupported types src0=%s src1=%s ne=[%lld %lld %lld %lld]\n",
                       ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type),
                       (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3]);
        return false;
    }
    return true;
}

// src1 在设备上恒为 half (fp16 存储), 所以只有 gemv/gemm x q4_1/half 四种变体
enum mul_mat_extra {
    EXTRA_GEMV_Q4_1,  // blk_k, blk_bytes, blks_per_row
    EXTRA_GEMV_HALF,  // gqa_kv, gqa_b
    EXTRA_GEMM_N,     // num_wg_n
};

struct mul_mat_variant {
    const char *  src_id;
    const char *  fn;
    bool          gemv;
    bool          q4_1;
    int           rows_per_subgroup;  // gemv: 每个 subgroup 处理的行数
    int           bm;                 // gemm: token 方向 tile
    int           bn;                 // gemm: neuron 方向 tile
    mul_mat_extra extra;
};

// tile 必须和 .cl 里的 #define 一致, 否则静默算错
static const mul_mat_variant g_mul_mat_variants[] = {
    { "mul_mat/gemv_f16_f16",  "gemv_f16_f16",  true,  false, 4, 0,  0,  EXTRA_GEMV_HALF },
    { "mul_mat/gemv_q4_1_f16", "gemv_q4_1_f16", true,  true,  3, 0,  0,  EXTRA_GEMV_Q4_1 },
    { "mul_mat/gemm_f16_f16",  "gemm_f16_f16",  false, false, 0, 16, 16, EXTRA_GEMM_N },
    { "mul_mat/gemm_q4_1_f16", "gemm_q4_1_f16", false, true,  0, 16, 64, EXTRA_GEMM_N },
};

bool mul_mat_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const bool gemv = (ne1 == 1);
    const bool q4_1 = (src0->type == GGML_TYPE_Q4_1);

    const mul_mat_variant * v = nullptr;
    for (const mul_mat_variant & cand : g_mul_mat_variants) {
        if (cand.gemv == gemv && cand.q4_1 == q4_1) {
            v = &cand;
            break;
        }
    }
    if (v == nullptr) {
        GGML_LOG_ERROR("ggml-ocl: MUL_MAT no variant for src0=%s gemv=%d\n", ggml_type_name(src0->type), (int) gemv);
        return false;
    }

    cl_kernel k = b->kmgr->get(v->src_id, v->fn);
    if (k == nullptr) {
        return false;
    }

    constexpr int nth = 256;

    ocl_kernel_call call;
    call.kernel   = k;
    call.op_name  = "MUL_MAT";
    call.ndims    = 3;
    call.local[0] = nth;
    call.local[1] = 1;
    call.local[2] = 1;

    if (v->gemv) {
        const int rows_per_group = nth * v->rows_per_subgroup / 64;
        const int num_groups     = (int) ((ne0 + rows_per_group - 1) / rows_per_group);
        call.global[0] = (size_t) num_groups * nth;
    } else {
        const int num_wg_m = (int) ((ne1 + v->bm - 1) / v->bm);
        const int num_wg_n = (int) ((ne0 + v->bn - 1) / v->bn);
        call.global[0] = (size_t) (num_wg_m * num_wg_n) * nth;
    }
    call.global[1] = (size_t) ne2;
    call.global[2] = (size_t) ne3;

    ocl_pack_binary_args(call, src0, src1, dst, /*strides_u64=*/false);

    switch (v->extra) {
        case EXTRA_GEMV_Q4_1: {
            constexpr int blk_k     = 32;
            constexpr int blk_bytes = 20;
            call.arg_i32(blk_k);
            call.arg_i32(blk_bytes);
            call.arg_i32((cl_int) (ne00 / blk_k));
            break;
        }
        case EXTRA_GEMV_HALF:
            call.arg_i32((cl_int) (ne12 / ne02));
            call.arg_i32((cl_int) (ne13 / ne03));
            break;
        case EXTRA_GEMM_N:
            call.arg_i32((cl_int) ((ne0 + v->bn - 1) / v->bn));
            break;
    }

    call.enqueue(b);
    return true;
}

}  // namespace ops
