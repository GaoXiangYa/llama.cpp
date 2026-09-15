
#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ggml.h"

#include <CL/cl_platform.h>

namespace ops {

// fp16 存储模式下激活以 half 存放, 选对应的 kernel 变体
static cl_kernel mul_mat_kernel(ggml_ocl_backend *  b,
                                const char *        src_id,
                                const char *        fn_f32,
                                const char *        fn_f16,
                                const ggml_tensor * src1) {
    return ocl_pick_kernel(b, src_id, fn_f32, fn_f16, ocl_f16_packed(src1));
}
// 调试: 进一步细分禁用 MUL_MAT，避免一刀切全部回退 CPU。
// 用法示例:
//   GGML_OCL_DISABLE_MUL_MAT=q4_1
//   GGML_OCL_DISABLE_MUL_MAT=f16
//   GGML_OCL_DISABLE_MUL_MAT=f32
//   GGML_OCL_DISABLE_MUL_MAT=gemv
//   GGML_OCL_DISABLE_MUL_MAT=gemm
//   GGML_OCL_DISABLE_MUL_MAT=q4_1,gemv
static bool mul_mat_debug_disabled(const ggml_tensor * op) {
    const char * env = getenv("GGML_OCL_DISABLE_MUL_MAT");
    if (env == nullptr || env[0] == 0) {
        return false;
    }

    const bool has_q4_1 = strstr(env, "q4_1") != nullptr;
    const bool has_f16  = strstr(env, "f16")  != nullptr;
    const bool has_f32  = strstr(env, "f32")  != nullptr;
    const bool has_gemv = strstr(env, "gemv") != nullptr;
    const bool has_gemm = strstr(env, "gemm") != nullptr;

    const bool has_any_type   = has_q4_1 || has_f16 || has_f32;
    const bool has_any_kernel = has_gemv || has_gemm;

    const bool type_match =
        (op->src[0]->type == GGML_TYPE_Q4_1 && has_q4_1) ||
        (op->src[0]->type == GGML_TYPE_F16  && has_f16)  ||
        (op->src[0]->type == GGML_TYPE_F32  && has_f32);

    const bool kernel_match =
        (op->ne[1] == 1 && has_gemv) ||
        (op->ne[1] != 1 && has_gemm);

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

static bool gemv_q4_1_f32_run(ggml_ocl_backend *  b,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor *       dst) {
    cl_kernel k = mul_mat_kernel(b, "mul_mat/gemv_q4_1_f32", "gemv_q4_1_f32", "gemv_q4_1_f16", src1);
    if (k == nullptr) {
        GGML_LOG_ERROR("cannot find gemv_q4_1_f32 kernel");
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;
    ocl_kernel_call call;
    call.kernel  = k;
    call.op_name = "MUL_MAT";
    call.ndims   = 3;

    constexpr int nth = 256;
    call.local[0]     = nth;
    call.local[1]     = 1;
    call.local[2]     = 1;

    constexpr int subgroup_size  = 64;
    constexpr int rows_per_group = nth / subgroup_size;
    const int     num_groups     = (ne0 + rows_per_group - 1) / rows_per_group;
    call.global[0]               = num_groups * nth;
    call.global[1]               = ne2;
    call.global[2]               = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst,  ed);

    constexpr int blk_k        = 32;
    const int     blks_per_row = ne00 / blk_k;
    constexpr int blk_bytes    = 20;

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_i32(ocl_nb(src0, nb00));
    call.arg_i32(ocl_nb(src0, nb01));
    call.arg_i32(ocl_nb(src0, nb02));
    call.arg_i32(ocl_nb(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(ocl_nb(src1, nb10));
    call.arg_i32(ocl_nb(src1, nb11));
    call.arg_i32(ocl_nb(src1, nb12));
    call.arg_i32(ocl_nb(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(ocl_nb(dst, nb0));
    call.arg_i32(ocl_nb(dst, nb1));
    call.arg_i32(ocl_nb(dst, nb2));
    call.arg_i32(ocl_nb(dst, nb3));
    call.arg_i32(blk_k);
    call.arg_i32(blk_bytes);
    call.arg_i32(blks_per_row);

    call.enqueue(b);
    return true;
}

static bool gemv_f32_f32_run(ggml_ocl_backend *  b,
                             const ggml_tensor * src0,
                             const ggml_tensor * src1,
                             ggml_tensor *       dst) {
    cl_kernel k = mul_mat_kernel(b, "mul_mat/gemv_f32_f32", "gemv_f32_f32", "gemv_f32_f16", src1);
    if (k == nullptr) {
        GGML_LOG_ERROR("cannot find gemv_f32_f32 kernel!\n");
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;
    ocl_kernel_call call;
    call.kernel  = k;
    call.op_name = "MUL_MAT";
    call.ndims   = 3;

    constexpr int nth = 256;
    call.local[0]     = nth;
    call.local[1]     = 1;
    call.local[2]     = 1;

    constexpr int subgroup_size  = 64;
    constexpr int rows_per_group = nth / subgroup_size;
    const int     num_groups     = (ne0 + rows_per_group - 1) / rows_per_group;
    call.global[0]               = num_groups * nth;
    call.global[1]               = ne2;
    call.global[2]               = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst,  ed);

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_i32(ocl_nb(src0, nb00));
    call.arg_i32(ocl_nb(src0, nb01));
    call.arg_i32(ocl_nb(src0, nb02));
    call.arg_i32(ocl_nb(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(ocl_nb(src1, nb10));
    call.arg_i32(ocl_nb(src1, nb11));
    call.arg_i32(ocl_nb(src1, nb12));
    call.arg_i32(ocl_nb(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(ocl_nb(dst, nb0));
    call.arg_i32(ocl_nb(dst, nb1));
    call.arg_i32(ocl_nb(dst, nb2));
    call.arg_i32(ocl_nb(dst, nb3));

    call.enqueue(b);
    return true;
}

static bool gemv_f16_f32_run(ggml_ocl_backend *  b,
                             const ggml_tensor * src0,
                             const ggml_tensor * src1,
                             ggml_tensor *       dst) {
    cl_kernel k = mul_mat_kernel(b, "mul_mat/gemv_f16_f32", "gemv_f16_f32", "gemv_f16_f16", src1);
    if (k == nullptr) {
        GGML_LOG_ERROR("cannot find gemv_f16_f32 kernel!\n");
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;
    ocl_kernel_call call;
    call.kernel  = k;
    call.op_name = "MUL_MAT";
    call.ndims   = 3;

    constexpr int nth = 256;
    call.local[0]     = nth;
    call.local[1]     = 1;
    call.local[2]     = 1;

    constexpr int subgroup_size  = 64;
    constexpr int rows_per_group = nth / subgroup_size;
    const int     num_groups     = (ne0 + rows_per_group - 1) / rows_per_group;
    call.global[0]               = num_groups * nth;
    call.global[1]               = ne2;
    call.global[2]               = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst,  ed);

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_i32(ocl_nb(src0, nb00));
    call.arg_i32(ocl_nb(src0, nb01));
    call.arg_i32(ocl_nb(src0, nb02));
    call.arg_i32(ocl_nb(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(ocl_nb(src1, nb10));
    call.arg_i32(ocl_nb(src1, nb11));
    call.arg_i32(ocl_nb(src1, nb12));
    call.arg_i32(ocl_nb(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(ocl_nb(dst, nb0));
    call.arg_i32(ocl_nb(dst, nb1));
    call.arg_i32(ocl_nb(dst, nb2));
    call.arg_i32(ocl_nb(dst, nb3));

    call.enqueue(b);
    return true;
}

static bool gemm_f32_f32_run(ggml_ocl_backend *  b,
                             const ggml_tensor * src0,
                             const ggml_tensor * src1,
                             ggml_tensor *       dst) {
    cl_kernel k = mul_mat_kernel(b, "mul_mat/gemm_f32_f32", "gemm_f32_f32", "gemm_f32_f16", src1);
    if (k == nullptr) {
        GGML_LOG_ERROR("cannot find gemm_f32_f32 kernel!");
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;
    ocl_kernel_call call;
    call.kernel  = k;
    call.op_name = "MUL_MAT";
    call.ndims   = 3;

    constexpr int nth = 256;
    call.local[0]     = nth;
    call.local[1]     = 1;
    call.local[2]     = 1;

    constexpr int bm         = 16;
    constexpr int bn         = 16;
    const int     num_wg_m   = (ne1 + bm - 1) / bm;  // token 方向组数
    const int     num_wg_n   = (ne0 + bn - 1) / bn;  // neuron 方向组数
    const int     num_groups = num_wg_m * num_wg_n;
    call.global[0]           = num_groups * nth;
    call.global[1]           = ne2;
    call.global[2]           = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst,  ed);

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_i32(ocl_nb(src0, nb00));
    call.arg_i32(ocl_nb(src0, nb01));
    call.arg_i32(ocl_nb(src0, nb02));
    call.arg_i32(ocl_nb(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(ocl_nb(src1, nb10));
    call.arg_i32(ocl_nb(src1, nb11));
    call.arg_i32(ocl_nb(src1, nb12));
    call.arg_i32(ocl_nb(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(ocl_nb(dst, nb0));
    call.arg_i32(ocl_nb(dst, nb1));
    call.arg_i32(ocl_nb(dst, nb2));
    call.arg_i32(ocl_nb(dst, nb3));
    call.arg_i32(num_wg_n);

    call.enqueue(b);
    return true;
}

static bool gemm_f16_f32_run(ggml_ocl_backend *  b,
                             const ggml_tensor * src0,
                             const ggml_tensor * src1,
                             ggml_tensor *       dst) {
    cl_kernel k = mul_mat_kernel(b, "mul_mat/gemm_f16_f32", "gemm_f16_f32", "gemm_f16_f16", src1);
    if (k == nullptr) {
        GGML_LOG_ERROR("cannot find gemm_f16_f32 kernel!");
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;
    ocl_kernel_call call;
    call.kernel  = k;
    call.op_name = "MUL_MAT";
    call.ndims   = 3;

    constexpr int nth = 256;
    call.local[0]     = nth;
    call.local[1]     = 1;
    call.local[2]     = 1;

    constexpr int bm         = 16;
    constexpr int bn         = 16;
    const int     num_wg_m   = (ne1 + bm - 1) / bm;  // token 方向组数
    const int     num_wg_n   = (ne0 + bn - 1) / bn;  // neuron 方向组数
    const int     num_groups = num_wg_m * num_wg_n;
    call.global[0]           = num_groups * nth;
    call.global[1]           = ne2;
    call.global[2]           = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst,  ed);

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_i32(ocl_nb(src0, nb00));
    call.arg_i32(ocl_nb(src0, nb01));
    call.arg_i32(ocl_nb(src0, nb02));
    call.arg_i32(ocl_nb(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(ocl_nb(src1, nb10));
    call.arg_i32(ocl_nb(src1, nb11));
    call.arg_i32(ocl_nb(src1, nb12));
    call.arg_i32(ocl_nb(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(ocl_nb(dst, nb0));
    call.arg_i32(ocl_nb(dst, nb1));
    call.arg_i32(ocl_nb(dst, nb2));
    call.arg_i32(ocl_nb(dst, nb3));
    call.arg_i32(num_wg_n);

    call.enqueue(b);
    return true;
}

static bool gemm_q4_1_f32_run(ggml_ocl_backend *  b,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor *       dst) {
    cl_kernel k = mul_mat_kernel(b, "mul_mat/gemm_q4_1_f32", "gemm_q4_1_f32", "gemm_q4_1_f16", src1);
    if (k == nullptr) {
        GGML_LOG_ERROR("cannot find gemm_q4_1_f32 kernel!\n");
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;
    ocl_kernel_call call;
    call.kernel  = k;
    call.op_name = "MUL_MAT";
    call.ndims   = 3;

    constexpr int nth = 256;
    call.local[0]     = nth;
    call.local[1]     = 1;
    call.local[2]     = 1;

    constexpr int bm         = 4;
    constexpr int bn         = 64;
    const int     num_wg_m   = (ne1 + bm - 1) / bm;  // token 方向组数
    const int     num_wg_n   = (ne0 + bn - 1) / bn;  // neuron 方向组数
    const int     num_groups = num_wg_m * num_wg_n;
    call.global[0]           = num_groups * nth;
    call.global[1]           = ne2;
    call.global[2]           = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst,  ed);

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_i32(ocl_nb(src0, nb00));
    call.arg_i32(ocl_nb(src0, nb01));
    call.arg_i32(ocl_nb(src0, nb02));
    call.arg_i32(ocl_nb(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(ocl_nb(src1, nb10));
    call.arg_i32(ocl_nb(src1, nb11));
    call.arg_i32(ocl_nb(src1, nb12));
    call.arg_i32(ocl_nb(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(ocl_nb(dst, nb0));
    call.arg_i32(ocl_nb(dst, nb1));
    call.arg_i32(ocl_nb(dst, nb2));
    call.arg_i32(ocl_nb(dst, nb3));
    call.arg_i32(num_wg_n);

    call.enqueue(b);
    return true;
}

static bool mul_mat_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // gemv
    if (dst->ne[1] == 1) {
        if (src0->type == GGML_TYPE_Q4_1) {
            return gemv_q4_1_f32_run(b, src0, src1, dst);
        }
        if (src0->type == GGML_TYPE_F16) {
            return gemv_f16_f32_run(b, src0, src1, dst);
        }
        return gemv_f32_f32_run(b, src0, src1, dst);
    }
    // gemm
    if (src0->type == GGML_TYPE_Q4_1) {
        return gemm_q4_1_f32_run(b, src0, src1, dst);
    }
    if (src0->type == GGML_TYPE_F16) {
        return gemm_f16_f32_run(b, src0, src1, dst);
    }
    return gemm_f32_f32_run(b, src0, src1, dst);
}

}  // namespace ops

extern const ocl_op ocl_ops_op_mul_mat[] = {
    { GGML_OP_MUL_MAT, 0, ops::mul_mat_supports, ops::mul_mat_run },
    { GGML_OP_NONE,    0, nullptr,               nullptr          }, // 哨兵终止 (遍历依赖)
};
