#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ops/mul_mat/mul_mat_variants.h"

namespace ops {

// 单 token (gemv), 权重 Q4_1. 这里的配置必须和 kernels/mul_mat/gemv_q4_1_f16.cl 一致
namespace {
constexpr int NTH               = 256;  // work-group 大小
constexpr int SUBGROUP_SIZE     = 64;
constexpr int ROWS_PER_SUBGROUP = 3;    // 每个 subgroup 处理 3 行
constexpr int BLK_K             = 32;   // Q4_1 量化块大小
constexpr int BLK_BYTES         = 20;   // 每块字节数
}  // namespace

bool gemv_q4_1_f16_run(ggml_ocl_backend *  b,
                       const ggml_tensor * src0,
                       const ggml_tensor * src1,
                       ggml_tensor *       dst) {
    cl_kernel k = b->kmgr->get("mul_mat/gemv_q4_1_f16", "gemv_q4_1_f16");
    if (k == nullptr) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    ocl_kernel_call call;
    call.kernel   = k;
    call.op_name  = "MUL_MAT";
    call.ndims    = 3;
    call.local[0] = NTH;
    call.local[1] = 1;
    call.local[2] = 1;

    const int rows_per_group = NTH * ROWS_PER_SUBGROUP / SUBGROUP_SIZE;
    call.global[0] = (size_t) ((ne0 + rows_per_group - 1) / rows_per_group) * NTH;
    call.global[1] = (size_t) ne2;
    call.global[2] = (size_t) ne3;

    ocl_pack_binary_args(call, src0, src1, dst, /*strides_u64=*/false);

    call.arg_i32(BLK_K);
    call.arg_i32(BLK_BYTES);
    call.arg_i32((cl_int) (ne00 / BLK_K));

    call.enqueue(b);
    return true;
}

}  // namespace ops
