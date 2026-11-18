#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ops/mul_mat/mul_mat_variants.h"

namespace ops {

// 单 token (gemv), 权重 F16/F32 (F32 在设备上也是 half).
// 配置必须和 kernels/mul_mat/gemv_f16_f16.cl 一致
namespace {
constexpr int NTH               = 256;
constexpr int SUBGROUP_SIZE     = 64;
constexpr int ROWS_PER_SUBGROUP = 4;
}  // namespace

bool gemv_f16_f16_run(ggml_ocl_backend *  b,
                      const ggml_tensor * src0,
                      const ggml_tensor * src1,
                      ggml_tensor *       dst) {
    cl_kernel k = b->kmgr->get("mul_mat/gemv_f16_f16", "gemv_f16_f16");
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

    // GQA 广播比
    call.arg_i32((cl_int) (ne12 / ne02));
    call.arg_i32((cl_int) (ne13 / ne03));

    call.enqueue(b);
    return true;
}

}  // namespace ops
