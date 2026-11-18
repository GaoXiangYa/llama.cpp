#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ops/mul_mat/mul_mat_variants.h"

namespace ops {

// 多 token (gemm), 权重 F16/F32 (F32 在设备上也是 half).
// 配置必须和 kernels/mul_mat/gemm_f16_f16.cl 一致
namespace {
constexpr int NTH = 256;
constexpr int BM  = 16;  // token 方向 tile
constexpr int BN  = 16;  // neuron 方向 tile
}  // namespace

bool gemm_f16_f16_run(ggml_ocl_backend *  b,
                      const ggml_tensor * src0,
                      const ggml_tensor * src1,
                      ggml_tensor *       dst) {
    cl_kernel k = b->kmgr->get("mul_mat/gemm_f16_f16", "gemm_f16_f16");
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

    const int num_wg_m = (int) ((ne1 + BM - 1) / BM);
    const int num_wg_n = (int) ((ne0 + BN - 1) / BN);
    call.global[0] = (size_t) (num_wg_m * num_wg_n) * NTH;
    call.global[1] = (size_t) ne2;
    call.global[2] = (size_t) ne3;

    ocl_pack_binary_args(call, src0, src1, dst, /*strides_u64=*/false);

    call.arg_i32((cl_int) num_wg_n);

    call.enqueue(b);
    return true;
}

}  // namespace ops
