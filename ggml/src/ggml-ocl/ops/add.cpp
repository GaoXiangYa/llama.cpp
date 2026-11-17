#include "ggml-ocl-internal.h"
#include "ops/ops.h"

namespace ops {
bool add_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_ADD) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

bool add_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) {
    cl_kernel k = b->kmgr->get("add/kernel_add_f16", "kernel_add_f16");
    if (k == nullptr) {
        return false;
    }

    // 3D grid: 每 workgroup 处理 src0 的一行; src1 的广播在 kernel 内取模
    const unsigned int nth  = MIN(64u, (unsigned int) s0->ne[0]);
    const int          ne01 = (int) s0->ne[1];
    const int          ne02 = (int) s0->ne[2];
    const int          ne03 = (int) s0->ne[3];

    ocl_kernel_call call;
    call.kernel    = k;
    call.op_name   = "ADD";
    call.ndims     = 3;
    call.global[0] = (size_t) ne01 * nth;
    call.global[1] = (size_t) ne02;
    call.global[2] = (size_t) ne03;
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;

    ocl_pack_binary_args(call, s0, s1, dst, /*strides_u64=*/true);

    call.enqueue(b);
    return true;
}

}  // namespace ops
