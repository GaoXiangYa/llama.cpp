#include "ggml-ocl-internal.h"
#include "ggml.h"

namespace ops {
static bool rmsnorm_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_RMS_NORM) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

static bool rmsnorm_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor* s1, ggml_tensor * dst) {
    (void) s1;
    cl_kernel k = b->kmgr->get("rmsnorm/rmsnorm", "rmsnorm");
    if (k == nullptr) {
        return false;
    }

    ggml_ocl_tensor_extra * e0       = (ggml_ocl_tensor_extra *) s0->extra;
    ggml_ocl_tensor_extra * ed       = (ggml_ocl_tensor_extra *) dst->extra;

    const unsigned int nth  = MIN(256u, (unsigned int) s0->ne[0]);
    const int          ne01 = (int) s0->ne[1];
    const int          ne02 = (int) s0->ne[2];
    const int          ne03 = (int) s0->ne[3];

    float eps = 0.0f;
    memcpy(&eps, dst->op_params + 0, sizeof(float));

    ocl_kernel_call call;
    call.kernel    = k;
    call.op_name   = "rmsnorm";
    call.ndims     = 3;
    call.global[0] = (size_t) ne01 * nth;
    call.global[1] = (size_t) ne02;
    call.global[2] = (size_t) ne03;
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(e0->offset + s0->view_offs);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(ed->offset + dst->view_offs);

    call.arg_i32((cl_int) dst->ne[0]);
    call.arg_i32((cl_int) dst->nb[1]);
    call.arg_i32((cl_int) dst->nb[2]);
    call.arg_i32((cl_int) dst->nb[3]);

    call.arg_f32(eps);
    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_rmsnorm[] = {
    { GGML_OP_RMS_NORM, 0, ops::rmsnorm_supports, ops::rmsnorm_run },
    { GGML_OP_NONE,     0, nullptr,               nullptr          }, // 哨兵终止 (遍历依赖)
};