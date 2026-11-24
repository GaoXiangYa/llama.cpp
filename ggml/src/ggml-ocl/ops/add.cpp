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

bool add_rmsnorm_mul_run(ggml_ocl_backend *  b,
                         const ggml_tensor * add,
                         const ggml_tensor * norm,
                         const ggml_tensor * w,
                         ggml_tensor *       out) {
    cl_kernel k = b->kmgr->get("add/add_rmsnorm_mul_f16", "add_rmsnorm_mul_f16");
    if (k == nullptr) {
        return false;
    }

    const ggml_tensor * a = add->src[0];
    const ggml_tensor * c = add->src[1];
    if (a == nullptr || c == nullptr) {
        return false;
    }

    auto * ea = static_cast<ggml_ocl_tensor_extra *>(a->extra);
    auto * ec = static_cast<ggml_ocl_tensor_extra *>(c->extra);
    auto * ew = static_cast<ggml_ocl_tensor_extra *>(w->extra);
    auto * er = static_cast<ggml_ocl_tensor_extra *>(add->extra);
    auto * eo = static_cast<ggml_ocl_tensor_extra *>(out->extra);
    if (ea == nullptr || ec == nullptr || ew == nullptr || er == nullptr || eo == nullptr) {
        return false;
    }

    float eps = 0.0f;
    memcpy(&eps, norm->op_params, sizeof(float));

    const int nth  = (int) MIN(256u, (unsigned int) a->ne[0]);
    const int ne01 = (int) a->ne[1];
    const int ne02 = (int) a->ne[2];
    const int ne03 = (int) a->ne[3];

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

    call.arg_cl_mem(ea->data_device);
    call.arg_u64(ocl_dev_offset(a, ea));
    call.arg_cl_mem(ec->data_device);
    call.arg_u64(ocl_dev_offset(c, ec));
    call.arg_cl_mem(ew->data_device);
    call.arg_u64(ocl_dev_offset(w, ew));
    call.arg_cl_mem(er->data_device);
    call.arg_u64(ocl_dev_offset(add, er));
    call.arg_cl_mem(eo->data_device);
    call.arg_u64(ocl_dev_offset(out, eo));

    for (int j = 0; j < 4; j++) {
        call.arg_i32((cl_int) a->ne[j]);
    }
    for (int j = 0; j < 4; j++) {
        call.arg_u64(ocl_nb64(a, a->nb[j]));
    }
    for (int j = 0; j < 4; j++) {
        call.arg_i32((cl_int) c->ne[j]);
    }
    for (int j = 0; j < 4; j++) {
        call.arg_u64(ocl_nb64(c, c->nb[j]));
    }
    for (int j = 0; j < 4; j++) {
        call.arg_i32((cl_int) out->ne[j]);
    }
    for (int j = 0; j < 4; j++) {
        call.arg_u64(ocl_nb64(out, out->nb[j]));
    }

    call.arg_f32(eps);

    call.enqueue(b);
    return true;
}

}  // namespace ops
