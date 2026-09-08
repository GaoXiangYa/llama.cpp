#include "ggml-ocl-internal.h"
#include "ggml.h"

namespace ops {
static bool softmax_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_SOFT_MAX) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

static bool softmax_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) {
    (void) s1;
    cl_kernel k = b->kmgr->get("softmax/softmax", "softmax");
    if (k == nullptr) {
        return false;
    }

    ggml_ocl_tensor_extra * e0       = (ggml_ocl_tensor_extra *) s0->extra;
    ggml_ocl_tensor_extra * e1       = s1 ? (ggml_ocl_tensor_extra *) s1->extra : nullptr;
    ggml_ocl_tensor_extra * e2       = dst->src[2] ? (ggml_ocl_tensor_extra *) dst->src[2]->extra : nullptr;
    ggml_ocl_tensor_extra * ed       = (ggml_ocl_tensor_extra *) dst->extra;
    float                   scale    = 0.0f;
    float                   max_bias = 0.0f;
    memcpy(&scale, dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, dst->op_params + 1, sizeof(float));

    const int n_head      = s0->ne[2];
    const int n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    ulong offset0 = e0->offset + s0->view_offs;
    ulong offset1 = s1 ? (e1->offset + s1->view_offs) : offset0;
    ulong offset2 = dst->src[2] ? (e2->offset + dst->src[2]->view_offs) : offset0;
    ulong offsetd = ed->offset + dst->view_offs;

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const unsigned int nth  = MIN(256u, (unsigned int) s0->ne[0]);
    const int          ne01 = (int) s0->ne[1];
    const int          ne02 = (int) s0->ne[2];
    const int          ne03 = (int) s0->ne[3];

    ocl_kernel_call call;
    call.kernel    = k;
    call.op_name   = "SOFTMAX";
    call.ndims     = 3;
    call.global[0] = (size_t) ne01 * nth;
    call.global[1] = (size_t) ne02;
    call.global[2] = (size_t) ne03;
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;
    int has_mask   = s1 ? 1 : 0;
    int has_sinks  = dst->src[2] ? 1 : 0;

    cl_int ne12 = s1 ? s1->ne[2] : 0;
    cl_int ne13 = s1 ? s1->ne[3] : 0;
    cl_int nb11 = s1 ? s1->nb[1] : 0;
    cl_int nb12 = s1 ? s1->nb[2] : 0;
    cl_int nb13 = s1 ? s1->nb[3] : 0;

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1 ? e1->data_device : e0->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(e2 ? e2->data_device : e0->data_device);
    call.arg_u64(offset2);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);

    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(nb11);
    call.arg_i32(nb12);
    call.arg_i32(nb13);

    call.arg_i32((cl_int) dst->ne[0]);
    call.arg_i32((cl_int) dst->nb[1]);
    call.arg_i32((cl_int) dst->nb[2]);
    call.arg_i32((cl_int) dst->nb[3]);

    call.arg_i32(has_mask);
    call.arg_i32(has_sinks);

    call.arg_f32(scale);
    call.arg_f32(max_bias);
    call.arg_f32(m0);
    call.arg_f32(m1);
    call.arg_i32(n_head_log2);

    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_softmax[] = {
    { GGML_OP_SOFT_MAX, 0, ops::softmax_supports, ops::softmax_run },
    { GGML_OP_NONE,     0, nullptr,               nullptr          }, // 哨兵终止 (遍历依赖)
};
