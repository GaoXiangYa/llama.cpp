#include "ggml-ocl-internal.h"

namespace ops {
static bool add_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_ADD) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

static bool add_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) {
    cl_kernel k = b->kmgr->get("add/add", "kernel_add");
    if (k == nullptr) {
        return false;
    }

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) s0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) s1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;
    GGML_ASSERT(e0 != nullptr && e1 != nullptr && ed != nullptr);

    // 3D grid: 每 workgroup 处理 src0 的一行 (ADD 语义: dst 与 src0 同形状, src1 广播)
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

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(e0->offset + s0->view_offs);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(e1->offset + s1->view_offs);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(ed->offset + dst->view_offs);
    call.arg_i32((cl_int) s0->ne[0]);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_u64((cl_ulong) s0->nb[0]);
    call.arg_u64((cl_ulong) s0->nb[1]);
    call.arg_u64((cl_ulong) s0->nb[2]);
    call.arg_u64((cl_ulong) s0->nb[3]);
    call.arg_i32((cl_int) s1->ne[0]);
    call.arg_i32((cl_int) s1->ne[1]);
    call.arg_i32((cl_int) s1->ne[2]);
    call.arg_i32((cl_int) s1->ne[3]);
    call.arg_u64((cl_ulong) s1->nb[0]);
    call.arg_u64((cl_ulong) s1->nb[1]);
    call.arg_u64((cl_ulong) s1->nb[2]);
    call.arg_u64((cl_ulong) s1->nb[3]);
    call.arg_i32((cl_int) dst->ne[0]);
    call.arg_i32((cl_int) dst->ne[1]);
    call.arg_i32((cl_int) dst->ne[2]);
    call.arg_i32((cl_int) dst->ne[3]);
    call.arg_u64((cl_ulong) dst->nb[0]);
    call.arg_u64((cl_ulong) dst->nb[1]);
    call.arg_u64((cl_ulong) dst->nb[2]);
    call.arg_u64((cl_ulong) dst->nb[3]);

    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_add[] = {
    { GGML_OP_ADD, 0, ops::add_supports, ops::add_run },
};
