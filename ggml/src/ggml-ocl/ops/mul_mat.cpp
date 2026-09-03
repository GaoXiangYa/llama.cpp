
#include "ggml-ocl-internal.h"
#include "ggml.h"

#include <CL/cl_platform.h>

namespace ops {
static bool mul_mat_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    // gemv kernel 仅支持 f32 x f32 单列 decode (src1: [K, 1]);
    // Q4_1 等量化权重与 prefill (多列) 回 CPU
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[1]->ne[1] != 1) {
        return false;
    }
    return true;
}

static bool gemv_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cl_kernel k = b->kmgr->get("mul_mat/gemv", "gemv");
    if (k == nullptr) {
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

    constexpr int subgroup_size = 64;
    constexpr int rows_per_group = nth / subgroup_size;
    const int num_groups = (ne0 + rows_per_group - 1) / rows_per_group;
    call.global[0] = num_groups * nth;
    call.global[1] = ne2;
    call.global[2] = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = e0->offset + src0->view_offs;
    cl_ulong offset1 = e1->offset + src1->view_offs;
    cl_ulong offsetd = ed->offset + dst->view_offs;

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
    call.arg_i32(nb00);
    call.arg_i32(nb01);
    call.arg_i32(nb02);
    call.arg_i32(nb03);
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(nb10);
    call.arg_i32(nb11);
    call.arg_i32(nb12);
    call.arg_i32(nb13);
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(nb0);
    call.arg_i32(nb1);
    call.arg_i32(nb2);
    call.arg_i32(nb3);

    call.enqueue(b);
    return true;
}

static bool mul_mat_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // supports 已保证: f32 x f32 单列 (dst->ne[1] == 1)
    return gemv_run(b, src0, src1, dst);
}

}  // namespace ops

extern const ocl_op ocl_ops_op_mul_mat[] = {
    { GGML_OP_MUL_MAT, 0, ops::mul_mat_supports, ops::mul_mat_run },
    { GGML_OP_NONE, 0, nullptr, nullptr },   // 哨兵终止 (遍历依赖)
};
