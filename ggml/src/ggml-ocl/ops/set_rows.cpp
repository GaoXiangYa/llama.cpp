#include "ggml-ocl-internal.h"
#include "ggml.h"
#include <CL/cl_platform.h>

namespace ops {
static bool set_rows_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    return true;
}

static bool set_rows_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const char* kernel_name = dst->type == GGML_TYPE_F32 ? "set_rows_f32_i64_f32" : "set_rows_f16_i64_f16";
    cl_kernel k = b->kmgr->get("set_rows/set_rows", kernel_name);
    if (k == nullptr) {
        return false;
    }

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;
    
    ocl_kernel_call call;
    call.kernel    = k;
    call.op_name   = "SET_ROWS";
    call.ndims     = 3;

    constexpr int nth = 256;

    GGML_TENSOR_BINARY_OP_LOCALS;
    
    cl_ulong offset0 = e0->offset + src0->view_offs;
    cl_ulong offset1 = e1->offset + src1->view_offs;
    cl_ulong offsetd = ed->offset + dst->view_offs;

    int nblk0 = ne0 / ggml_blck_size(dst->type);

    call.global[0] = (size_t) ne01 * nth;
    call.global[1] = (size_t) ne02;
    call.global[2] = (size_t) ne03;
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_u64(nb00);
    call.arg_u64(nb01);
    call.arg_u64(nb02);
    call.arg_u64(nb03);
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_u64(nb10);
    call.arg_u64(nb11);
    call.arg_u64(nb12);
    call.arg_u64(nb13);
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_u64(nb0);
    call.arg_u64(nb1);
    call.arg_u64(nb2);
    call.arg_u64(nb3);
    call.arg_i32(nblk0);

    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_set_rows[] = {
    { GGML_OP_SET_ROWS, 0, ops::set_rows_supports, ops::set_rows_run },
};
