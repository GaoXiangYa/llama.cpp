#include "ggml-ocl-internal.h"
#include "ggml.h"

#include <CL/cl_platform.h>

namespace ops {
static bool get_rows_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    return true;
}

static bool get_rows_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cl_kernel k = b->kmgr->get("get_rows/get_rows", "get_rows_f32_i32_f32");
    if (k == nullptr) {
        return false;
    }

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    ocl_kernel_call call;
    call.kernel  = k;
    call.op_name = "GET_ROWS";
    call.ndims   = 3;

    constexpr int nth = 256;

    GGML_TENSOR_BINARY_OP_LOCALS;

    cl_ulong offset0 = e0->offset + src0->view_offs;
    cl_ulong offset1 = e1->offset + src1->view_offs;
    cl_ulong offsetd = ed->offset + dst->view_offs;

    int nblk0 = ne0 / ggml_blck_size(dst->type);

    call.global[0] = (size_t) ne1 * nth;
    call.global[1] = (size_t) ne2;
    call.global[2] = (size_t) ne3;
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(offsetd);
    call.arg_u64(nb01);
    call.arg_u64(nb02);
    call.arg_u64(nb03);
    call.arg_u64(nb11);
    call.arg_u64(nb12);
    call.arg_u64(nb13);
    call.arg_u64(nb1);
    call.arg_u64(nb2);
    call.arg_u64(nb3);
    call.arg_i32(nblk0);

    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_get_rows[] = {
    { GGML_OP_GET_ROWS, 0, ops::get_rows_supports, ops::get_rows_run },
};
