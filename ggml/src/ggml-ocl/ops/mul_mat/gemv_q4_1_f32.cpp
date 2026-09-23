#include "mul_mat.h"

bool gemv_q4_1_f32_run(ggml_ocl_backend *  b,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor *       dst) {
    cl_kernel k = b->kmgr->get("mul_mat/gemv_q4_1_f32", "gemv_q4_1_f32");
    if (k == nullptr) {
        GGML_LOG_ERROR("cannot find gemv_q4_1_f32 kernel");
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

    constexpr int subgroup_size  = 64;
    constexpr int rows_per_group = nth / subgroup_size;
    const int     num_groups     = (ne0 + rows_per_group - 1) / rows_per_group;
    call.global[0]               = num_groups * nth;
    call.global[1]               = ne2;
    call.global[2]               = ne3;

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) src0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) src1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;

    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst, ed);

    constexpr int blk_k        = 32;
    const int     blks_per_row = ne00 / blk_k;
    constexpr int blk_bytes    = 20;

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
    call.arg_i32(ocl_nb(src0, nb00));
    call.arg_i32(ocl_nb(src0, nb01));
    call.arg_i32(ocl_nb(src0, nb02));
    call.arg_i32(ocl_nb(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_i32(ocl_nb(src1, nb10));
    call.arg_i32(ocl_nb(src1, nb11));
    call.arg_i32(ocl_nb(src1, nb12));
    call.arg_i32(ocl_nb(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_i32(ocl_nb(dst, nb0));
    call.arg_i32(ocl_nb(dst, nb1));
    call.arg_i32(ocl_nb(dst, nb2));
    call.arg_i32(ocl_nb(dst, nb3));
    call.arg_i32(blk_k);
    call.arg_i32(blk_bytes);
    call.arg_i32(blks_per_row);

    call.enqueue(b);
    return true;
}
