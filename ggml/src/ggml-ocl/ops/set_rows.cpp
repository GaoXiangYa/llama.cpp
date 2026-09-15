#include "ggml-ocl-internal.h"
#include "ggml.h"
#include <CL/cl_platform.h>

#include <cstring>

namespace ops {
static bool set_rows_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_SET_ROWS) {
        return false;
    }
    const ggml_type rows_type = op->src[0]->type;
    const ggml_type dst_type  = op->type;
    if (rows_type != GGML_TYPE_F32 && rows_type != GGML_TYPE_F16) {
        return false;
    }
    if (dst_type != GGML_TYPE_F32 && dst_type != GGML_TYPE_F16) {
        return false;
    }
    // kernel 里索引按 global long 读, 只对 I64 正确; I32 需要单独的变体
    if (op->src[1]->type != GGML_TYPE_I64) {
        return false;
    }
    return true;
}

static bool set_rows_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // fp16 存储下 packed 的 F32 在显存里就是 half, 所以设备上的有效类型是
    // F16 或 packed F32 -> half, 其余 F32 -> float。四种组合共用现有 kernel。
    const bool src_half = ocl_f16_packed(src0) || src0->type == GGML_TYPE_F16;
    const bool dst_half = ocl_f16_packed(dst)  || dst->type  == GGML_TYPE_F16;

    const char * kernel_name = nullptr;
    if (src_half && dst_half) {
        kernel_name = "set_rows_f16_i64_f16";
    } else if (dst_half) {
        kernel_name = "set_rows_f32_i64_f16";
    } else if (src_half) {
        kernel_name = "set_rows_f16_i64_f32";
    } else {
        kernel_name = "set_rows_f32_i64_f32";
    }

    // 拆出去的三个各自成源: set_rows/<函数名>.cl; 纯 f32 那个还在 set_rows/set_rows.cl
    cl_kernel k = ocl_pick_kernel(b, "set_rows/set_rows", "set_rows_f32_i64_f32", kernel_name,
                                  strcmp(kernel_name, "set_rows_f32_i64_f32") != 0);
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
    
    cl_ulong offset0 = ocl_dev_offset(src0, e0);
    cl_ulong offset1 = ocl_dev_offset(src1, e1);
    cl_ulong offsetd = ocl_dev_offset(dst,  ed);

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
    call.arg_u64(ocl_nb64(src0, nb00));
    call.arg_u64(ocl_nb64(src0, nb01));
    call.arg_u64(ocl_nb64(src0, nb02));
    call.arg_u64(ocl_nb64(src0, nb03));
    call.arg_i32(ne10);
    call.arg_i32(ne11);
    call.arg_i32(ne12);
    call.arg_i32(ne13);
    call.arg_u64(ocl_nb64(src1, nb10));
    call.arg_u64(ocl_nb64(src1, nb11));
    call.arg_u64(ocl_nb64(src1, nb12));
    call.arg_u64(ocl_nb64(src1, nb13));
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_u64(ocl_nb64(dst, nb0));
    call.arg_u64(ocl_nb64(dst, nb1));
    call.arg_u64(ocl_nb64(dst, nb2));
    call.arg_u64(ocl_nb64(dst, nb3));
    call.arg_i32(nblk0);

    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_set_rows[] = {
    { GGML_OP_SET_ROWS, 0, ops::set_rows_supports, ops::set_rows_run },
    { GGML_OP_NONE,     0, nullptr,                nullptr }, // 哨兵终止
};
