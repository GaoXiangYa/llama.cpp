#include "ggml-ocl-internal.h"
#include "ggml.h"

namespace ops {

static ggml_type cpy_dev_type(const ggml_tensor * t) {
    return ocl_f16_packed(t) ? GGML_TYPE_F16 : t->type;
}

static bool cpy_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_CONT && op->op != GGML_OP_CPY && op->op != GGML_OP_DUP) {
        return false;
    }
    if (op->src[0] == nullptr || ggml_is_empty(op)) {
        return false;
    }
    if (ggml_nelements(op->src[0]) != ggml_nelements(op)) {
        return false;
    }

    const ggml_type dt0 = cpy_dev_type(op->src[0]);
    const ggml_type dt1 = cpy_dev_type(op);
    if ((dt0 != GGML_TYPE_F16 && dt0 != GGML_TYPE_F32) || (dt1 != GGML_TYPE_F16 && dt1 != GGML_TYPE_F32)) {
        return false;
    }

    if (ggml_is_contiguous(op)) {
        return true;
    }
    return op->src[0]->ne[0] == op->ne[0] && op->src[0]->ne[1] == op->ne[1] && op->src[0]->ne[2] == op->ne[2] &&
           op->src[0]->ne[3] == op->ne[3];
}

static bool cpy_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) {
    GGML_UNUSED(s1);

    const bool src_half = cpy_dev_type(s0) == GGML_TYPE_F16;
    const bool dst_half = cpy_dev_type(dst) == GGML_TYPE_F16;

    const char * fn = src_half ? (dst_half ? "cpy_f16_f16" : "cpy_f16_f32") :
                                 (dst_half ? "cpy_f32_f16" : "cpy_f32_f32");

    cl_kernel k = b->kmgr->get("cpy/cpy", fn);
    if (k == nullptr) {
        return false;
    }

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) s0->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;
    GGML_ASSERT(e0 != nullptr && ed != nullptr);

    const unsigned int nth = MIN(64u, (unsigned int) s0->ne[0]);

    ocl_kernel_call call;
    call.kernel    = k;
    call.op_name   = "CPY";
    call.ndims     = 3;
    call.global[0] = (size_t) s0->ne[1] * nth;
    call.global[1] = (size_t) s0->ne[2];
    call.global[2] = (size_t) s0->ne[3];
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(ocl_dev_offset(s0, e0));
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(ocl_dev_offset(dst, ed));
    call.arg_i32((cl_int) s0->ne[0]);
    call.arg_i32((cl_int) s0->ne[1]);
    call.arg_i32((cl_int) s0->ne[2]);
    call.arg_i32((cl_int) s0->ne[3]);
    call.arg_u64(ocl_nb64(s0, s0->nb[0]));
    call.arg_u64(ocl_nb64(s0, s0->nb[1]));
    call.arg_u64(ocl_nb64(s0, s0->nb[2]));
    call.arg_u64(ocl_nb64(s0, s0->nb[3]));
    call.arg_u64(ocl_nb64(dst, dst->nb[0]));
    call.arg_u64(ocl_nb64(dst, dst->nb[1]));
    call.arg_u64(ocl_nb64(dst, dst->nb[2]));
    call.arg_u64(ocl_nb64(dst, dst->nb[3]));
    call.arg_i32(ggml_is_contiguous(dst) ? 1 : 0);

    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_cpy[] = {
    { GGML_OP_CONT, 0, ops::cpy_supports, ops::cpy_run },
    { GGML_OP_CPY,  0, ops::cpy_supports, ops::cpy_run },
    { GGML_OP_DUP,  0, ops::cpy_supports, ops::cpy_run },
    { GGML_OP_NONE, 0, nullptr,           nullptr      }, // 哨兵终止
};
