#include "ggml-ocl-internal.h"
#include "ggml.h"

namespace ops {
static bool glu_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_GLU) {
        return false;
    }
    // 非 split 变体 (ggml_swiglu / ggml_geglu 等) 只有 src[0], src[1] 为 NULL;
    // swiglu kernel 需要两个独立输入, 这类一律回 CPU
    if (op->src[1] == nullptr) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    // kernel 只实现了 SWIGLU 且不支持 swapped; supports 必须和 run 一致,
    // 否则 dispatch 里会撞上 GGML_ASSERT(ok)
    ggml_glu_op glu_op;
    int32_t     swapped;
    memcpy(&glu_op, op->op_params + 0, sizeof(ggml_glu_op));
    memcpy(&swapped, op->op_params + sizeof(int32_t), sizeof(int32_t));
    if (glu_op != GGML_GLU_OP_SWIGLU || swapped != 0) {
        return false;
    }
    return true;
}

static bool swiglu_run(ggml_ocl_backend * b, const ggml_tensor* s0, const ggml_tensor* s1, ggml_tensor*dst) {
    const bool f16 = ocl_f16_packed(s0);
    cl_kernel k = ocl_pick_kernel(b, "glu/glu", "swiglu", "swiglu_f16", f16);
    if (k == nullptr) {
        return false;
    }

    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) s0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) s1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;
    GGML_ASSERT(e0 != nullptr && e1 != nullptr && ed != nullptr);

    const unsigned int nth  = 256;
    const int          ne01 = (int) s0->ne[1];
    const int          ne02 = (int) s0->ne[2];
    const int          ne03 = (int) s0->ne[3];

    ocl_kernel_call call;
    call.kernel    = k;
    call.op_name   = "GLU";
    call.ndims     = 3;
    call.global[0] = (size_t) ne01 * nth;
    call.global[1] = (size_t) ne02;
    call.global[2] = (size_t) ne03;
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(ocl_dev_offset(s0, e0));
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(ocl_dev_offset(s1, e1));
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(ocl_dev_offset(dst, ed));
    // kernel 对三个操作数用同一组步长, 依赖它们布局一致 (原有的隐含假设)
    call.arg_i32((cl_int) s0->ne[0]);
    call.arg_u64(ocl_nb64(s0, s0->nb[1]));
    call.arg_u64(ocl_nb64(s0, s0->nb[2]));
    call.arg_u64(ocl_nb64(s0, s0->nb[3]));

    call.enqueue(b);
    return true;
}

static bool glu_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) {
    ggml_glu_op op;
    memcpy(&op, dst->op_params + 0, sizeof(ggml_glu_op));
    if (op != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    return swiglu_run(b, s0, s1, dst);
}

}  // namespace ops

extern const ocl_op ocl_ops_op_glu[] = {
    { GGML_OP_GLU, 0, ops::glu_supports, ops::glu_run },
    { GGML_OP_NONE, 0, nullptr, nullptr }, // 哨兵终止
};
