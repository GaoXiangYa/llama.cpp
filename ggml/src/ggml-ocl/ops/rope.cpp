#include "ggml-ocl-internal.h"
#include <CL/cl_platform.h>

namespace ops {
static bool rope_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_ADD) {
        return false;
    }
    return true;
}

static bool rope_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) {
    ggml_ocl_tensor_extra * extra0 = (ggml_ocl_tensor_extra *)s0->extra;
    ggml_ocl_tensor_extra * extra1 = (ggml_ocl_tensor_extra *)s1->extra;
    ggml_ocl_tensor_extra * extrad = (ggml_ocl_tensor_extra *)dst->extra;

    cl_ulong offset0 = extra0->offset + s0->view_offs;
    cl_ulong offset1 = extra1->offset + s1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_tensor * src2 = dst->src[2];
    ggml_ocl_tensor_extra * extra2 = src2 ? (ggml_ocl_tensor_extra *)src2->extra : nullptr;

    cl_ulong offset2 = extra2 ? extra2->offset + src2->view_offs : offset0;

    const int  ne00 = s0 ? s0->ne[0] : 0;
    const int  ne01 = s0 ? s0->ne[1] : 0;
    const int  ne02 = s0 ? s0->ne[2] : 0;
    const int  ne03 = s0 ? s0->ne[3] : 0;

    const cl_ulong  nb00 = s0 ? s0->nb[0] : 0;
    const cl_ulong  nb01 = s0 ? s0->nb[1] : 0;
    const cl_ulong  nb02 = s0 ? s0->nb[2] : 0;
    const cl_ulong  nb03 = s0 ? s0->nb[3] : 0;

    const int ne10 = s1 ? s1->ne[0] : 0;
    const int ne11 = s1 ? s1->ne[1] : 0; 
    const int ne12 = s1 ? s1->ne[2] : 0; 
    const int ne13 = s1 ? s1->ne[3] : 0;

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;
    const int  ne2 = dst ? dst->ne[2] : 0;
    const int  ne3 = dst ? dst->ne[3] : 0;

    const cl_ulong  nb0 = dst ? dst->nb[0] : 0;
    const cl_ulong  nb1 = dst ? dst->nb[1] : 0;
    const cl_ulong  nb2 = dst ? dst->nb[2] : 0;
    const cl_ulong  nb3 = dst ? dst->nb[3] : 0;

    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    int nth = MIN(64, ne00);

    const int n_past     = ((int *) dst->op_params)[0];
    const int n_dims     = ((int *) dst->op_params)[1];
    const int mode       = ((int *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t sections[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int32_t)*4);

    const bool is_neox = mode & 2;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    const int  is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

    cl_kernel kernel;
    const char* kernel_name = nullptr;

    if (is_neox) {
        switch (s0->type) {
            case GGML_TYPE_F32:
                kernel_name = "kernel_rope_neox_f32";
                break;
            case GGML_TYPE_F16:
                kernel_name = "kernel_rope_neox_f16";
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_mrope && !is_vision) {
        switch (s0->type) {
            case GGML_TYPE_F32:
                kernel_name = "kernel_rope_multi_f32";
                break;
            case GGML_TYPE_F16:
                kernel_name = "kernel_rope_multi_f16";
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_vision) {
        switch (s0->type) {
            case GGML_TYPE_F32:
                kernel_name = "kernel_rope_vision_f32";
                break;
            case GGML_TYPE_F16:
                kernel_name = "kernel_rope_vision_f16";
                break;
            default:
                GGML_ASSERT(false);
        }
    } else {
        switch (s0->type) {
            case GGML_TYPE_F32:
                kernel_name = "kernel_rope_norm_f32";
                break;
            case GGML_TYPE_F16:
                kernel_name = "kernel_rope_norm_f16";
                break;
            default:
                GGML_ASSERT(false);
        };
    }

    kernel = b->kmgr->get("rope/rope", kernel_name);

    ocl_kernel_call call;
    call.kernel    = kernel;
    call.op_name   = "ROPE";
    call.ndims     = 3;
    call.global[0] = (size_t) ne01 * nth;
    call.global[1] = (size_t) ne02;
    call.global[2] = (size_t) ne03;
    call.local[0]  = nth;
    call.local[1]  = 1;
    call.local[2]  = 1;

    call.arg_cl_mem(extra0->data_device);
    call.arg_u64(offset0);
    call.arg_cl_mem(extra1->data_device);
    call.arg_u64(offset1);
    call.arg_cl_mem(extra2 ? extra2->data_device : extra0->data_device);
    call.arg_u64(offset2);
    call.arg_cl_mem(extrad->data_device);
    call.arg_u64(offsetd);
    call.arg_i32(ne00);
    call.arg_i32(ne01);
    call.arg_i32(ne02);
    call.arg_i32(ne03);
    call.arg_u64(nb00);
    call.arg_u64(nb01);
    call.arg_u64(nb02);
    call.arg_u64(nb03);
    call.arg_i32(ne0);
    call.arg_i32(ne1);
    call.arg_i32(ne2);
    call.arg_i32(ne3);
    call.arg_u64(nb0);
    call.arg_u64(nb1);
    call.arg_u64(nb2);
    call.arg_u64(nb3);
    call.arg_i32(n_past);
    call.arg_i32(n_dims);
    call.arg_i32(n_ctx_orig);
    call.arg_f32(freq_base);
    call.arg_f32(freq_scale);
    call.arg_f32(ext_factor);
    call.arg_f32(attn_factor);
    call.arg_f32(beta_fast);
    call.arg_f32(beta_slow);

    // both mrope and vision kernels have sections
    if (is_mrope || is_vision) {
        cl_int4 sec;
        sec.x = sections[0];
        sec.y = sections[1];
        sec.z = sections[2];
        sec.w = sections[3];
        call.arg_i32x4(sec);
    }
    // only mrope has is_imrope
    if (is_mrope && !is_vision) {
        call.arg_i32(is_imrope);
    }


    call.enqueue(b);
    return true;
}

}  // namespace ops

extern const ocl_op ocl_ops_op_rope[] = {
    { GGML_OP_ROPE, 0, ops::rope_supports, ops::rope_run },
    { GGML_OP_NONE, 0, nullptr, nullptr }, // 哨兵终止
};