#include "ggml-ocl-internal.h"
#include <CL/cl_platform.h>

#include <cstdio>

namespace ops {
static bool rope_supports(const ggml_ocl_caps * caps, const ggml_tensor * op) {
    (void) caps;
    if (op->op != GGML_OP_ROPE) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) {
        return false;
    }
    // fp16 存储下 F32 的 freq_factors 在设备上也是 half, 而 _f16 kernel 按 float 读它。
    // 带 freq_factors 的用例回退 CPU, 不走错路。
    if (ocl_f16_packed(op->src[0]) && op->src[2] != nullptr) {
        return false;
    }
    return true;
}

static bool rope_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) {
    ggml_ocl_tensor_extra * extra0 = (ggml_ocl_tensor_extra *)s0->extra;
    ggml_ocl_tensor_extra * extra1 = (ggml_ocl_tensor_extra *)s1->extra;
    ggml_ocl_tensor_extra * extrad = (ggml_ocl_tensor_extra *)dst->extra;

    cl_ulong offset0 = ocl_dev_offset(s0, extra0);
    cl_ulong offset1 = ocl_dev_offset(s1, extra1);
    cl_ulong offsetd = ocl_dev_offset(dst, extrad);

    ggml_tensor * src2 = dst->src[2];
    ggml_ocl_tensor_extra * extra2 = src2 ? (ggml_ocl_tensor_extra *)src2->extra : nullptr;

    cl_ulong offset2 = extra2 ? ocl_dev_offset(src2, extra2) : offset0;

    const int  ne00 = s0 ? s0->ne[0] : 0;
    const int  ne01 = s0 ? s0->ne[1] : 0;
    const int  ne02 = s0 ? s0->ne[2] : 0;
    const int  ne03 = s0 ? s0->ne[3] : 0;

    const cl_ulong  nb00 = s0 ? ocl_nb64(s0, s0->nb[0]) : 0;
    const cl_ulong  nb01 = s0 ? ocl_nb64(s0, s0->nb[1]) : 0;
    const cl_ulong  nb02 = s0 ? ocl_nb64(s0, s0->nb[2]) : 0;
    const cl_ulong  nb03 = s0 ? ocl_nb64(s0, s0->nb[3]) : 0;

    const int ne10 = s1 ? s1->ne[0] : 0;
    const int ne11 = s1 ? s1->ne[1] : 0; 
    const int ne12 = s1 ? s1->ne[2] : 0; 
    const int ne13 = s1 ? s1->ne[3] : 0;

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;
    const int  ne2 = dst ? dst->ne[2] : 0;
    const int  ne3 = dst ? dst->ne[3] : 0;

    const cl_ulong  nb0 = dst ? ocl_nb64(dst, dst->nb[0]) : 0;
    const cl_ulong  nb1 = dst ? ocl_nb64(dst, dst->nb[1]) : 0;
    const cl_ulong  nb2 = dst ? ocl_nb64(dst, dst->nb[2]) : 0;
    const cl_ulong  nb3 = dst ? ocl_nb64(dst, dst->nb[3]) : 0;

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

    // kernel 后缀由"设备上的有效元素类型"决定: F32 未打包 -> _f32; F16 真实张量
    // 或 fp16 存储下的 F32 -> _f16 (两者在显存里都是 half, 共用同一批 kernel)
    const char * rope_variant = (ocl_f16_packed(s0) || s0->type == GGML_TYPE_F16) ? "_f16" : "_f32";
    const char * rope_base    = is_neox                   ? "kernel_rope_neox"   :
                                (is_mrope && !is_vision)  ? "kernel_rope_multi"  :
                                is_vision                 ? "kernel_rope_vision" :
                                                            "kernel_rope_norm";

    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name), "%s%s", rope_base, rope_variant);

    cl_kernel kernel = b->kmgr->get("rope/rope", kernel_name);
    if (kernel == nullptr) {
        GGML_LOG_ERROR("rope: cannot find kernel %s\n", kernel_name);
        return false;
    }

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