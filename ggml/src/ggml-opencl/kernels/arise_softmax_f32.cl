kernel void kernel_arise_softmax_f32(
    global char * src0, ulong offset0,         // QK^T scores
    global char * src1, ulong offset1,         // mask（可为空）
    global char * src2, ulong offset2,         // 保留（可为空）
    global char * dst,  ulong offsetd,
    int n_kv,                                  // 每行元素数
    ulong stride_q, ulong stride_h, ulong stride_b,       // src0 strides
    int n_heads_mask, int n_batchs_mask,                        // mask dims
    ulong stride_mask_q, ulong stride_mask_h, ulong stride_mask_b,       // mask strides
    ulong stride_dst_q, ulong stride_dst_h, ulong stride_dst_b,          // dst strides
    float scale,                               // 1/√d_k
    float max_bias,                            // alibi 偏置
    float m0, float m1,                        // 保留
    int n_head_log2                            // 保留
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst = dst + offsetd;

    const int i_query = get_group_id(0);
    const int i_head = get_group_id(1);
    const int i_batch= get_group_id(2);

    const lid = get_local_id(0);
    const int local_size = get_local_size(0);

    const int i_mask_query = i_query;
    const int i_mask_head = i_head % ne12;
    const int i_mask_batch = i_batch % ne13;

    const __global float* psrc0 = (src0 + i_query * stride_q + i_head * stride_h + i_batch * stride_h);
    const __global float* pmask = (src1 != src0) ? (src1 + i_mask_query * stride_mask_q + i_mask_head * stride_mask_h + i_mask_batch * stride_mask_h) : 0;
    const __global float* psrc2 = (src2 != src0) ? (__global float*)src2 : 0;
    __global float* dst = (dst + i_query * stride_q + i_head * stride_h + i_batch * stride_h);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // step 1: parallel max
    float lmax = psrc2 ? psrc2[i_head] : -INFINITY;
    for (int i = lid; i < n_kv; i += local_size) {
        lmax = fmax(lmax, psrc0[i] * scale + pmask ? pmask[i] * slope : 0.0f);
    }
    float gmax = sub_group_reduce_max(lmax);

    // step 2 : parallel add
    float lsum = 0.0f;
    float exp_val = 0.0f;
    for (int i = lid; i < n_kv; i += local_size) {
        float val = psrc0[i] * scale + (pmask ? pmask[i] * slope + 0.0f);
        exp_val = exp(val - gmax);
        pdst[i] = exp_val;
        lsum += exp_val;
    }
    float inv_sum = 1.0f / sub_group_reduce_add(lsum);

    // step3 : normalize
    for (int i = lid; i < n_kv; i += local_size) {
        pdst[i] *= inv_sum;
    }

}