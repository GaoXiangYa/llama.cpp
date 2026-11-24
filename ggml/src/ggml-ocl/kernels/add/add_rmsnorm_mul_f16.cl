#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline float group_reduce_add(float val, local float * sdata) {
    const int lid       = get_local_id(0);
    const int warp_nums = get_num_sub_groups();
    const int warp_id   = lid >> 6;
    const int lane_id   = lid & 63;

    val = sub_group_reduce_add(val);
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    val = lid < warp_nums ? sdata[lid] : 0.0f;
    if (warp_id == 0) {
        val = sub_group_reduce_add(val);
    }
    return val;
}

kernel void add_rmsnorm_mul_f16(global char * src0,
                                ulong         offset0,
                                global char * src1,
                                ulong         offset1,
                                global char * srcw,
                                ulong         offsetw,
                                global char * dst0,
                                ulong         offsetd0,
                                global char * dst1,
                                ulong         offsetd1,
                                int           ne00,
                                int           ne01,
                                int           ne02,
                                int           ne03,
                                ulong         nb00,
                                ulong         nb01,
                                ulong         nb02,
                                ulong         nb03,
                                int           ne10,
                                int           ne11,
                                int           ne12,
                                int           ne13,
                                ulong         nb10,
                                ulong         nb11,
                                ulong         nb12,
                                ulong         nb13,
                                int           ne0,
                                int           ne1,
                                int           ne2,
                                int           ne3,
                                ulong         nb0,
                                ulong         nb1,
                                ulong         nb2,
                                ulong         nb3,
                                float         eps) {
    const int i3  = get_group_id(2);
    const int i2  = get_group_id(1);
    const int i1  = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    const int i13 = (ne03 != ne13) ? i3 % ne13 : i3;
    const int i12 = (ne02 != ne12) ? i2 % ne12 : i2;
    const int i11 = (ne01 != ne11) ? i1 % ne11 : i1;

    global char * pa   = src0 + offset0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    global char * pb   = src1 + offset1 + i13 * nb13 + i12 * nb12 + i11 * nb11;
    global half * pw   = (global half *) (srcw + offsetw);
    global half * pres = (global half *) (dst0 + offsetd0 + i3 * nb3 + i2 * nb2 + i1 * nb1);
    global half * pout = (global half *) (dst1 + offsetd1 + i3 * nb3 + i2 * nb2 + i1 * nb1);

    local float sdata[64];
    float sum_sq = 0.0f;

    // pass 1: residual = a + b, computed in float (half would overflow on large activations)
    for (int i = lid; i < ne0; i += lsz) {
        const int   i10 = i % ne10;
        const float a   = convert_float(*(const global half *) (pa + (size_t) i * nb00));
        const float b   = convert_float(*(const global half *) (pb + (size_t) i10 * nb10));
        const float sum = a + b;
        pres[i] = convert_half(sum);
        sum_sq += sum * sum;
    }

    sum_sq = group_reduce_add(sum_sq, sdata);

    local float s_rms;
    if (lid == 0) {
        s_rms = 1.0f / sqrt(sum_sq / ne0 + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // pass 2: read the residual back, normalize and scale by the norm weight
    for (int i = lid; i < ne0; i += lsz) {
        pout[i] = convert_half(convert_float(pres[i]) * s_rms * convert_float(pw[i]));
    }
}
