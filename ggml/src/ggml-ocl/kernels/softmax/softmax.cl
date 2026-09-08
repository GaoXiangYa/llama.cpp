#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define MAX(a, b) a < b ? b : a
#define WARP_SIZE 64

inline float group_reduce_max(float val, local float * sdata) {
    const int i0        = get_local_id(0);
    const int warp_nums = get_num_sub_groups();
    const int warp_id   = i0 >> 6;
    const int lane_id   = i0 & (WARP_SIZE - 1);

    val = sub_group_reduce_max(val);
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    val = i0 < warp_nums ? sdata[i0] : -INFINITY;

    if (warp_id == 0) {
        val = sub_group_reduce_max(val);
    }

    return val;
}

inline float group_reduce_add(float val, local float * sdata) {
    const int i0        = get_local_id(0);
    const int warp_nums = get_num_sub_groups();
    const int warp_id   = i0 >> 6;
    const int lane_id   = i0 & (WARP_SIZE - 1);

    val = sub_group_reduce_add(val);
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    val = i0 < warp_nums ? sdata[i0] : 0.0f;

    if (warp_id == 0) {
        val = sub_group_reduce_add(val);
    }

    return val;
}

kernel void softmax(global char * src0,
                    ulong         offset0,
                    global char * src1,
                    ulong         offset1,
                    global char * src2,
                    ulong         offset2,
                    global char * dst,
                    ulong         offsetd,
                    int           ne12,
                    int           ne13,
                    int           nb11,
                    int           nb12,
                    int           nb13,
                    int           ne0,
                    int           nb1,
                    int           nb2,
                    int           nb3,
                    int has_mask,
                    int has_sinks,
                    float         scale,
                    float         max_bias,
                    float         m0,
                    float         m1,
                    int           n_head_log2) {
    const int i1  = get_group_id(0);
    const int i2  = get_group_id(1);
    const int i3  = get_group_id(2);
    const int i0  = get_local_id(0);
    const int lsz = get_local_size(0);

    const int i11 = i1;
    const int i12 = ne12 > 0 ? i2 % ne12 : 0;
    const int i13 = ne13 > 0 ? i3 % ne13 : 0;

    global float * src0_ptr = (global float *) (src0 + offset0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    global float * src1_ptr = (global float *) (src1 + offset1 + i11 * nb11 + i12 * nb12 + i13 * nb13);
    global float * src2_ptr = (global float *) (src2 + offset2);
    global float * dst_ptr  = (global float *) (dst + offsetd + i1 * nb1 + i2 * nb2 + i3 * nb3);

    // ALiBi
    float slope = 0.0f;
    if (max_bias > 0.0f) {
        int h = i2;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    local float sdata_max[WARP_SIZE];
    local float sdata_sum[WARP_SIZE];
    local float s_max;
    local float s_sum;

    float max_num = src2_ptr ? src2_ptr[i2] : -INFINITY;
    for (int i = i0; i < ne0; i += lsz) {
        max_num = MAX(max_num, src0_ptr[i] * scale + (has_mask ? slope * src1_ptr[i] : 0.0f));
    }

    max_num = lsz <= 64 ? sub_group_reduce_max(max_num) : group_reduce_max(max_num, sdata_max);

    if (i0 == 0) {
        s_max = max_num;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum_num = 0.0f;
    for (int i = i0; i < ne0; i += lsz) {
        float tmp = src0_ptr[i] * scale + (has_mask ? slope * src1_ptr[i] : 0.0f);
        float expv = exp(tmp - s_max);
        dst_ptr[i] = expv;
        sum_num += expv;
    }

    sum_num = lsz <= 64 ? sub_group_reduce_add(sum_num) : group_reduce_add(sum_num, sdata_sum);

    if (has_sinks) {
        sum_num += exp(src2_ptr[i2] - s_max);
    }
    if (i0 == 0) {
        s_sum = 1.0f / sum_num;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = i0; i < ne0; i += lsz) {
        dst_ptr[i] = dst_ptr[i] * s_sum;
    }
}
