#define MAX(a, b) a < b ? b : a
#define WARP_SIZE 64

inline float group_reduce_max(float val, local float * sdata) {
    const int i0 = get_local_id(0);
    const int warp_nums = get_num_sub_groups();
    const int warp_id = i0 >> 6;
    const int lane_id = i0 & (WARP_SIZE - 1);

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
    const int i0 = get_local_id(0);
    const int warp_nums = get_num_sub_groups();
    const int warp_id = i0 >> 6;
    const int lane_id = i0 & (WARP_SIZE - 1);

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
               global char * dst,
               ulong         offsetd,
               int           ne0,
               int           ne1,
               int           ne2,
               int           ne3,
               int           nb0,
               int           nb1,
               int           nb2,
               int           nb3,
               float scale) {
    src0 += offset0;
    dst += offsetd;

    const int i1 = get_group_id(0);
    const int i2 = get_group_id(1);
    const int i3 = get_group_id(2);
    const int i0 = get_local_id(0);
    const int lsz = get_local_size(0);

    global float* src0_ptr = (global float*)(src0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    global float* dst_ptr = (global float*)(dst + i1 * nb1 + i2 * nb2 + i3 * nb3);

    local float sdata_max[WARP_SIZE];
    local float sdata_sum[WARP_SIZE];
    local float s_max;
    local float s_sum;

    float max_num = -INFINITY;
    for (int i = i0; i < ne0; i += lsz) {
        max_num = MAX(src0_ptr[i] * scale, max_num);
    }

    max_num = lsz <= 64 ? sub_group_reduce_max(max_num) : group_reduce_max(max_num, sdata_max);

    if (i0 == 0) {
        s_max = max_num;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum_num = 0.0f;
    for (int i = i0; i < ne0; i += lsz) {
        float tmp = src0_ptr[i] * scale;
        float expv = exp(tmp - s_max);
        dst_ptr[i] = expv;
        sum_num += expv;
    }

    sum_num = lsz <= 64 ? sub_group_reduce_add(sum_num) : group_reduce_add(sum_num, sdata_sum);
    if (i0 == 0) {
        s_sum = 1.0f / sum_num;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = i0; i < ne0; i += lsz) {
        dst_ptr[i] = dst_ptr[i] * s_sum;
    }
}
