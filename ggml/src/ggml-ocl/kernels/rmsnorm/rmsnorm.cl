#pragma OPENCL EXTENSION cl_khr_subgroups : enable

inline float group_reduce_add(float val, local float* sdata) {
    const int lid = get_local_id(0);
    const int warp_nums = get_num_sub_groups();
    const int warp_id = lid >> 6;
    const int lane_id = lid & 63;

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



kernel void rmsnorm(global char * src0,
                    ulong         offset0,
                    global char * dst,
                    ulong         offsetd,
                    int           ne0,
                    int           nb1,
                    int           nb2,
                    int           nb3,
                    float eps) {
    const int i1 = get_group_id(0);
    const int i2 = get_group_id(1);
    const int i3 = get_group_id(2);
    const int i0 = get_local_id(0);
    const int lsz = get_local_size(0);

    global float* psrc0 = (global float*)(src0 + offset0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    global float* pdst = (global float*)(dst + offsetd + i1 * nb1 + i2 * nb2 + i3 * nb3);

    local float sdata[64];
    float sum = 0.0f;
    for (int i = i0; i < ne0; i += lsz) {
        sum += psrc0[i] * psrc0[i];
    }
    sum = group_reduce_add(sum, sdata);

    local float s_rms;
    if (i0 == 0) {
        s_rms = 1.0f / sqrt(sum / ne0 + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = i0; i < ne0; i += lsz) {
        pdst[i] = psrc0[i] * s_rms;
    }
}
