kernel void gemv_f32_f32(
        global const char* src0, ulong offset0,
        constant char* src1, ulong offset1,
        global char* dst, ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        int nb00, int nb01, int nb02, int nb03,
        int ne10, int ne11, int ne12, int ne13,
        int nb10, int nb11, int nb12, int nb13,
        int ne0, int ne1, int ne2, int ne3,
        int nb0, int nb1, int nb2, int nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    const int i0 = get_group_id(0);
    const int i1 = get_group_id(1);
    const int i2 = get_group_id(2);

    const int i11 = i1;
    const int i12 = i2;

    const int i01 = i1 / ne02;
    const int i02 = i2 / ne03;

    const int warp_size = get_sub_group_size();
    const int warp_id = get_sub_group_id();
    const int lane_id = get_sub_group_local_id();
    const int g_row = i0 * get_num_sub_groups() + warp_id;
    if (g_row >= ne01) {
        return;
    }

    global const float* src0_ptr = (global const float*)(src0 + g_row * nb01 + i01 * nb02 + i02 * nb03);
    constant float* src1_ptr = (constant float*)(src1 + i11 * nb12 + i12 * nb13);
    global float* dst_ptr = (global float*)(dst + i1 * nb1 + i2 * nb2);

    float sum = 0.0f;
    for (int i = lane_id; i < ne00; i += warp_size) {
        sum += src0_ptr[i] * src1_ptr[i];
    }
    sum = sub_group_reduce_add(sum);

    if (lane_id == 0) {
        dst_ptr[g_row] = sum;
    }
}