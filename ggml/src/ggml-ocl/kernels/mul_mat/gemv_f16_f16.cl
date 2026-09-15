#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// fp16 存储: src0 本来就是 F16 (KV cache), src1/dst 是 F32 -> half
kernel void gemv_f16_f16(
        global const char* src0, ulong offset0,
        global const char* src1, ulong offset1,
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

    const int i01 = i1 / (ne12 / ne02);
    const int i02 = i2 / (ne13 / ne03);

    const int warp_size = get_sub_group_size();
    const int warp_id = get_sub_group_id();
    const int lane_id = get_sub_group_local_id();
    const int g_row = i0 * get_num_sub_groups() + warp_id;
    if (g_row >= ne01) {
        return;
    }

    global const half* src0_ptr = (global const half*)(src0 + g_row * nb01 + i01 * nb02 + i02 * nb03);
    global const half* src1_ptr = (global const half*)(src1 + i11 * nb12 + i12 * nb13);
    global half* dst_ptr = (global half*)(dst + i1 * nb2 + i2 * nb3);

    float sum = 0.0f;
    for (int i = lane_id; i < ne00; i += warp_size) {
        sum += convert_float(src0_ptr[i]) * convert_float(src1_ptr[i]);
    }
    sum = sub_group_reduce_add(sum);

    if (lane_id == 0) {
        dst_ptr[g_row] = convert_half(sum);
    }
}
