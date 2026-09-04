#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void gemv_q4_1(
        global const char* src0, ulong offset0,
        global const char* src1, ulong offset1,
        global char* dst, ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        int nb00, int nb01, int nb02, int nb03,
        int ne10, int ne11, int ne12, int ne13,
        int nb10, int nb11, int nb12, int nb13,
        int ne0, int ne1, int ne2, int ne3,
        int nb0, int nb1, int nb2, int nb3,
        int blk_k, int blk_bytes, int blks_per_row
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    const int i0 = get_group_id(0);
    const int i1 = get_group_id(1);
    const int i2 = get_group_id(2);

    const int i11 = i1;
    const int i12 = i2;

    const int i01 = i1 / ne02;
    const int i02 = i2 / ne03;

    const int warp_size = get_sub_group_size();
    const int warp_id   = get_sub_group_id();
    const int lane_id   = get_sub_group_local_id();
    const int g_row = i0 * get_num_sub_groups() + warp_id;
    if (g_row >= ne01) {
        return;
    }

    global const uchar* src0_row = (global const uchar*)(src0 + g_row * nb01 + i01 * nb02 + i02 * nb03);
    global const float* src1_ptr = (global const float*)(src1 + i11 * nb12 + i12 * nb13);
    global float* dst_ptr = (global float*) (dst + i1 * nb1 + i12 * nb2);

    float sum = 0.0f;
    for (int blk = lane_id; blk < blks_per_row; blk += warp_size) {
        global const uchar* b = src0_row + blk * blk_bytes;
        const half  d = vload_half(0, (const half*) b);       // scale
        const half  m = vload_half(1, (const half*) b);       // min
        global const uchar* qs = b + 4;
        const int col = blk * blk_k;

        const float fd = (float) d;
        const float fm = (float) m;
        for (int j = 0; j < blk_k / 2; ++j) {
            const uchar packed = qs[j];
            const float v0 = (float) (packed & 0x0F);
            const float v1 = (float) (packed >> 4);
            sum += (v0 * fd + fm) * src1_ptr[col + j];
            sum += (v1 * fd + fm) * src1_ptr[col + blk_k / 2 + j];
        }
    }
    sum = sub_group_reduce_add(sum);

    if (lane_id == 0) {
        dst_ptr[g_row] = sum;
    }
}
