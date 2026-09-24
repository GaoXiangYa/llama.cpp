#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK4_1     32
#define QS4_1     16
#define BLK_BYTES 20

kernel void gemv_q4_1_f16(const global char * src0,
                          ulong               offset0,
                          const global char * src1,
                          ulong               offset1,
                          global char *       dst,
                          ulong               offsetd,
                          int                 ne00,
                          int                 ne01,
                          int                 ne02,
                          int                 ne03,
                          int                 nb00,
                          int                 nb01,
                          int                 nb02,
                          int                 nb03,
                          int                 ne10,
                          int                 ne11,
                          int                 ne12,
                          int                 ne13,
                          int                 nb10,
                          int                 nb11,
                          int                 nb12,
                          int                 nb13,
                          int                 ne0,
                          int                 ne1,
                          int                 ne2,
                          int                 ne3,
                          int                 nb0,
                          int                 nb1,
                          int                 nb2,
                          int                 nb3,
                          int                 blk_k,
                          int                 blk_bytes,
                          int                 blks_per_row) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    const int i0 = get_group_id(0);
    const int i1 = get_group_id(1);
    const int i2 = get_group_id(2);

    const int i11 = i1;
    const int i12 = i2;

    local int l_src0_off;
    if (get_local_id(0) == 0) {
        l_src0_off = (i1 / (ne12 / ne02)) * nb02 + (i2 / (ne13 / ne03)) * nb03;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int src0_off = l_src0_off;

    const int warp_id   = get_sub_group_id();
    const int lane_id   = get_sub_group_local_id();
    const int g_row     = (i0 * get_num_sub_groups() + warp_id) * 3;
    if (g_row >= ne01) {
        return;
    }

    const int row0 = g_row;
    const int row1 = min(g_row + 1, ne01 - 1);
    const int row2 = min(g_row + 2, ne01 - 1);

    const global uchar * src0_row0 = (const global uchar *) (src0 + row0 * nb01 + src0_off);
    const global uchar * src0_row1 = (const global uchar *) (src0 + row1 * nb01 + src0_off);
    const global uchar * src0_row2 = (const global uchar *) (src0 + row2 * nb01 + src0_off);

    const global half * src1_ptr = (const global half *) (src1 + i11 * nb12 + i12 * nb13);
    global half *       dst_ptr  = (global half *)       (dst  + i1  * nb2  + i2  * nb3);

    const int qs_total    = blks_per_row * QS4_1;
    const int qs_per_lane = qs_total >> 6;
    int       i           = lane_id * qs_per_lane;
    const int i_end       = i + qs_per_lane;

    half sum0 = 0.0;
    half sum1 = 0.0;
    half sum2 = 0.0;

    while (i < i_end) {
        const int blk  = i / QS4_1;
        const int boff = i % QS4_1;

        const global uchar * b0 = src0_row0 + blk * BLK_BYTES;
        const global uchar * b1 = src0_row1 + blk * BLK_BYTES;
        const global uchar * b2 = src0_row2 + blk * BLK_BYTES;

        const global half * x = src1_ptr + blk * QK4_1;

        const half2 dm0 = *((const global half2 *) b0);
        const half2 dm1 = *((const global half2 *) b1);
        const half2 dm2 = *((const global half2 *) b2);

        half  s    = 0.0;
        half2 acc0 = (half2) (0.0, 0.0);
        half2 acc1 = (half2) (0.0, 0.0);
        half2 acc2 = (half2) (0.0, 0.0);

        const int           chunk = min(QS4_1 - boff, i_end - i);
        global const uint * w0    = (const global uint *) (&b0[4 + boff]);
        global const uint * w1    = (const global uint *) (&b1[4 + boff]);
        global const uint * w2    = (const global uint *) (&b2[4 + boff]);

        for (int j = 0; j < (chunk >> 2); ++j) {
            const uint ww0 = w0[j];
            const uint ww1 = w1[j];
            const uint ww2 = w2[j];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int   k  = (j << 2) + q;
                const half2 xs = (half2) (x[boff + k], x[boff + QS4_1 + k]);  // 3 行共用, 只取一次
                s += xs.x;                                                    // 只加一次
                s += xs.y;

                const uchar p0 = (uchar) (ww0 >> (q << 3));
                const uchar p1 = (uchar) (ww1 >> (q << 3));
                const uchar p2 = (uchar) (ww2 >> (q << 3));

                acc0 = fma((half2) ((half) (p0 & 0x0F), (half) (p0 >> 4)), xs, acc0);
                acc1 = fma((half2) ((half) (p1 & 0x0F), (half) (p1 >> 4)), xs, acc1);
                acc2 = fma((half2) ((half) (p2 & 0x0F), (half) (p2 >> 4)), xs, acc2);
            }
        }

        sum0 = fma(acc0.x + acc0.y, dm0.x, sum0);
        sum0 = fma(dm0.y, s, sum0);

        sum1 = fma(acc1.x + acc1.y, dm1.x, sum1);
        sum1 = fma(dm1.y, s, sum1);

        sum2 = fma(acc2.x + acc2.y, dm2.x, sum2);
        sum2 = fma(dm2.y, s, sum2);

        i += chunk;
    }

    const float2 s01 = (float2) ((float)sum0, (float)sum1);
    const float2 r01 = sub_group_reduce_add(s01, false, false);
    const half   r2  = sub_group_reduce_add(sum2);

    if (lane_id == 0) {
        if (row0 < ne01) { dst_ptr[row0] = (half) r01.s0; }
        if (row1 < ne01) { dst_ptr[row1] = (half) r01.s1; }
        if (row2 < ne01) { dst_ptr[row2] = r2; }
    }
}
