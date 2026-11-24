#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK4_1     32
#define QS4_1     16
#define BLK_BYTES 20

// rows per subgroup; the host side must use the same value (ROWS_PER_SUBGROUP).
// small values give more workgroups, which matters when N is only a few thousand.
#define ROWS      3

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
    const int g_row = (i0 * get_num_sub_groups() + warp_id) * ROWS;
    if (g_row >= ne01) {
        return;
    }

    // per-row state kept as int offsets: 64-bit pointers cost too many registers
    int row_off[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        row_off[r] = min(g_row + r, ne01 - 1) * nb01;
    }

    const global half * src1_ptr = (const global half *) (src1 + i11 * nb12 + i12 * nb13);
    global half *       dst_ptr  = (global half *) (dst + i1 * nb2 + i2 * nb3);
    const global uchar * src0_base = (const global uchar *) src0 + src0_off;

    const int qs_total    = blks_per_row * QS4_1;
    const int qs_per_lane = qs_total >> 6;
    int       i           = lane_id * qs_per_lane;
    const int i_end       = i + qs_per_lane;

    float sum[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sum[r] = 0.0f;
    }

    while (i < i_end) {
        const int blk  = i / QS4_1;
        const int boff = i % QS4_1;

        const global uchar * blk_base = src0_base + blk * BLK_BYTES;
        const global half *  x        = src1_ptr + blk * QK4_1;

        float s = 0.0f;
        half2 acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc[r] = (half2) (0.0, 0.0);
        }

        const int chunk = min(QS4_1 - boff, i_end - i);

        for (int j = 0; j < (chunk >> 2); ++j) {
            uint ww[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                ww[r] = ((const global uint *) (blk_base + row_off[r] + 4 + boff))[j];
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int   k  = (j << 2) + q;
                // one activation pair feeds all rows
                const half2 xs = (half2) (x[boff + k], x[boff + QS4_1 + k]);
                s += (float) xs.x;
                s += (float) xs.y;
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    const uchar p = (uchar) (ww[r] >> (q << 3));
                    acc[r] = fma((half2) ((half) (p & 0x0F), (half) (p >> 4)), xs, acc[r]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const half2 dm = *((const global half2 *) (blk_base + row_off[r]));
            sum[r] = fma((float) (acc[r].x + acc[r].y), (float) dm.x, sum[r]);
            sum[r] = fma((float) dm.y, s, sum[r]);
        }

        i += chunk;
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int   rr  = min(g_row + r, ne01 - 1);
        const float red = sub_group_reduce_add(sum[r]);
        if (lane_id == 0 && rr < ne01) {
            dst_ptr[rr] = (half) red;
        }
    }
}
