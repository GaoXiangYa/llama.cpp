#define BM 16
#define BN 16
#define BK 16

kernel void gemm_f16_f32(global char * src0,
                         ulong         offset0,
                         global char * src1,
                         ulong         offset1,
                         global char * dst,
                         ulong         offsetd,
                         int           ne00,
                         int           ne01,
                         int           ne02,
                         int           ne03,
                         int           nb00,
                         int           nb01,
                         int           nb02,
                         int           nb03,
                         int           ne10,
                         int           ne11,
                         int           ne12,
                         int           ne13,
                         int           nb10,
                         int           nb11,
                         int           nb12,
                         int           nb13,
                         int           ne0,
                         int           ne1,
                         int           ne2,
                         int           ne3,
                         int           nb0,
                         int           nb1,
                         int           nb2,
                         int           nb3,
                         int           num_groups) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst + offsetd;

    const int i0 = get_group_id(0);
    const int i1 = get_group_id(1);
    const int i2 = get_group_id(2);

    const int i11 = i1;
    const int i12 = i2;

    const int i01 = i1 % ne02;
    const int i02 = i2 % ne03;

    const int lid      = get_local_id(0);
    const int warp_id  = lid >> 6;  // 0..3
    const int lane_id  = lid & 63;  // 0..63
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int lane_row = lane_id >> 3;
    const int lane_col = lane_id & 7;

    local half  lA[BK * BN];
    local float lB[BK * BM];

    const int group_row  = i0 / num_groups;
    const int group_col  = i0 % num_groups;
    const int local_row  = (warp_row << 3) + lane_row;
    const int local_col  = (warp_col << 3) + lane_col;
    const int global_row = group_row * BM + local_row;
    const int global_col = group_col * BN + local_col;

    float sum = 0.0f;

    for (int k = 0; k < ne00; k += BK) {
        // load src0 to local memory
        const int global_a_row = global_col;
        const int global_a_col = k + local_row;

        if (global_a_row < ne01 && global_a_col < ne00) {
            lA[local_col * BK + local_row] =
                *(global half *) (src0 + global_a_col * nb00 + global_a_row * nb01 + i01 * nb02 + i02 * nb03);
        } else {
            lA[local_col * BK + local_row] = 0.0f;
        }

        // load src1 to local memory
        const int global_b_row = global_row;
        const int global_b_col = k + local_col;
        if (global_b_row < ne11 && global_b_col < ne10) {
            lB[local_row * BN + local_col] =
                *(global float *) (src1 + global_b_col * nb10 + global_b_row * nb11 + i11 * nb12 + i12 * nb13);
        } else {
            lB[local_row * BN + local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int kk = 0; kk < BK; ++kk) {
            sum += (float) lA[local_col * BK + kk] * lB[local_row * BK + kk];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_col < ne0 && global_row < ne1) {
        *(global float *) (dst + global_col * nb0 + global_row * nb1 + i1 * nb2 + i2 * nb3) = sum;
    }
}
