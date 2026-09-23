#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define BM       16
#define BN       64
#define BK       32
#define TM       4
#define BK_HALF  16
#define A_STRIDE 20
#define B_STRIDE BK

#if (BN != 64)
#    error "gemm_q4_1_f16: 64 个 lane 全铺在 neuron 方向, BN 必须等于 64"
#endif
#if (BM != 4 * TM)
#    error "gemm_q4_1_f16: 行组数 = 4 个 wavefront, BM 必须等于 4 * TM"
#endif

kernel void gemm_q4_1_f16(global uchar * src0,
                          ulong          offset0,
                          global char *  src1,
                          ulong          offset1,
                          global char *  dst,
                          ulong          offsetd,
                          int            ne00,
                          int            ne01,
                          int            ne02,
                          int            ne03,
                          int            nb00,
                          int            nb01,
                          int            nb02,
                          int            nb03,
                          int            ne10,
                          int            ne11,
                          int            ne12,
                          int            ne13,
                          int            nb10,
                          int            nb11,
                          int            nb12,
                          int            nb13,
                          int            ne0,
                          int            ne1,
                          int            ne2,
                          int            ne3,
                          int            nb0,
                          int            nb1,
                          int            nb2,
                          int            nb3,
                          int            num_groups) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst + offsetd;

    const int i0  = get_group_id(0);
    const int i1  = get_group_id(1);
    const int i2  = get_group_id(2);
    const int i11 = i1;
    const int i12 = i2;
    const int i01 = i1 / (ne12 / ne02);
    const int i02 = i2 / (ne13 / ne03);

    const int lid     = get_local_id(0);
    const int nth     = (int) get_local_size(0);
    const int warp_id = lid >> 6;  // 0..3 -> token 行组
    const int lane_id = lid & 63;  // 0..63 -> neuron 列

    const int local_row_st = warp_id * TM;
    const int local_col_st = lane_id;

    const int group_row     = i0 / num_groups;
    const int group_col     = i0 % num_groups;
    const int global_row_st = group_row * BM + local_row_st;
    const int global_col_st = group_col * BN + local_col_st;

    local uchar lA[BN * A_STRIDE];
    local half  lB[BM * B_STRIDE];
    local half  lDM[BN * 2];

    const int seg_per_row = nth / BN;
    const int ld_row      = lid / seg_per_row;
    const int ld_seg      = lid % seg_per_row;

    half   acc0 = 0.0f;
    half   acc1 = 0.0f;
    half   acc2 = 0.0f;
    half   acc3 = 0.0f;
    uchar4 tmp  = (uchar4) (0, 0, 0, 0);

    for (int k = 0; k < ne00; k += BK) {
        {
            const int a_row = group_col * BN + ld_row;
            if (a_row < ne01) {
                const global uchar * blk = src0 + a_row * nb01 + i01 * nb02 + i02 * nb03 + (k >> 5) * nb00;
                if (ld_seg == 0) {
                    lDM[ld_row * 2 + 0] = vload_half(0, (const global half *) blk);
                    lDM[ld_row * 2 + 1] = vload_half(1, (const global half *) blk);
                }
                *(local uint *) (lA + ld_row * A_STRIDE + ld_seg * 4) = *(const global uint *) (blk + 4 + ld_seg * 4);
                // vstore4(vload4(0, blk + 4 + ld_seg * 4), 0, lA + ld_row * A_STRIDE + ld_seg * 4);
            } else {
                if (ld_seg == 0) {
                    lDM[ld_row * 2 + 0] = (half) 0.0f;
                    lDM[ld_row * 2 + 1] = (half) 0.0f;
                }
                vstore4((uchar4) (0, 0, 0, 0), 0, lA + ld_row * A_STRIDE + ld_seg * 4);
            }
        }

        // 激活: BM x BK 个 half, 全 256 线程分摊
        for (int idx = lid; idx < BM * BK; idx += nth) {
            const int row   = idx / BK;
            const int col   = idx % BK;
            const int b_row = group_row * BM + row;
            const int b_col = k + col;
            half      v     = (half) 0.0f;
            if (b_row < ne11 && b_col < ne10) {
                v = *(const global half *) (src1 + b_col * nb10 + b_row * nb11 + i11 * nb12 + i12 * nb13);
            }
            lB[row * B_STRIDE + col] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int  fa = local_col_st * A_STRIDE;
        const int  fb = local_row_st * B_STRIDE;
        const half dd = lDM[local_col_st * 2 + 0];
        const half mm = lDM[local_col_st * 2 + 1];

        for (int ik = 0; ik < BK; ++ik) {
            const uchar packed = lA[fa + (ik & (BK_HALF - 1))];
            const half  q      = dd * (ik < BK_HALF ? (packed & 0x0F) : (packed >> 4)) + mm;

            acc0 += q * lB[fb + 0 * B_STRIDE + ik];
            acc1 += q * lB[fb + 1 * B_STRIDE + ik];
            acc2 += q * lB[fb + 2 * B_STRIDE + ik];
            acc3 += q * lB[fb + 3 * B_STRIDE + ik];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_col_st < ne0) {
        global char * dbase = dst + global_col_st * nb0 + i1 * nb2 + i2 * nb3;
        if (global_row_st + 0 < ne1) {
            *(global half *) (dbase + (global_row_st + 0) * nb1) = acc0;
        }
        if (global_row_st + 1 < ne1) {
            *(global half *) (dbase + (global_row_st + 1) * nb1) = acc1;
        }
        if (global_row_st + 2 < ne1) {
            *(global half *) (dbase + (global_row_st + 2) * nb1) = acc2;
        }
        if (global_row_st + 3 < ne1) {
            *(global half *) (dbase + (global_row_st + 3) * nb1) = acc3;
        }
    }
}
