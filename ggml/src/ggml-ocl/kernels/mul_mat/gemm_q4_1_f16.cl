#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// tile is BM x BN = NTH * TM * TN, each lane owns TM x TN outputs
#define BM 64
#define BN 256
#define BK 16

#define NTH 256
#define TM 8
#define TN 8

// padding keeps float4 loads aligned and spreads the transposed LDS writes
#define LDA (BN + 4)
#define LDB (BM + 4)

#if (BN * BM != NTH * TM * TN)
#    error "tile must be NTH * TM * TN"
#endif
#if (BN != NTH)
#    error "dequant maps one thread per Q4_1 block, so BN must equal NTH"
#endif

// dst[neuron][token] = sum_k dequant(src0)[k][neuron] * src1[k][token]
// src0 is Q4_1: weights are dequantized to float once per k block, so the inner
// loop is plain float math with float4 loads and vector FMA.
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
    dst  = dst  + offsetd;

    const int i0 = get_group_id(0);
    const int i1 = get_group_id(1);
    const int i2 = get_group_id(2);

    const int i11 = i1;
    const int i12 = i2;

    const int i01 = i1 / (ne12 / ne02);
    const int i02 = i2 / (ne13 / ne03);

    const int lid = get_local_id(0);

    const int group_row = i0 / num_groups;  // token
    const int group_col = i0 % num_groups;  // neuron

    const int local_row_st = (lid / (BN / TN)) * TM;
    const int local_col_st = (lid % (BN / TN)) * TN;

    // declared as float4 so the base address is 16-byte aligned
    local float4 lA4[BK * LDA / 4];
    local float4 lB4[BK * LDB / 4];
    local float * lAf = (local float *) lA4;  // lAf[kk * LDA + col]
    local float * lBf = (local float *) lB4;  // lBf[kk * LDB + row]

    // one thread dequantizes one whole Q4_1 block (BN == NTH)
    const int acol = lid;  // 0..BN-1

    // TN/4 float4 per row: TN columns split into 4-wide halves
    float4 acc[TM][TN / 4];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int h = 0; h < TN / 4; ++h) {
            acc[r][h] = (float4) (0.0f);
        }
    }

    for (int k = 0; k < ne00; k += BK) {
        // weights: dequantize half of a Q4_1 block (d, m, 16 quant bytes) into BK floats
        {
            const int gcol = group_col * BN + acol;
            if (gcol < ne01) {
                const global uchar * blk = src0 + gcol * nb01 + i01 * nb02 + i02 * nb03 + (k >> 5) * nb00;
                const float2 dm = vload_half2(0, (const global half *) blk);
                const float dd = dm.x;
                const float mm = dm.y;

                // BK=16 covers half of a Q4_1 block: k/16 selects the nibble
                const int is_high = (k >> 4) & 1;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint w = *(const global uint *) (blk + 4 + j * 4);  // 4 bytes = 8 values
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uchar b    = (uchar) ((w >> (8 * t)) & 0xFF);
                        const int   kk   = j * 4 + t;  // 0..15
                        const int   nib  = is_high ? (b >> 4) : (b & 0x0F);
                        lAf[kk * LDA + acol] = fma(dd, (float) nib, mm);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < BK; ++j) {
                    lAf[j * LDA + acol] = 0.0f;
                }
            }
        }

        // activations: half -> float
        for (int idx = lid; idx < BK * BM; idx += NTH) {
            const int kk   = idx % BK;
            const int row  = idx / BK;
            const int grow = group_row * BM + row;
            const int gk   = k + kk;
            float v = 0.0f;
            if (grow < ne11 && gk < ne10) {
                v = (float) *(const global half *) (src1 + gk * nb10 + grow * nb11 + i11 * nb12 + i12 * nb13);
            }
            lBf[kk * LDB + row] = v;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 3 LDS float4 loads + 8 vector FMA for 64 MACs, 16 independent chains
        // no unroll: 16 accumulators already fill the register budget
#pragma unroll 1
        for (int kk = 0; kk < BK; ++kk) {
            const float4 ra0 = *(const local float4 *) (&lAf[kk * LDA + local_col_st + 0]);
            const float4 ra1 = *(const local float4 *) (&lAf[kk * LDA + local_col_st + 4]);
#pragma unroll
            for (int r = 0; r < TM; r += 4) {
                const float4 rb = *(const local float4 *) (&lBf[kk * LDB + local_row_st + r]);
                acc[r + 0][0] = fma((float4) rb.x, ra0, acc[r + 0][0]);
                acc[r + 1][0] = fma((float4) rb.y, ra0, acc[r + 1][0]);
                acc[r + 2][0] = fma((float4) rb.z, ra0, acc[r + 2][0]);
                acc[r + 3][0] = fma((float4) rb.w, ra0, acc[r + 3][0]);
                acc[r + 0][1] = fma((float4) rb.x, ra1, acc[r + 0][1]);
                acc[r + 1][1] = fma((float4) rb.y, ra1, acc[r + 1][1]);
                acc[r + 2][1] = fma((float4) rb.z, ra1, acc[r + 2][1]);
                acc[r + 3][1] = fma((float4) rb.w, ra1, acc[r + 3][1]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int grow = group_row * BM + local_row_st + r;
        if (grow >= ne1) {
            continue;
        }
#pragma unroll
        for (int h = 0; h < TN / 4; ++h) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int gcol = group_col * BN + local_col_st + h * 4 + t;
                if (gcol >= ne0) {
                    continue;
                }
                *(global half *) (dst + gcol * nb0 + grow * nb1 + i1 * nb2 + i2 * nb3) = (half) acc[r][h][t];
            }
        }
    }
}
