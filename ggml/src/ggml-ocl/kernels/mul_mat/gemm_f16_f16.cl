#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// tile is BM x BN = NTH * TM * TN, each lane owns TM x TN outputs
#define BM 64
#define BN 64
#define BK 16

#define NTH 256
#define TM 4
#define TN 4

// padding keeps float4 loads aligned and spreads the transposed LDS writes
#define LDA (BN + 4)
#define LDB (BM + 4)

// dst[neuron][token] = sum_k src0[k][neuron] * src1[k][token]
// LDS holds float data as [k][dim], so the inner loop reads float4 and uses vector FMA.
kernel void gemm_f16_f16(global char * src0,
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
    local float * lA = (local float *) lA4;  // lA[kk * LDA + col]
    local float * lB = (local float *) lB4;  // lB[kk * LDB + row]

    float4 acc[TM];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        acc[r] = (float4) (0.0f);
    }

    for (int k = 0; k < ne00; k += BK) {
        // src0 [K, N] into lA[kk][col], converted once
        for (int idx = lid; idx < BK * BN; idx += NTH) {
            const int col  = idx / BK;
            const int kk   = idx % BK;
            const int gcol = group_col * BN + col;
            const int gk   = k + kk;
            float v = 0.0f;
            if (gcol < ne01 && gk < ne00) {
                v = (float) *(const global half *) (src0 + gk * nb00 + gcol * nb01 + i01 * nb02 + i02 * nb03);
            }
            lA[kk * LDA + col] = v;
        }

        // src1 [K, M] into lB[kk][row]
        for (int idx = lid; idx < BK * BM; idx += NTH) {
            const int row  = idx / BK;
            const int kk   = idx % BK;
            const int grow = group_row * BM + row;
            const int gk   = k + kk;
            float v = 0.0f;
            if (grow < ne11 && gk < ne10) {
                v = (float) *(const global half *) (src1 + gk * nb10 + grow * nb11 + i11 * nb12 + i12 * nb13);
            }
            lB[kk * LDB + row] = v;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 2 LDS float4 loads + 4 vector FMA for 16 MACs, 4 independent accumulator chains
        // unroll 2 only: full unroll keeps too many values live and spills
#pragma unroll 2
        for (int kk = 0; kk < BK; ++kk) {
            const float4 ra = *(const local float4 *) (&lA[kk * LDA + local_col_st]);
            const float4 rb = *(const local float4 *) (&lB[kk * LDB + local_row_st]);
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                acc[r] = fma((float4) rb[r], ra, acc[r]);
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
        for (int c = 0; c < TN; ++c) {
            const int gcol = group_col * BN + local_col_st + c;
            if (gcol >= ne0) {
                continue;
            }
            *(global half *) (dst + gcol * nb0 + grow * nb1 + i1 * nb2 + i2 * nb3) = (half) acc[r][c];
        }
    }
}
