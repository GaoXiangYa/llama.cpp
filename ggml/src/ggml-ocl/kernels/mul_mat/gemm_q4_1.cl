#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define BM 4
#define BN 64
#define BK 32
#define BK_HALF 16
#define BLOCK_SIZE 20

kernel void gemm_q4_1(
        global uchar* src0, ulong offset0,
        global char* src1, ulong offset1,
        global char* dst, ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        int nb00, int nb01, int nb02, int nb03,
        int ne10, int ne11, int ne12, int ne13,
        int nb10, int nb11, int nb12, int nb13,
        int ne0, int ne1, int ne2, int ne3,
        int nb0, int nb1, int nb2, int nb3,
        int num_groups
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    const int i0 = get_group_id(0);
    const int i1 = get_group_id(1);
    const int i2 = get_group_id(2);

    const int i11 = i1;
    const int i12 = i2;

    const int i01 = i1 % ne02;
    const int i02 = i2 % ne03;

    const int lid      = get_local_id(0);
    const int warp_id  = lid >> 6;      // 0..3
    const int lane_id  = lid & 63;      // 0..63
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int lane_row = lane_id >> 5;
    const int lane_col = lane_id & 31;

    local uchar lA[BN * BK_HALF];
    local float lB[BM * BK];

    const int group_row = i0 / num_groups;
    const int group_col = i0 % num_groups;
    const int local_row = (warp_row << 1) + lane_row;
    const int local_col = (warp_col << 5) + lane_col;
    const int global_row = group_row * BM + local_row;
    const int global_col = group_col * BN + local_col;

    float d_frag = 0.0f;
    float m_frag = 0.0f;

    float acc = 0.0f;

    for (int k = 0; k < ne00; k += BK) {
        // load src0 to local memory
        const int global_a_row = global_col;

        if (global_a_row < ne01) {
            const int blk_col = k >> 5;
            global uchar* a_block_per_row = src0 + global_a_row * nb01 + i01 * nb02 + i02 * nb03;
            global uchar* blk = a_block_per_row + blk_col * nb00;
            half d = vload_half(0, (const half*) blk);       // scale
            half m = vload_half(1, (const half*) blk);       // min
            d_frag = (float) d;
            m_frag = (float) m;
            if (local_row == 0) {
                global uchar* qs = (global uchar*)(blk + 4);
                for (int i = 0; i < BK_HALF; ++ i) {
                    lA[local_col * BK_HALF + i] = qs[i];
                }
            }
        }

        const int global_b_row = global_row;
        const int global_b_col = k + local_col;
        if (local_col < BK) {
            if (global_b_row < ne11 && global_b_col < ne10) {
                lB[local_row * BK + local_col] = *(global float*)(src1 + global_b_col * nb10 + global_b_row * nb11 + i11 * nb12 + i12 * nb13);
            } else {
                lB[local_row * BK + local_col] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int ik = 0; ik < BK; ++ ik) {
            uchar packed = lA[local_col * BK_HALF + (ik & (BK_HALF - 1))];
            float q = (float)(ik < BK_HALF ? (packed & 0x0F) : (packed >> 4));
            float b = lB[local_row * BK + ik];
            sum0 += q * b;
            sum1 += b;
        }
        acc += (d_frag * sum0 + m_frag * sum1);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_col < ne0 && global_row < ne1) {
        *(global float*)(dst + global_col * nb0 + global_row * nb1 + i1 * nb2 + i2 * nb3) = acc;
    }
}