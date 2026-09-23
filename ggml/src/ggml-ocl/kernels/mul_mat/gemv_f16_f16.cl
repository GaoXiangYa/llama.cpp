#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void gemv_f16_f16(const global char * src0,
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
                         int                 gqa_kv,
                         int                 gqa_b) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst + offsetd;

    const int i0 = get_group_id(0);
    const int i1 = get_group_id(1);
    const int i2 = get_group_id(2);

    const int i11 = i1;
    const int i12 = i2;

    local int l_src0_off;
    const int lid = get_local_id(0);
    if (lid == 0) {
        l_src0_off = (i1 / gqa_kv) * nb02 + (i2 / gqa_b) * nb03;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int src0_off = l_src0_off;

    const int warp_size = get_sub_group_size();
    const int warp_id   = get_sub_group_id();
    const int lane_id   = get_sub_group_local_id();
    const int row_base  = (i0 * get_num_sub_groups() + warp_id) << 2;
    if (row_base >= ne01) {
        return;
    }
    const int row0 = row_base;
    const int row1 = row_base + 1;
    const int row2 = row_base + 2;
    const int row3 = row_base + 3;

    const global half * a0 = (const global half *) (src0 + row0 * nb01 + src0_off);
    const global half * a1 = (const global half *) (src0 + row1 * nb01 + src0_off);
    const global half * a2 = (const global half *) (src0 + row2 * nb01 + src0_off);
    const global half * a3 = (const global half *) (src0 + row3 * nb01 + src0_off);

    const global half * src1_ptr = (const global half *) (src1 + i11 * nb12 + i12 * nb13);
    global half *       dst_ptr  = (global half *) (dst + i1 * nb2 + i2 * nb3);

    float4 sums = (float4) (0.0, 0.0, 0.0, 0.0);

    for (int i = lane_id; i < ne00; i += warp_size) {
        const float xv = (float) src1_ptr[i];

        sums.s0 = fma((float) a0[i], xv, sums.s0);
        sums.s1 = fma((float) a1[i], xv, sums.s1);
        sums.s2 = fma((float) a2[i], xv, sums.s2);
        sums.s3 = fma((float) a3[i], xv, sums.s3);
    }

    float4 ret = sub_group_reduce_add(sums, false, false);

    if (lane_id == 0) {
        if (row0 < ne01) {
            dst_ptr[row0] = (half) ret.s0;
        }
        if (row1 < ne01) {
            dst_ptr[row1] = (half) ret.s1;
        }
        if (row2 < ne01) {
            dst_ptr[row2] = (half) ret.s2;
        }
        if (row3 < ne01) {
            dst_ptr[row3] = (half) ret.s3;
        }
    }
}
