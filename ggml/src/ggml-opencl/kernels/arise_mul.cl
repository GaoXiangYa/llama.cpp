kernel void kernel_arise_mul_f32(
    // === src0 (A): shape [n0, n1, n2, n3] ===
    global char * A,
    ulong         offset_A,
    int           n0_A,              // ne00
    int           n1_A,              // ne01
    int           n2_A,              // ne02
    int           n3_A,              // ne03
    int           stride_A_d0,       // nb00
    int           stride_A_d1,       // nb01
    int           stride_A_d2,       // nb02
    int           stride_A_d3,       // nb03

    // === src1 (B): shape [n0_B, n1_B, n2_B, n3_B] ===
    global char * B,
    ulong         offset_B,
    int           n0_B,              // ne10
    int           n1_B,              // ne11
    int           n2_B,              // ne12
    int           n3_B,              // ne13
    int           stride_B_d0,       // nb10
    int           stride_B_d1,       // nb11
    int           stride_B_d2,       // nb12
    int           stride_B_d3,       // nb13 / sizeof(float)

    // === dst (C = A + B): shape [n0_C, n1_C, n2_C, n3_C] ===
    // C，A has the same shape
    global char * C,
    ulong         offset_C,
    int           n0_C,              // ne0
    int           n1_C,              // ne1
    int           n2_C,              // ne2
    int           n3_C,              // ne3
    int           stride_C_d0,       // nb0
    int           stride_C_d1,       // nb1
    int           stride_C_d2,       // nb2
    int           stride_C_d3        // nb3
) {
    A = A + offset_A;
    B = B + offset_B;
    C = C + offset_C;

    const int group0 = get_group_id(0);
    const int group1 = get_group_id(1);
    const int group2 = get_group_id(2);

    int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    A = A + group0 * stride_A_d1 + group1 * stride_A_d2 + group2 * stride_A_d3;
    B = B + (group0 % n1_B) * stride_B_d1 + (group1 % n2_B) * stride_B_d2 + (group2 % n3_B) * stride_B_d3;
    C = C + group0 * stride_C_d1 + group1 * stride_C_d2 + group2 * stride_C_d3;

    for (int idx = lid; idx < n0_C; idx += local_size) {
        *((global float*)(C + idx * stride_C_d0)) = *((global float*)(A + idx * stride_A_d0)) * *((global float*)(B + idx % n0_B * stride_B_d0));
    }
}