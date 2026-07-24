#define GEMM_A(i, j) A[(i) * lda + (j)]
#define GEMM_B(i, j) B[(i) * ldb + (j)]
#define GEMM_C(i, j) C[(i) * ldc + (j)]

#define BM 64
#define BN 64
#define BK 16
#define BM_PAD (BM + 1)
#define BN_PAD (BN + 1)
#define BK_PAD (BK + 1)

#define MICRO_SIZE 4

#define sa(i, j) sa[(i) * BM_PAD + j]
#define sb(i, j) sb[(i) * (BN_PAD) + j]
#define sum(i, j) sum[i * 4 + j]

#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;

#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;

kernel void kernel_arise_mul_mm_f32_f32_l4_lm(
    global char * src0,
    ulong offset0,
    int ne00,   // K
    int ne01,   // M
    int ne02,
    int nb00,
    int nb01,
    int nb02,
    int nb03
    global char * src1,
    ulong offset1,
    int ne11,   // N
    int ne12,
    int ne13,
    int nb10,
    int nb11,
    int nb12,
    int nb13,
    global char * dst,
    ulong offsetd,

    int r2, // head group size
    int r3, // batch group size
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    const int batch_idx = get_group_id(2);
    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;

    const int i03 = i13 / r3;
    const int i02 = i12 / r2;

    global float* A = (global float*)(src0 + i02 * nb02 + i03 * nb03);
    global float* B = (global float*)(src1 + ne12 * nb12 + ne13 * nb13);
    global float* C = (global float*)(dst + i02 * nb02 + i03 * nb03);

    const int lda = ne00, ldb = ne00, ldc = ne11;

    const int gp_x = get_group_id(0);
    const int gp_y = get_group_id(1);

    // 64x64 tile to divide matrix C
    C = &GEMM_C((gp_y << 6), (gp_x << 6));
    A = &GEMM_A((gp_y << 6), 0);
    B = &GEMM_B(0, (gp_x << 6));

    const int lid = get_local_id(0);
    const int warp_id = lid >> 6;
    const int lane_id = lid & 63;

    // local size 256, warp size 64
    const int warp_row = warp_id & 3;
    const int warp_col = warp_id >> 2;

    // thread in warp layout
    //      col0 col1 col2 ... col15
    // row0
    // row1
    // row2
    // row3
    const int lane_row = lane_id & 3;
    const int lane_col = lane_id >> 2;
    // each thread calculate 4x4 sub matrix, global C index
    const int c_row = (warp_row << 4) + (lane_row << 2);
    const int c_col = (lane_col << 2);
    const int a_row = c_row;
    const int b_col = c_col;

    __local float sa[BK * BM_PAD];
    __local float sb[BK * BN_PAD];

    float4 vec_a;
    float4 vec_b;
    float4 reg_c[MICRO_SIZE];
    reg_c[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    reg_c[1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    reg_c[2] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    reg_c[3] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    const int b_k = lid >> 4;
    const int b_col_ld = (lid & 15) << 2;

    for (int k = 0; k < K; k += BK) {
    const int a_col = k + lane_col;
    vec_a.x = GEMM_A(a_row, a_col);
    vec_a.y = GEMM_A(a_row + 1, a_col);
    vec_a.z = GEMM_A(a_row + 2, a_col);
    vec_a.w = GEMM_A(a_row + 3, a_col);
    *(float4*)&sa(lane_col, a_row) = vec_a;

    vload(vec_b, &GEMM_B(k + b_k, b_col_ld));
    vstore(&sb(b_k, b_col_ld), vec_b);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int ik = 0; ik < BK; ++ ik) {
        vload(vec_b, &sb(ik, b_col));
        vload(vec_a, &sa(ik, a_row));

        vscal(reg_c[0], vec_a, vec_b.x);
        vscal(reg_c[1], vec_a, vec_b.y);
        vscal(reg_c[2], vec_a, vec_b.z);
        vscal(reg_c[3], vec_a, vec_b.w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    }
    float4 vec_c;
    vload(vec_c, &GEMM_C(c_row, c_col));
    vec_c.x = reg_c[0].x * alpha + vec_c.x * beta;
    vec_c.y = reg_c[1].x * alpha + vec_c.y * beta;
    vec_c.z = reg_c[2].x * alpha + vec_c.z * beta;
    vec_c.w = reg_c[3].x * alpha + vec_c.w * beta;
    vstore(&GEMM_C(c_row, c_col), vec_c);

    vload(vec_c, &GEMM_C(c_row + 1, c_col));
    vec_c.x = reg_c[0].y * alpha + vec_c.x * beta;
    vec_c.y = reg_c[1].y * alpha + vec_c.y * beta;
    vec_c.z = reg_c[2].y * alpha + vec_c.z * beta;
    vec_c.w = reg_c[3].y * alpha + vec_c.w * beta;
    vstore(&GEMM_C(c_row + 1, c_col), vec_c);

    vload(vec_c, &GEMM_C(c_row + 2, c_col));
    vec_c.x = reg_c[0].z * alpha + vec_c.x * beta;
    vec_c.y = reg_c[1].z * alpha + vec_c.y * beta;
    vec_c.z = reg_c[2].z * alpha + vec_c.z * beta;
    vec_c.w = reg_c[3].z * alpha + vec_c.w * beta;
    vstore(&GEMM_C(c_row + 2, c_col), vec_c);

    vload(vec_c, &GEMM_C(c_row + 3, c_col));
    vec_c.x = reg_c[0].w * alpha + vec_c.x * beta;
    vec_c.y = reg_c[1].w * alpha + vec_c.y * beta;
    vec_c.z = reg_c[2].w * alpha + vec_c.z * beta;
    vec_c.w = reg_c[3].w * alpha + vec_c.w * beta;
    vstore(&GEMM_C(c_row + 3, c_col), vec_c);
}