#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void cpy_f16_f16(
        global char * src0, ulong offset0,
        global char * dst,  ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int dst_cont
) {
    src0 += offset0;
    dst  += offsetd;

    const int i01 = get_group_id(0);
    const int i02 = get_group_id(1);
    const int i03 = get_group_id(2);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    global const char * src_row = src0 + i01*nb01 + i02*nb02 + i03*nb03;

    if (dst_cont) {
        const long k_row = ((long) i03*ne02*ne01 + (long) i02*ne01 + i01)*ne00;
        global char * dst_row = dst + k_row*sizeof(half);
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global half *) (dst_row + (ulong) i00*sizeof(half)) = *(global const half *) (src_row + (ulong) i00*nb00);
        }
    } else {
        global char * dst_row = dst + i01*nb1 + i02*nb2 + i03*nb3;
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global half *) (dst_row + (ulong) i00*nb0) = *(global const half *) (src_row + (ulong) i00*nb00);
        }
    }
}

kernel void cpy_f32_f32(
        global char * src0, ulong offset0,
        global char * dst,  ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int dst_cont
) {
    src0 += offset0;
    dst  += offsetd;

    const int i01 = get_group_id(0);
    const int i02 = get_group_id(1);
    const int i03 = get_group_id(2);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    global const char * src_row = src0 + i01*nb01 + i02*nb02 + i03*nb03;

    if (dst_cont) {
        const long k_row = ((long) i03*ne02*ne01 + (long) i02*ne01 + i01)*ne00;
        global char * dst_row = dst + k_row*sizeof(float);
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global float *) (dst_row + (ulong) i00*sizeof(float)) = *(global const float *) (src_row + (ulong) i00*nb00);
        }
    } else {
        global char * dst_row = dst + i01*nb1 + i02*nb2 + i03*nb3;
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global float *) (dst_row + (ulong) i00*nb0) = *(global const float *) (src_row + (ulong) i00*nb00);
        }
    }
}

kernel void cpy_f16_f32(
        global char * src0, ulong offset0,
        global char * dst,  ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int dst_cont
) {
    src0 += offset0;
    dst  += offsetd;

    const int i01 = get_group_id(0);
    const int i02 = get_group_id(1);
    const int i03 = get_group_id(2);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    global const char * src_row = src0 + i01*nb01 + i02*nb02 + i03*nb03;

    if (dst_cont) {
        const long k_row = ((long) i03*ne02*ne01 + (long) i02*ne01 + i01)*ne00;
        global char * dst_row = dst + k_row*sizeof(float);
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global float *) (dst_row + (ulong) i00*sizeof(float)) = (float) *(global const half *) (src_row + (ulong) i00*nb00);
        }
    } else {
        global char * dst_row = dst + i01*nb1 + i02*nb2 + i03*nb3;
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global float *) (dst_row + (ulong) i00*nb0) = (float) *(global const half *) (src_row + (ulong) i00*nb00);
        }
    }
}

kernel void cpy_f32_f16(
        global char * src0, ulong offset0,
        global char * dst,  ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int dst_cont
) {
    src0 += offset0;
    dst  += offsetd;

    const int i01 = get_group_id(0);
    const int i02 = get_group_id(1);
    const int i03 = get_group_id(2);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    global const char * src_row = src0 + i01*nb01 + i02*nb02 + i03*nb03;

    if (dst_cont) {
        const long k_row = ((long) i03*ne02*ne01 + (long) i02*ne01 + i01)*ne00;
        global char * dst_row = dst + k_row*sizeof(half);
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global half *) (dst_row + (ulong) i00*sizeof(half)) = (half) *(global const float *) (src_row + (ulong) i00*nb00);
        }
    } else {
        global char * dst_row = dst + i01*nb1 + i02*nb2 + i03*nb3;
        for (int i00 = lid; i00 < ne00; i00 += lsz) {
            *(global half *) (dst_row + (ulong) i00*nb0) = (half) *(global const float *) (src_row + (ulong) i00*nb00);
        }
    }
}
