kernel void set_rows_f32_i64_f32(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global char * dst,  ulong offsetd,
        int ne00, int ne01,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12, int ne13,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne0, int ne1, int ne2, int ne3,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int nblk0
) {
    src0 += offset0;
    src1 += offset1;
    dst  += offsetd;

    const int i03 = get_group_id(2);
    const int i02 = get_group_id(1);
    const int i01 = get_group_id(0);

    const int i13 = i03 % ne13;
    const int i12 = i02 % ne12;
    const int i11 = i01 % ne11;

    const int lsz0 = get_local_size(0);
    const int i00  = get_local_id(0);

    global float * src0_ptr = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global long  * src1_ptr = (global long  *)(src1 + i01*nb10 + i12*nb11 + i13*nb12);

    const long dst_row = src1_ptr[0];

    global float * dst_ptr = (global float *)(dst + dst_row*nb1 + i02*nb2 + i03*nb3);

    for (int i = i00; i < nblk0; i += lsz0) {
        dst_ptr[i] = src0_ptr[i];
    }
}

kernel void set_rows_f16_i64_f16(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global char * dst,  ulong offsetd,
        int ne00, int ne01,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12, int ne13,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne0, int ne1, int ne2, int ne3,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int nblk0
) {
    src0 += offset0;
    src1 += offset1;
    dst  += offsetd;

    const int i03 = get_group_id(2);
    const int i02 = get_group_id(1);
    const int i01 = get_group_id(0);

    const int i13 = i03 % ne13;
    const int i12 = i02 % ne12;
    const int i11 = i01 % ne11;

    const int lsz0 = get_local_size(0);
    const int i00  = get_local_id(0);

    global half* src0_ptr = (global half *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global long  * src1_ptr = (global long  *)(src1 + i01*nb10 + i12*nb11 + i13*nb12);

    const long dst_row = src1_ptr[0];

    global half * dst_ptr = (global half *)(dst + dst_row*nb1 + i02*nb2 + i03*nb3);

    for (int i = i00; i < nblk0; i += lsz0) {
        dst_ptr[i] = src0_ptr[i];
    }
}
