#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_add_f16(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global char * dst,  ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12, int ne13,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne0, int ne1, int ne2, int ne3,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3
) {
    src0 += offset0;
    src1 += offset1;
    dst  += offsetd;

    const int i03 = get_group_id(2);
    const int i02 = get_group_id(1);
    const int i01 = get_group_id(0);

    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    const int i13 = (ne03 != ne13) ? i03 % ne13 : i03;
    const int i12 = (ne02 != ne12) ? i02 % ne12 : i02;
    const int i11 = (ne01 != ne11) ? i01 % ne11 : i01;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;
    const int nevec = ne0 >> 3;
    for (int iv = lid; iv < nevec; iv += lsz) {
        const int i0  = iv << 3;
        const int i10 = i0 % ne10;
        const half8 a = vload8(0, (global const half *)(src0_ptr + i0 * nb00));
        const half8 b = vload8(0, (global const half *)(src1_ptr + i10 * nb10));
        vstore8(a + b, 0, (global half *)(dst_ptr + i0 * nb0));
    }
    for (int i0 = (nevec << 3) + lid; i0 < ne0; i0 += lsz) {
        const int i10 = i0 % ne10;
        const half a = *(global const half *)(src0_ptr + i0 * nb00);
        const half b = *(global const half *)(src1_ptr + i10 * nb10);
        *(global half *)(dst_ptr + i0 * nb0) = a + b;
    }
}
