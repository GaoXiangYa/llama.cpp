#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// fp16 存储: src0 / dst 在设备上都是 half, src1 索引不受影响
kernel void get_rows_f16_i32_f16(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global char * dst,  ulong offsetd,
        ulong nb01, ulong nb02, ulong nb03,
        ulong nb11, ulong nb12, ulong nb13,
        ulong nb1, ulong nb2, ulong nb3,
        int nblk0
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    const int lsz0 = get_local_size(0);
    const int i00 = get_local_id(0);

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    const int i12 = i3;
    const int i11 = i2;
    const int i10 = i1;

    const int i02 = i11;
    const int i03 = i12;

    global int* src1_ptr = (global int*)(src1 + i11 * nb11 + i12 * nb12);
    const int i01 = src1_ptr[i10];
    global half* src0_ptr = (global half*)(src0 + i01 * nb01 + i02 * nb02 + i03 * nb03);
    global half* dst_ptr = (global half*)(dst + i1 * nb1 + i2 * nb2 + i3 * nb3);

    for (int i = i00; i < nblk0; i += lsz0) {
        dst_ptr[i] = src0_ptr[i];
    }
}
