#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// fp16 存储: 三个操作数在设备上都是 half
kernel void swiglu_f16(global char * src0,
                       ulong         offset0,
                       global char * src1,
                       ulong         offset1,
                       global char * dst,
                       ulong         offsetd,
                       int           ne0,
                       ulong         nb1,
                       ulong         nb2,
                       ulong         nb3) {
    int i0  = get_local_id(0);
    int i1  = get_group_id(0);
    int i2  = get_group_id(1);
    int i3  = get_group_id(2);
    int lsz = get_local_size(0);

    global half * psrc0 = (global half *) (src0 + offset0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    global half * psrc1 = (global half *) (src1 + offset1 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    global half * pdst  = (global half *) (dst + offsetd + i1 * nb1 + i2 * nb2 + i3 * nb3);

    for (int i = i0; i < ne0; i += lsz) {
        float g = convert_float(psrc0[i]);
        float u = convert_float(psrc1[i]);
        pdst[i] = convert_half(g * 1 / (1 + exp(-1 * g)) * u);
    }
}
