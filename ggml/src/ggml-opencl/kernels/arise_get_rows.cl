kernel void kernel_arise_get_rows_f32(
    global char* src0, ulong offset0,
    global char* src1, ulong offset1,
    global char* dst,  ulong offsetd,
    int ne00,
    ulong nb01, ulong nb01, ulong nb02, ulong nb03,
    int ne10,
    ulong nb10, ulong nb11, ulong nb12, ulong nb13,
    ulong nb1, ulong nb2, ulong nb3) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    const int g01 = get_group_id(0);
    const int g02 = get_group_id(1);
    const int g03 = get_global_id(3);

    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    int idx = (global int*)(src1 + g01 * nb10 + g02 * nb11 + g03 * nb13)[0];

    global float* pdst = (global float*)(dst + g01 * nb1 + g02 * nb2 + g03 * nb3);

    for (int i = lid; i < ne00; i += lsz) {
        pdst[i] = (global float*)(src0 + idx * nb01 + g02 * nb02 + g03 * nb03)[i];
    }
}