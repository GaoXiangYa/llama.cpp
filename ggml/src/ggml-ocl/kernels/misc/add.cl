// elementwise add (placeholder kernel to validate the embed/build pipeline, S1)

kernel void kernel_add(
        global const float * src0, ulong offset0,
        global const float * src1, ulong offset1,
        global       float * dst,  ulong offsetd,
        int ne00,
        int ne10,
        ulong nb01, ulong nb02, ulong nb03,
        ulong nb11, ulong nb12, ulong nb13
) {
    src0 = (global const float *)((global const char *)src0 + offset0);
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global       float *)((global const char *)dst  + offsetd);

    const int i0 = get_global_id(0);
    const int i1 = get_global_id(1);
    const int i2 = get_global_id(2);

    const int i10 = i0 % ne10;

    dst[i0 + i1*nb01/sizeof(float) + i2*nb02/sizeof(float)] =
        src0[i0 + i1*nb01/sizeof(float) + i2*nb02/sizeof(float)] +
        src1[i10 + i1*nb11/sizeof(float) + i2*nb12/sizeof(float)];
}
