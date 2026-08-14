// elementwise add (DESIGN.md 19.2)
// 3D grid: 每个 workgroup 处理 src0 的一行 (i01, i02, i03),
// i0 在组内循环覆盖 dst 的 dim0; 广播通过 src1 维度取模实现.
// 全部 byte-index 寻址 (nb 驱动), 天然支持非连续/permuted 布局.

kernel void kernel_add(
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

    const int i13 = i03 % ne13;
    const int i12 = i02 % ne12;
    const int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global float *)(dst_ptr + i0*nb0)) =
            *((global float *)(src0_ptr + i0*nb00)) +
            *((global float *)(src1_ptr + i10*nb10));
    }
}
