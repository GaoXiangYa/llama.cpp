// ggml-ocl op test: SWIGLU (split variant)
// 使用 ggml_glu_split(a, b, GGML_GLU_OP_SWIGLU) 构造两个同形状 F32 输入。
// 公式与 CPU 的 swiglu 一致: dst[i] = silu(a[i]) * b[i]。

#include "../test-op.h"

struct glu_shape {
    int64_t n0, n1, n2, n3;
};

static ggml_tensor * build_swiglu(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const glu_shape * s = (const glu_shape *) userdata;

    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n1, s->n2, s->n3);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n1, s->n2, s->n3);
    ggml_tensor * out = ggml_glu_split(ctx, a, b, GGML_GLU_OP_SWIGLU);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(swiglu) {
    const glu_shape cases[] = {
        { 1,    1,    1, 1 },
        { 128,  1,    1, 1 },
        { 2048, 4,    1, 1 },
        { 4096, 1,    1, 1 },
        { 256,  8,    3, 2 },
        { 64,   128,  2, 1 },
    };

    bool all_ok = true;
    for (const glu_shape & s : cases) {
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_swiglu, (void *) &s, seed);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_swiglu, (void *) &s, seed);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  swiglu[%4lld x %4lld x %lld x %lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.n0, (long long) s.n1, (long long) s.n2, (long long) s.n3,
               ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}
