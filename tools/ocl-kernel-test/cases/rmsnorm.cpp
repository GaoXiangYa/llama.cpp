// ggml-ocl op test: RMS_NORM
// 语义: dst[i] = x[i] / sqrt(mean(x^2) + eps)，按 ne0 维度逐行归一化。

#include "../test-op.h"

struct rms_shape {
    int64_t n0, n1, n2, n3;
    float   eps;
};

static ggml_tensor * build_rmsnorm(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const rms_shape * s = (const rms_shape *) userdata;

    ggml_tensor * a   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n1, s->n2, s->n3);
    ggml_tensor * out = ggml_rms_norm(ctx, a, s->eps);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(rmsnorm) {
    const rms_shape cases[] = {
        // 单元素
        { 1,    1,    1, 1, 1e-5f },
        // 单行 / 多行
        { 128,  1,    1, 1, 1e-5f },
        { 2048, 4,    1, 1, 1e-5f },
        { 4096, 1,    1, 1, 1e-6f },
        // 高维 batch
        { 256,  8,    3, 2, 1e-5f },
        { 64,   128,  2, 1, 1e-5f },
    };

    bool all_ok = true;
    for (const rms_shape & s : cases) {
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_rmsnorm, (void *) &s, seed);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_rmsnorm, (void *) &s, seed);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  rmsnorm[%4lld x %4lld x %lld x %lld, eps=%g]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.n0, (long long) s.n1, (long long) s.n2, (long long) s.n3,
               s.eps, ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}
