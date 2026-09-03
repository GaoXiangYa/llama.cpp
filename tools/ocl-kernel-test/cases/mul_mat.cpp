// ggml-ocl op test: MUL_MAT decode 单列 (gemv) (DESIGN.md 23)
// 语义: dst[n] = sum_k w[k, n] * x[k]  (w: [K, N], x: [K, 1] -> dst: [N, 1])

#include "../test-op.h"

#include <functional>
#include <random>

struct mv_shape {
    int64_t k, n;   // w: [K, N]
};

static ggml_tensor * build_gemv(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mv_shape * s = (const mv_shape *) userdata;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->k, s->n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->k, 1);

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(gemv) {
    const mv_shape cases[] = {
        // Qwen3-1.7B 三类投影: attn (K=2048), FFN gate/up (K=2048, N=11008)
        { 2048, 2048 },
        { 2048, 11008 },
        // down 投影: K=11008, N=2048
        { 11008, 2048 },
        // 输出层 (缩小版, < 1GB 单 tensor 限制): K=2048, N=65536
        { 2048, 65536 },
        // 小形状
        { 128, 512 },
    };

    bool all_ok = true;
    for (const mv_shape & s : cases) {
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemv, (void *) &s, seed);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemv, (void *) &s, seed);

        double     nmse = 0.0;
        const bool ok = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemv[K=%5lld, N=%5lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}
