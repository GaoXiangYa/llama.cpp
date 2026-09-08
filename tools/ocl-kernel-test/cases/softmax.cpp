// ggml-ocl op test: SOFTMAX / SOFT_MAX
// 语义: softmax(a * scale)，沿 ne0（最快维度/每一行）归一化。
// 覆盖: 单元素、单行、多行、长行、多 batch、非 1 scale。

#include "../test-op.h"

struct sm_shape {
    int64_t n0, n1, n2, n3;
    float   scale;
};

static ggml_tensor * build_softmax(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const sm_shape * s = (const sm_shape *) userdata;

    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n1, s->n2, s->n3);
    ggml_tensor * out = ggml_soft_max_ext(ctx, a, nullptr, s->scale, 0.0f);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(softmax) {
    const sm_shape cases[] = {
        // 单元素
        { 1,    1,    1, 1, 1.0f   },
        // 短行，lsz < 64，单 subgroup 路径
        { 8,    1,    1, 1, 1.0f   },
        { 32,   4,    1, 1, 1.0f   },
        // 长行/多行，触发跨 subgroup reduce
        { 128,  1,    1, 1, 1.0f   },
        { 256,  4,    1, 1, 1.0f   },
        { 2048, 1,    1, 1, 1.0f   },
        { 1024, 8,    1, 1, 0.5f   },
        // 高维 batch
        { 64,   8,    3, 2, 1.0f   },
        // 带 scale
        { 128,  3,    2, 1, 0.125f },
    };

    bool all_ok = true;
    for (const sm_shape & s : cases) {
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_softmax, (void *) &s, seed);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_softmax, (void *) &s, seed);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  softmax[%4lld x %4lld x %lld x %lld, scale=%g]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.n0, (long long) s.n1, (long long) s.n2, (long long) s.n3,
               s.scale, ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}
