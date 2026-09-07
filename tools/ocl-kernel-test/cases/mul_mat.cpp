#include "../test-op.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include <functional>
#include <random>

struct mv_shape {
    int64_t k, n;  // w: [K, N]
};

static ggml_tensor * build_gemv_q4_1(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mv_shape * s = (const mv_shape *) userdata;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, s->k, s->n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->k, 1);

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

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
        { 2048,  2048  },
        { 2048,  11008 },
        // down 投影: K=11008, N=2048
        { 11008, 2048  },
        // 输出层 (缩小版, < 1GB 单 tensor 限制): K=2048, N=65536
        { 2048,  65536 },
        // 小形状
        { 128,   512   },
    };

    bool all_ok = true;
    for (const mv_shape & s : cases) {
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemv, (void *) &s, seed);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemv, (void *) &s, seed);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemv[K=%5lld, N=%5lld]: %s (nmse=%.2e, n=%zu)\n", (long long) s.k, (long long) s.n,
               ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}

static void make_fill_q4_1(ocl_op_fill_fn & fill) {
    fill = [](ggml_tensor * t, uint32_t seed) {
        std::mt19937 rng(seed);
        if (t->type == GGML_TYPE_Q4_1) {
            const int64_t                         n = ggml_nelements(t);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float>                    tmp(n);
            for (int64_t i = 0; i < n; i++) {
                tmp[i] = dist(rng);
            }
            std::vector<uint8_t> q(ggml_nbytes(t));
            const size_t         row_size = ggml_row_size(t->type, t->ne[0]);
            const int64_t        n_rows   = n / t->ne[0];
            for (int64_t r = 0; r < n_rows; r++) {
                quantize_row_q4_1_ref(tmp.data() + r * t->ne[0], (block_q4_1 *) (q.data() + r * row_size), t->ne[0]);
            }
            ggml_backend_tensor_set(t, q.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_F32) {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float>                    data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    };
}

OCL_OP_TEST(gemv_q4_1) {
    const mv_shape cases[] = {
        { 2048,  2048  },
        { 2048,  11008 },
        { 11008, 2048  },
        { 2048,  65536 }, // 输出层 (缩小版, < 1GB)
        { 128,   512   },
    };

    bool all_ok = true;
    for (const mv_shape & s : cases) {
        ocl_op_fill_fn fill;
        make_fill_q4_1(fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemv_q4_1, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemv_q4_1, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-4, &nmse);
        all_ok &= ok;
        printf("  gemv_q4_1[K=%5lld, N=%5lld]: %s (nmse=%.2e, n=%zu)\n", (long long) s.k, (long long) s.n,
               ok ? "PASS" : "FAIL", nmse, gpu.size());

    }
    return all_ok;
}

// ggml-ocl op test: MUL_MAT 多列 prefill (gemm): 权重 [K,N] x 激活 [K,M] -> dst [N,M]

struct mm_shape {
    int64_t k, n, m;
};

static ggml_tensor * build_gemm(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mm_shape * s = (const mm_shape *) userdata;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->k, s->n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->k, s->m);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(gemm) {
    const mm_shape cases[] = {
        { 2048, 2048, 8 },
        { 2048, 11008, 4 },
        { 11008, 2048, 16 },
        { 128, 512, 32 },
    };
    bool all_ok = true;
    for (const mm_shape & s : cases) {
        ocl_op_fill_fn fill;
        make_fill_q4_1(fill);
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemm, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemm, (void *) &s, seed, fill);
        double nmse = 0.0;
        const bool ok = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemm_f32[K=%5lld, N=%5lld, M=%3lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, (long long) s.m, ok ? "PASS" : "FAIL", nmse, gpu.size());

    }
    return all_ok;
}
