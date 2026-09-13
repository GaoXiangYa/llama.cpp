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

// ggml-ocl op test: MUL_MAT 多列 prefill + Q4_1 权重 (gemm_q4_1)

static ggml_tensor * build_gemm_q4_1(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mm_shape * s = (const mm_shape *) userdata;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, s->k, s->n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->k, s->m);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(gemm_q4_1) {
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
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemm_q4_1, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemm_q4_1, (void *) &s, seed, fill);
        double nmse = 0.0;
        const bool ok = ocl_op_compare_nmse(gpu, cpu, 1e-4, &nmse);
        all_ok &= ok;
        printf("  gemm_q4_1[K=%5lld, N=%5lld, M=%3lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, (long long) s.m, ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}

// ggml-ocl op test: MUL_MAT F16 权重 x F32 激活
// 重点覆盖 attention KV cache: [K, n_kv, n_head, 1] x [K, n_tokens, n_head, 1]

struct mm_shape_4d {
    int64_t k, n, m, n2, n3;
};

static ggml_tensor * build_gemv_f16_f32(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mv_shape * s = (const mv_shape *) userdata;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, s->k, s->n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->k, 1);

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

static ggml_tensor * build_gemm_f16_f32_4d(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mm_shape_4d * s = (const mm_shape_4d *) userdata;

    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s->k, s->n, s->n2, s->n3);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->k, s->m, s->n2, s->n3);

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

static void make_fill_f16_f32(ocl_op_fill_fn & fill) {
    fill = [](ggml_tensor * t, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = ggml_fp32_to_fp16(dist(rng));
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_F32) {
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    };
}

OCL_OP_TEST(gemv_f16) {
    const mv_shape cases[] = {
        { 128, 256 },
        { 128, 512 },
        { 256, 1024 },
    };

    bool all_ok = true;
    for (const mv_shape & s : cases) {
        ocl_op_fill_fn fill;
        make_fill_f16_f32(fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemv_f16_f32, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemv_f16_f32, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemv_f16_f32[K=%5lld, N=%5lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}

OCL_OP_TEST(gemm_f16) {
    const mm_shape_4d cases[] = {
        // attention KQ: [head_dim, n_kv, n_head, 1] x [head_dim, n_tokens, n_head, 1]
        { 128, 256, 2, 16, 1 },
        { 128, 128, 2, 16, 1 },
        { 128, 256, 4, 16, 1 },
        // attention KQV: [head_dim, n_kv, n_head, 1] x [n_kv, n_tokens, n_head, 1]
        { 256, 128, 2, 16, 1 },
    };

    bool all_ok = true;
    for (const mm_shape_4d & s : cases) {
        ocl_op_fill_fn fill;
        make_fill_f16_f32(fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemm_f16_f32_4d, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemm_f16_f32_4d, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemm_f16_f32[K=%5lld, N=%5lld, M=%lld, n2=%lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, (long long) s.m, (long long) s.n2,
               ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}

// ggml-ocl op test: F16 MUL_MAT + non-contiguous view (模拟 attention 里 permute/KV cache view)
// src0/src1 都是 view，nb01/nb11 比连续 row size 大。

struct mm_shape_view {
    int64_t k, n, m, n2, n3;
};

static ggml_tensor * build_gemm_f16_f32_view_4d(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mm_shape_view * s = (const mm_shape_view *) userdata;

    // 底层 tensor 的 N/M 维度放大 2 倍，通过 view stride 选一半，制造非连续行
    ggml_tensor * w_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s->k, 2*s->n, s->n2, s->n3);
    ggml_tensor * x_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->k, 2*s->m, s->n2, s->n3);

    ggml_tensor * w = ggml_view_4d(ctx, w_base, s->k, s->n, s->n2, s->n3,
                                   w_base->nb[1]*2, w_base->nb[2], w_base->nb[3], 0);
    ggml_tensor * x = ggml_view_4d(ctx, x_base, s->k, s->m, s->n2, s->n3,
                                   x_base->nb[1]*2, x_base->nb[2], x_base->nb[3], 0);

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(gemm_f16_view) {
    const mm_shape_view cases[] = {
        { 128, 256, 2, 16, 1 },
        { 128, 128, 2, 16, 1 },
        { 256, 128, 2, 16, 1 },
    };

    bool all_ok = true;
    for (const mm_shape_view & s : cases) {
        ocl_op_fill_fn fill;
        make_fill_f16_f32(fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemm_f16_f32_view_4d, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemm_f16_f32_view_4d, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemm_f16_view[K=%5lld, N=%5lld, M=%lld, n2=%lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, (long long) s.m, (long long) s.n2,
               ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}

// ggml-ocl op test: F16 MUL_MAT + view 非零 offset (模拟 KV cache 位置偏移)
struct mm_shape_off {
    int64_t k, n, m, n2, n3;
    int64_t pad;
};

static ggml_tensor * build_gemm_f16_f32_offset_view_4d(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mm_shape_off * s = (const mm_shape_off *) userdata;

    ggml_tensor * w_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s->k, s->n + s->pad, s->n2, s->n3);
    ggml_tensor * x_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->k, s->m + s->pad, s->n2, s->n3);

    ggml_tensor * w = ggml_view_4d(ctx, w_base, s->k, s->n, s->n2, s->n3,
                                   w_base->nb[1], w_base->nb[2], w_base->nb[3], s->pad * w_base->nb[1]);
    ggml_tensor * x = ggml_view_4d(ctx, x_base, s->k, s->m, s->n2, s->n3,
                                   x_base->nb[1], x_base->nb[2], x_base->nb[3], s->pad * x_base->nb[1]);

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(gemm_f16_offset_view) {
    const mm_shape_off cases[] = {
        { 128, 256, 2, 16, 1, 32 },
        { 128, 128, 2, 16, 1, 16 },
        { 256, 128, 2, 16, 1, 64 },
    };

    bool all_ok = true;
    for (const mm_shape_off & s : cases) {
        ocl_op_fill_fn fill;
        make_fill_f16_f32(fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemm_f16_f32_offset_view_4d, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemm_f16_f32_offset_view_4d, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemm_f16_offset_view[K=%5lld, N=%5lld, M=%lld, n2=%lld, pad=%lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, (long long) s.m, (long long) s.n2, (long long) s.pad,
               ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}

// ggml-ocl op test: F16 MUL_MAT with GQA/MQA broadcast
// src0: [K, N, n_head_kv, 1], src1: [K, M, n_head, 1], n_head % n_head_kv == 0
struct mm_shape_gqa {
    int64_t k, n, m, n_head_kv, n_head, n3;
};

static ggml_tensor * build_gemm_f16_f32_gqa_4d(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const mm_shape_gqa * s = (const mm_shape_gqa *) userdata;

    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s->k, s->n, s->n_head_kv, s->n3);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->k, s->m, s->n_head,    s->n3);

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(gemm_f16_gqa) {
    const mm_shape_gqa cases[] = {
        // Qwen3 常见 GQA: n_head_kv=8, n_head=16
        { 128, 256, 2, 8, 16, 1 },
        { 128, 128, 2, 8, 16, 1 },
        { 256, 128, 2, 8, 16, 1 },
    };

    bool all_ok = true;
    for (const mm_shape_gqa & s : cases) {
        ocl_op_fill_fn fill;
        make_fill_f16_f32(fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_gemm_f16_f32_gqa_4d, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_gemm_f16_f32_gqa_4d, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  gemm_f16_gqa[K=%5lld, N=%5lld, M=%lld, kv=%lld, q=%lld]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.k, (long long) s.n, (long long) s.m,
               (long long) s.n_head_kv, (long long) s.n_head,
               ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}
