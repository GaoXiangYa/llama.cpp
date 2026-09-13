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

// ggml-ocl op test: SOFTMAX with mask
// 重点覆盖 mask 中含 -INFINITY 的情况。
// 旧 kernel 在 max_bias == 0 时 slope=0.0，会算出 0 * (-INF) = NaN。

struct sm_mask_shape {
    int64_t n0, n1, n2, n3;
    float   scale;
};

static ggml_tensor * build_softmax_mask(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const sm_mask_shape * s = (const sm_mask_shape *) userdata;

    ggml_tensor * a    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n1, s->n2, s->n3);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n1, s->n2, s->n3);
    ggml_set_name(mask, "softmax_mask");

    ggml_tensor * out = ggml_soft_max_ext(ctx, a, mask, s->scale, 0.0f);
    ggml_build_forward_expand(graph, out);
    return out;
}

static void make_mask_fill(const sm_mask_shape & s, ocl_op_fill_fn & fill) {
    fill = [&s](ggml_tensor * t, uint32_t seed) {
        std::mt19937 rng(seed);

        if (ggml_get_name(t) != nullptr && std::string(ggml_get_name(t)) == "softmax_mask") {
            std::vector<float> data(ggml_nelements(t));
            const int64_t n0 = s.n0;
            const int64_t nrows = ggml_nelements(t) / n0;
            for (int64_t r = 0; r < nrows; r++) {
                for (int64_t i = 0; i < n0; i++) {
                    // 每行最后一个位置保持有效，其余位置 -INF，模拟 causal mask
                    data[r*n0 + i] = (i == n0 - 1) ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        } else {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    };
}

OCL_OP_TEST(softmax_mask) {
    const sm_mask_shape cases[] = {
        { 256, 2, 16, 1, 1.0f   },  // 对应失败 shape
        { 128, 4, 1,  1, 1.0f   },
        { 2048, 1, 1, 1, 1.0f   },
        { 64,  8, 3,  2, 0.5f   },
    };

    bool all_ok = true;
    for (const sm_mask_shape & s : cases) {
        ocl_op_fill_fn fill;
        make_mask_fill(s, fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_softmax_mask, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_softmax_mask, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  softmax_mask[%4lld x %4lld x %lld x %lld, scale=%g]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.n0, (long long) s.n1, (long long) s.n2, (long long) s.n3,
               s.scale, ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}
