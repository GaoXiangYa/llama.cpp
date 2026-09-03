// ggml-ocl op test: GET_ROWS (DESIGN.md 23)
// 语义: dst[:, i] = src0[:, src1[i]] (按行查表, embedding 场景)

#include "../test-op.h"

#include <functional>
#include <random>

struct gr_shape {
    int64_t w[4];      // src0 权重矩阵形状
    int64_t ids[4];    // src1 索引形状 (I32)
};

static ggml_tensor * build_get_rows(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const gr_shape * s = (const gr_shape *) userdata;

    ggml_tensor * w   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->w[0], s->w[1], s->w[2], s->w[3]);
    ggml_tensor * ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, s->ids[0], s->ids[1], s->ids[2], s->ids[3]);

    ggml_tensor * out = ggml_get_rows(ctx, w, ids);
    ggml_build_forward_expand(graph, out);
    return out;
}

// 自定义填充: F32 随机 [-1,1]; I32 ids 填 [0, w_n1) 的无重复随机排列前 n 个
static void make_fill(const gr_shape & s, ocl_op_fill_fn & fill) {
    fill = [&s](ggml_tensor * t, uint32_t seed) {
        if (t->type == GGML_TYPE_I32) {
            std::mt19937 rng(seed);
            std::vector<int32_t> pool(s.w[1]);
            for (int64_t i = 0; i < s.w[1]; i++) {
                pool[i] = (int32_t) i;
            }
            std::shuffle(pool.begin(), pool.end(), rng);
            std::vector<int32_t> idx(ggml_nelements(t));
            for (size_t i = 0; i < idx.size(); i++) {
                idx[i] = pool[i];
            }
            ggml_backend_tensor_set(t, idx.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_F32) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    };
}

OCL_OP_TEST(get_rows) {
    const gr_shape cases[] = {
        // 单索引查表
        { { 128, 256, 1, 1 }, { 1, 1, 1, 1 } },
        // 多索引 (FFN/attn 权重查表)
        { { 1024, 512, 1, 1 }, { 16, 1, 1, 1 } },
        // embedding 场景: [n_embd, n_vocab] x [n_tokens]
        { { 2048, 65536, 1, 1 }, { 8, 1, 1, 1 } },
        // 多 batch 索引
        { { 64, 128, 3, 1 }, { 5, 3, 1, 1 } },
    };

    bool all_ok = true;
    for (const gr_shape & s : cases) {
        ocl_op_fill_fn fill;
        make_fill(s, fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_get_rows, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_get_rows, (void *) &s, seed, fill);

        double     max_rel_err = 0.0;
        const bool ok          = ocl_op_compare(gpu, cpu, &max_rel_err);
        all_ok &= ok;
        printf("  get_rows[%4lld x %4lld x %lld x %lld, ids=%lld]: %s (max_rel_err=%.2e, n=%zu)\n",
               (long long) s.w[0], (long long) s.w[1], (long long) s.w[2], (long long) s.w[3],
               (long long) s.ids[0], ok ? "PASS" : "FAIL", max_rel_err, gpu.size());
    }
    return all_ok;
}
