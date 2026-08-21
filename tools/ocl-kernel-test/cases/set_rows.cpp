// ggml-ocl op test: SET_ROWS (DESIGN.md 23)
// 语义: dst[ids[i]] = rows[i] (逐列), ids 索引 dst 的第 1 维 (ne1)
// 覆盖: 单行/多行/多 batch/大量 ids/高维 batch

#include "../test-op.h"

#include <functional>
#include <random>

struct sr_shape {
    int64_t n0, n1, n2, n3;   // dst 形状
    int     n_ids;            // 写入行数
};

static ggml_tensor * build_set_rows(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const sr_shape * s = (const sr_shape *) userdata;

    ggml_tensor * dst  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n1, s->n2, s->n3);
    ggml_tensor * rows = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s->n0, s->n_ids, s->n2, s->n3);
    ggml_tensor * ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, s->n_ids);

    ggml_tensor * out = ggml_set_rows(ctx, dst, rows, ids);
    ggml_build_forward_expand(graph, out);
    return out;
}

// 自定义填充: F32 随机 [-1,1]; I64 ids 填 [0, n1) 的无重复随机排列前 n_ids 个
// (真实使用场景 KV 写入 ids 不重复; 随机重复索引会引发 GPU 并行写同一行的 race)
static void make_fill(const sr_shape & s, ocl_op_fill_fn & fill) {
    fill = [&s](ggml_tensor * t, uint32_t seed) {
        if (t->type == GGML_TYPE_I64) {
            std::mt19937 rng(seed);
            std::vector<int64_t> pool(s.n1);
            for (int64_t i = 0; i < s.n1; i++) {
                pool[i] = i;
            }
            std::shuffle(pool.begin(), pool.end(), rng);
            std::vector<int64_t> idx(ggml_nelements(t));
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

OCL_OP_TEST(set_rows) {
    const sr_shape cases[] = {
        // KV 写入场景: [head_dim, n_ctx, n_kv_head, 1], 写 1 / 4 个位置
        { 128, 4096, 8, 1, 1 },
        { 128, 4096, 8, 1, 4 },
        // 多行写入
        { 1024, 512, 1, 1, 16 },
        // 高维 batch
        { 64, 256, 3, 2, 5 },
    };

    bool all_ok = true;
    for (const sr_shape & s : cases) {
        ocl_op_fill_fn fill;
        make_fill(s, fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_set_rows, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_set_rows, (void *) &s, seed, fill);

        double     max_rel_err = 0.0;
        const bool ok          = ocl_op_compare(gpu, cpu, &max_rel_err);
        all_ok &= ok;
        printf("  set_rows[%4lld x %4lld x %lld x %lld, ids=%d]: %s (max_rel_err=%.2e, n=%zu)\n",
               (long long) s.n0, (long long) s.n1, (long long) s.n2, (long long) s.n3,
               s.n_ids, ok ? "PASS" : "FAIL", max_rel_err, gpu.size());
    }
    return all_ok;
}
