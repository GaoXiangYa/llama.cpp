// ggml-ocl op test: SET_ROWS (DESIGN.md 23)
// 语义: dst[ids[i]] = rows[i] (逐列), ids 索引 dst 的第 1 维 (ne1)
// 覆盖: 单行/多行/多 batch/大量 ids/高维 batch
// 重点覆盖: rows=F32, dst=F16, dst 是 view (KV cache 写入路径)

#include "../test-op.h"

#include <functional>
#include <random>

struct sr_shape {
    int64_t   n0, n1, n2, n3;   // dst 形状
    int       n_ids;            // 写入行数
    ggml_type dst_type;
    ggml_type rows_type;
    bool      view;
};

static ggml_tensor * build_set_rows(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const sr_shape * s = (const sr_shape *) userdata;

    // view 场景模拟 cache_k_l0 (view): 底层 tensor 比实际写入的 n1 略大
    ggml_tensor * base = ggml_new_tensor_4d(ctx, s->dst_type, s->n0, s->n1 + (s->view ? 64 : 0), s->n2, s->n3);

    ggml_tensor * dst = base;
    if (s->view) {
        dst = ggml_view_4d(ctx, base, s->n0, s->n1, s->n2, s->n3,
                           base->nb[1], base->nb[2], base->nb[3], 0);
    }

    ggml_tensor * rows = ggml_new_tensor_4d(ctx, s->rows_type, s->n0, s->n_ids, s->n2, s->n3);
    ggml_tensor * ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, s->n_ids);

    ggml_tensor * out = ggml_set_rows(ctx, dst, rows, ids);
    ggml_build_forward_expand(graph, out);
    return out;
}

// 自定义填充:
//   F32/F16 随机 [-1,1]
//   I64 ids 填 [0, n1) 的无重复随机排列
static void make_fill(const sr_shape & s, ocl_op_fill_fn & fill) {
    fill = [&s](ggml_tensor * t, uint32_t seed) {
        std::mt19937 rng(seed);

        if (t->type == GGML_TYPE_I64) {
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
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_F16) {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<ggml_fp16_t> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = ggml_fp32_to_fp16(dist(rng));
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    };
}

OCL_OP_TEST(set_rows) {
    const sr_shape cases[] = {
        // 原有 F32 权重路径
        { 128, 4096, 8, 1, 1,  GGML_TYPE_F32, GGML_TYPE_F32, false },
        { 128, 4096, 8, 1, 4,  GGML_TYPE_F32, GGML_TYPE_F32, false },
        { 1024, 512, 1, 1, 16, GGML_TYPE_F32, GGML_TYPE_F32, false },
        { 64,  256, 3, 2, 5,  GGML_TYPE_F32, GGML_TYPE_F32, false },

        // 真实 KV cache 路径: rows=F32, dst=F16, dst 是 view
        { 1024, 12032, 1, 1, 512, GGML_TYPE_F16, GGML_TYPE_F32, true },
        { 128,  4096,  8, 1, 4,   GGML_TYPE_F16, GGML_TYPE_F32, true },
        // F16 -> F16
        { 128,  4096,  8, 1, 4,   GGML_TYPE_F16, GGML_TYPE_F16, true },
        // F16 -> F32
        { 64,   256,   1, 1, 16,  GGML_TYPE_F32, GGML_TYPE_F16, false },
    };

    bool all_ok = true;
    for (const sr_shape & s : cases) {
        ocl_op_fill_fn fill;
        make_fill(s, fill);

        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_set_rows, (void *) &s, seed, fill);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_set_rows, (void *) &s, seed, fill);

        double     nmse = 0.0;
        const bool ok   = ocl_op_compare_nmse(gpu, cpu, 1e-6, &nmse);
        all_ok &= ok;
        printf("  set_rows[%4lld x %5lld x %lld x %lld, ids=%4d, dst=%s, rows=%s, view=%d]: %s (nmse=%.2e, n=%zu)\n",
               (long long) s.n0, (long long) s.n1, (long long) s.n2, (long long) s.n3,
               s.n_ids, ggml_type_name(s.dst_type), ggml_type_name(s.rows_type), s.view ? 1 : 0,
               ok ? "PASS" : "FAIL", nmse, gpu.size());
    }
    return all_ok;
}
