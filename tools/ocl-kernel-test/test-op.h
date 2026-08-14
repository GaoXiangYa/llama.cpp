#pragma once
// ggml-ocl 算子级测试框架 (DESIGN.md section 23)
// 用例用 ggml 公开 API 构造算子图, 在 ggml-ocl backend 执行, 与 ggml-cpu 结果比较.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

struct ocl_op_test {
    const char * name;
    // 测试入口: 用例自由控制 shape 循环, 内部用 ocl_op_eval/ocl_op_compare
    // 返回 true 表示全部 shape 通过
    bool (*run)(ggml_backend_t backend_ocl, ggml_backend_t backend_cpu, uint32_t seed);
};

inline std::vector<ocl_op_test> & ocl_op_test_registry() {
    static std::vector<ocl_op_test> registry;
    return registry;
}

inline bool ocl_op_test_register(const char * name, bool (*run)(ggml_backend_t, ggml_backend_t, uint32_t)) {
    ocl_op_test_registry().push_back({ name, run });
    return true;
}

#define OCL_OP_TEST(name)                                                      \
    static bool ocl_op_test_##name(ggml_backend_t backend_ocl,                 \
                                   ggml_backend_t backend_cpu, uint32_t seed); \
    static const bool ocl_op_registered_##name =                               \
        ocl_op_test_register(#name, ocl_op_test_##name);                       \
    static bool ocl_op_test_##name(ggml_backend_t backend_ocl,                 \
                                   ggml_backend_t backend_cpu, uint32_t seed)

// 算子图构建回调 (用例提供): 用 ggml API 建图, 返回输出 tensor (F32)
typedef ggml_tensor * (*ocl_op_build_fn)(ggml_context * ctx, ggml_cgraph * graph, void * userdata);

// 在指定 backend 上执行算子图并返回输出 (F32)
// 流程: 建 ctx -> build -> 分配 -> F32 叶子填随机输入 -> compute -> 读输出 -> 清理
inline std::vector<float> ocl_op_eval(ggml_backend_t backend, ocl_op_build_fn build,
                                      void * userdata, uint32_t seed) {
    ggml_init_params params = {
        /*.mem_size   =*/ 512 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_tensor * out = build(ctx, graph, userdata);
    GGML_ASSERT(out != nullptr);
    GGML_ASSERT(out->type == GGML_TYPE_F32 && "S9: only f32 outputs supported");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    // F32 叶子输入填随机数据 (seed 固定, OCL/CPU 两次 eval 输入一致)
    uint32_t t_idx = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op != GGML_OP_NONE || t->view_src != nullptr) {
            continue;
        }
        if (t->type != GGML_TYPE_F32) {
            continue;
        }
        std::mt19937 rng(seed + t_idx++);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dist(rng);
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// 比较两个 backend 的结果 (相对误差, 默认 1e-5)
inline bool ocl_op_compare(const std::vector<float> & gpu, const std::vector<float> & cpu,
                           double * max_rel_err_out = nullptr) {
    GGML_ASSERT(gpu.size() == cpu.size());
    double max_rel_err = 0.0;
    for (size_t i = 0; i < gpu.size(); i++) {
        const double ref_abs = std::fabs((double) cpu[i]);
        const double diff    = std::fabs((double) gpu[i] - (double) cpu[i]);
        const double rel     = diff / (ref_abs > 1e-12 ? ref_abs : 1e-12);
        max_rel_err = std::max(max_rel_err, rel);
    }
    if (max_rel_err_out) {
        *max_rel_err_out = max_rel_err;
    }
    return max_rel_err <= 1e-5;
}
