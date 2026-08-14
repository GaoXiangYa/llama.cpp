// ggml-ocl op test: ADD (DESIGN.md 23.2 用例模板, 纯 ggml 公开 API)

#include "../test-op.h"

struct add_shape {
    int64_t n0, n1;
};

static ggml_tensor * build_add(ggml_context * ctx, ggml_cgraph * graph, void * userdata) {
    const add_shape * s = (const add_shape *) userdata;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->n0, s->n1);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s->n0, s->n1);
    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_build_forward_expand(graph, out);
    return out;
}

OCL_OP_TEST(add) {
    const add_shape shapes[] = {
        { 1024,   1   },
        { 1024,   512 },
        { 2048,   2048},
        { 151936, 1   },   // vocab 宽
    };

    bool all_ok = true;
    for (const add_shape & s : shapes) {
        const std::vector<float> gpu = ocl_op_eval(backend_ocl, build_add, (void *) &s, seed);
        const std::vector<float> cpu = ocl_op_eval(backend_cpu, build_add, (void *) &s, seed);

        double max_rel_err = 0.0;
        const bool ok = ocl_op_compare(gpu, cpu, &max_rel_err);
        all_ok &= ok;
        printf("  add[%5lld x %5lld]: %s (max_rel_err=%.2e, n=%zu)\n",
               (long long) s.n0, (long long) s.n1, ok ? "PASS" : "FAIL", max_rel_err, gpu.size());
    }
    return all_ok;
}
