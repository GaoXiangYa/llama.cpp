#include "ops.h"

#include <cstdlib>

namespace ops {
bool fuse_debug() {
    static const bool on = getenv("GGML_OCL_FUSE_DEBUG") != nullptr;
    return on;
}

static bool fuse_weight_1d_ok(const ggml_tensor * w, const ggml_tensor * x) {
    return w != nullptr && w->type == GGML_TYPE_F32 && w->buffer != nullptr && w->ne[0] == x->ne[0] && w->ne[1] == 1 &&
           w->ne[2] == 1 && w->ne[3] == 1;
}

static int fuse_rmsnorm_mul(ggml_ocl_backend * b, ggml_cgraph * graph, int i) {
    if (!ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        return 0;
    }
    ggml_tensor * rms = graph->nodes[i];
    ggml_tensor * mul = graph->nodes[i + 1];
    if (!fuse_weight_1d_ok(mul->src[1], rms)) {
        return 0;
    }
    return rmsnorm_mul_run(b, rms->src[0], mul->src[1], mul) ? 1 : 0;
}

static int fuse_add_rmsnorm_mul(ggml_ocl_backend * b, ggml_cgraph * graph, int i) {
    if (i + 2 >= graph->n_nodes) {
        return 0;
    }
    ggml_tensor * add = graph->nodes[i];
    ggml_tensor * rms = graph->nodes[i + 1];
    ggml_tensor * mul = graph->nodes[i + 2];
    if (add->op != GGML_OP_ADD || rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return 0;
    }
    if (rms->src[0] != add || mul->src[0] != rms) {
        return 0;
    }
    if (ggml_node_get_use_count(graph, i + 1) != 1) {  // norm 只能喂给 mul
        return 0;
    }
    if (!fuse_weight_1d_ok(mul->src[1], add) || add->data == mul->data) {
        return 0;
    }
    return add_rmsnorm_mul_run(b, add, rms, mul->src[1], mul) ? 2 : 0;
}

int try_fuse(ggml_ocl_backend * b, ggml_cgraph * graph, int i) {
    if (int n = fuse_add_rmsnorm_mul(b, graph, i)) {
        return n;
    }
    if (int n = fuse_rmsnorm_mul(b, graph, i)) {
        return n;
    }
    if (fuse_debug()) {
        GGML_LOG_INFO("ggml-ocl: no fusion at node %d (%s)\n", i, ggml_op_name(graph->nodes[i]->op));
    }
    return 0;
}

}  // namespace ops
