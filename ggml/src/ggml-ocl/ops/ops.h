#pragma once

#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ggml.h"

namespace ops {

bool add_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool add_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);
bool add_rmsnorm_mul_run(ggml_ocl_backend *  b,
                         const ggml_tensor * add,
                         const ggml_tensor * norm,
                         const ggml_tensor * w,
                         ggml_tensor *       out);

bool cpy_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool cpy_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool get_rows_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool get_rows_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool glu_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool glu_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool mul_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool mul_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool mul_mat_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool mul_mat_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool rmsnorm_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool rmsnorm_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);
bool rmsnorm_mul_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool rope_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool rope_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool set_rows_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool set_rows_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool softmax_supports(const ggml_ocl_caps * caps, const ggml_tensor * op);
bool softmax_run(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst);

bool fuse_debug();
int  try_fuse(ggml_ocl_backend * b, ggml_cgraph * graph, int i);
}  // namespace ops
