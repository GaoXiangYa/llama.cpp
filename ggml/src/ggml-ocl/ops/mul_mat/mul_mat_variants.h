#pragma once

#include "ggml-ocl-internal.h"

// MUL_MAT 各 kernel 的 host 入口: 一个 kernel 一个函数, 各自带自己的 launch 配置
namespace ops {

bool gemv_f16_f16_run (ggml_ocl_backend *  b,
                       const ggml_tensor * src0,
                       const ggml_tensor * src1,
                       ggml_tensor *       dst);

bool gemv_q4_1_f16_run(ggml_ocl_backend *  b,
                       const ggml_tensor * src0,
                       const ggml_tensor * src1,
                       ggml_tensor *       dst);

bool gemm_f16_f16_run (ggml_ocl_backend *  b,
                       const ggml_tensor * src0,
                       const ggml_tensor * src1,
                       ggml_tensor *       dst);

bool gemm_q4_1_f16_run(ggml_ocl_backend *  b,
                       const ggml_tensor * src0,
                       const ggml_tensor * src1,
                       ggml_tensor *       dst);

}  // namespace ops
