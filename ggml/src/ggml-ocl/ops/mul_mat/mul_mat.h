#pragma once

#include "ggml-impl.h"
#include "ggml-ocl-internal.h"
#include "ggml.h"

// MUL_MAT 每个 kernel 一个文件、一个 _run 函数, 文件名 == kernel 函数名。
// 每个 _run 里的启动配置 (grid 倍率 / tile) 必须和对应 .cl 里写死的一致,
// 不一致不会报错, 只会静默算错。

bool gemm_f16_f16_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemm_f16_f32_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemm_f32_f16_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemm_f32_f32_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemm_q4_1_f16_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemm_q4_1_f32_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemv_f16_f16_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemv_f16_f32_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemv_f32_f16_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemv_f32_f32_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemv_q4_1_f16_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool gemv_q4_1_f32_run(ggml_ocl_backend * b, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
