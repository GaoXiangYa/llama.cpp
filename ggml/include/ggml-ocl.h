#ifndef GGML_OCL_H
#define GGML_OCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ocl_reg(void);
GGML_BACKEND_API bool ggml_backend_is_ocl(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OCL_H
