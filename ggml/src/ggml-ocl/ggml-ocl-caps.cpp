// ggml-ocl backend: device capability probe (DESIGN.md section 13)

#include "ggml-ocl-internal.h"

#include <cstdio>
#include <cstring>

// non-fatal device query: on failure zero-fill and return the error
static cl_int ocl_query_device(cl_device_id dev, cl_device_info param, size_t size, void * out) {
    cl_int err = clGetDeviceInfo(dev, param, size, out, nullptr);
    if (err != CL_SUCCESS) {
        memset(out, 0, size);
    }
    return err;
}

void ggml_ocl_caps_probe(ggml_ocl_caps * caps) {
    cl_device_id dev = caps->device;
    GGML_ASSERT(dev != nullptr);

    ocl_query_device(dev, CL_DEVICE_NAME,    sizeof(caps->device_name), caps->device_name);
    ocl_query_device(dev, CL_DRIVER_VERSION, sizeof(caps->driver_version), caps->driver_version);

    char cver[64] = { 0 };
    if (ocl_query_device(dev, CL_DEVICE_OPENCL_C_VERSION, sizeof(cver), cver) == CL_SUCCESS) {
        int major = 0;
        int minor = 0;
        if (sscanf(cver, "OpenCL C %d.%d", &major, &minor) == 2) {
            caps->ocl_c_major = major;
            caps->ocl_c_minor = minor;
        }
    }
}

void ggml_ocl_caps_print(const ggml_ocl_caps * caps) {
    GGML_LOG_INFO("ggml-ocl: device: %s\n", caps->device_name);
    GGML_LOG_INFO("ggml-ocl: driver: %s\n", caps->driver_version);
    GGML_LOG_INFO("ggml-ocl: OpenCL C: %d.%d\n", caps->ocl_c_major, caps->ocl_c_minor);
}
