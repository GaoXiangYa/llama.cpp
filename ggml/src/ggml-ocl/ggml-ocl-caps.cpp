// ggml-ocl backend: device capability probe (DESIGN.md section 13)

#include "ggml-ocl-internal.h"

#include <cstring>
#include <cstdio>

// non-fatal device query: on failure zero-fill and return the error
static cl_int ocl_query_device(cl_device_id dev, cl_device_info param, size_t size, void * out) {
    cl_int err = clGetDeviceInfo(dev, param, size, out, nullptr);
    if (err != CL_SUCCESS) {
        memset(out, 0, size);
    }
    return err;
}

static bool ocl_ext_has(const char * ext, const char * name) {
    return ext != nullptr && strstr(ext, name) != nullptr;
}

// 自研 GPU 代际识别: 命名规则待硬件团队提供, 当前返回 0 (未知)
int ggml_ocl_parse_chip(const char * device_name) {
    (void) device_name;
    return 0;
}

// 自研编译器版本契约: 版本串格式待与驱动团队约定, 当前返回 -1
static void ocl_parse_compiler_version(const char * driver_version,
                                       int * compiler_major, int * compiler_minor) {
    (void) driver_version;
    *compiler_major = -1;
    *compiler_minor = -1;
}

void ggml_ocl_caps_probe(ggml_ocl_caps * caps) {
    cl_device_id dev = caps->device;
    GGML_ASSERT(dev != nullptr);

    ocl_query_device(dev, CL_DEVICE_NAME,       sizeof(caps->device_name), caps->device_name);
    ocl_query_device(dev, CL_DRIVER_VERSION,    sizeof(caps->driver_version), caps->driver_version);
    ocl_query_device(dev, CL_DEVICE_VENDOR_ID,  sizeof(caps->vendor_id), &caps->vendor_id);

    char cver[64] = {0};
    if (ocl_query_device(dev, CL_DEVICE_OPENCL_C_VERSION, sizeof(cver), cver) == CL_SUCCESS) {
        int major = 0, minor = 0;
        if (sscanf(cver, "OpenCL C %d.%d", &major, &minor) == 2) {
            caps->ocl_c_major = major;
            caps->ocl_c_minor = minor;
        }
    }

    char ext[4096] = {0};
    ocl_query_device(dev, CL_DEVICE_EXTENSIONS, sizeof(ext), ext);

    caps->fp16             = ocl_ext_has(ext, "cl_khr_fp16");
    caps->dp4a             = ocl_ext_has(ext, "cl_khr_integer_dot_product");
    caps->subgroup_shuffle = ocl_ext_has(ext, "cl_khr_subgroup_shuffle");

    cl_bool image_support = CL_FALSE;
    ocl_query_device(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support);
    caps->images = image_support == CL_TRUE;
    ocl_query_device(dev, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                     sizeof(caps->image_max_buffer), &caps->image_max_buffer);
    caps->image1d_buffer = caps->images && caps->image_max_buffer > 0;

    // subgroup sizes (OpenCL 2.1+); fall back to the confirmed wave size 64
    if (caps->ocl_c_major > 2 || (caps->ocl_c_major == 2 && caps->ocl_c_minor >= 1)) {
        size_t sizes[16] = {0};
        size_t ret = 0;
        if (clGetDeviceInfo(dev, CL_DEVICE_SUBGROUP_SIZES, sizeof(sizes), sizes, &ret) == CL_SUCCESS) {
            const size_t count = ret / sizeof(size_t);
            if (count > 0) {
                caps->subgroups = true;
                caps->wave_size = 0;
                for (size_t i = 0; i < count && i < 16; i++) {
                    caps->wave_size = MAX(caps->wave_size, sizes[i]);
                }
            }
        }
    }

    ocl_query_device(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,   sizeof(caps->max_wg),      &caps->max_wg);
    ocl_query_device(dev, CL_DEVICE_LOCAL_MEM_SIZE,        sizeof(caps->local_mem),   &caps->local_mem);
    ocl_query_device(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE,    sizeof(caps->max_alloc),   &caps->max_alloc);
    ocl_query_device(dev, CL_DEVICE_GLOBAL_MEM_SIZE,       sizeof(caps->global_mem),  &caps->global_mem);

    cl_uint base_align_bits = 0;
    ocl_query_device(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(base_align_bits), &base_align_bits);
    caps->alignment = base_align_bits > 0 ? base_align_bits / 8 : 128;

    cl_command_queue_properties queue_props = 0;
    ocl_query_device(dev, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_props), &queue_props);
    caps->out_of_order_queues = (queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;

    cl_device_svm_capabilities svm_caps = 0;
    ocl_query_device(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps);
    caps->svm = svm_caps != 0;

    caps->chip_gen = ggml_ocl_parse_chip(caps->device_name);
    ocl_parse_compiler_version(caps->driver_version, &caps->compiler_major, &caps->compiler_minor);

    // 带宽无法直接查询, 使用常量 (DESIGN.md section 14.1)
    caps->dram_bandwidth_mbps = GGML_OCL_DRAM_BW_MBPS;
}

void ggml_ocl_caps_print(const ggml_ocl_caps * caps) {
    GGML_LOG_INFO("ggml-ocl: device:      %s\n", caps->device_name);
    GGML_LOG_INFO("ggml-ocl: driver:      %s\n", caps->driver_version);
    GGML_LOG_INFO("ggml-ocl: vendor_id:   0x%04X chip_gen: %d compiler: %d.%d\n",
                  caps->vendor_id, caps->chip_gen, caps->compiler_major, caps->compiler_minor);
    GGML_LOG_INFO("ggml-ocl: OpenCL C:    %d.%d\n", caps->ocl_c_major, caps->ocl_c_minor);
    GGML_LOG_INFO("ggml-ocl: fp16:%d dp4a:%d matrix_unit:%d images:%d image1d:%d subgroup_shuffle:%d\n",
                  caps->fp16, caps->dp4a, caps->has_matrix_unit,
                  caps->images, caps->image1d_buffer, caps->subgroup_shuffle);
    GGML_LOG_INFO("ggml-ocl: wave:%zu max_wg:%zu local_mem:%zu B alignment:%zu\n",
                  caps->wave_size, caps->max_wg, caps->local_mem, caps->alignment);
    GGML_LOG_INFO("ggml-ocl: global_mem: %.2f GB max_alloc: %.2f GB image_max_buffer: %.2f MB\n",
                  caps->global_mem / 1024.0 / 1024.0 / 1024.0,
                  caps->max_alloc / 1024.0 / 1024.0 / 1024.0,
                  caps->image_max_buffer / 1024.0 / 1024.0);
    GGML_LOG_INFO("ggml-ocl: dram_bw:    %zu MB/s (constant) ooo_queues:%d svm:%d\n",
                  caps->dram_bandwidth_mbps, caps->out_of_order_queues, caps->svm);
}
