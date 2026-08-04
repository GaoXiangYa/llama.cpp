// ggml-ocl backend: backend registration (DESIGN.md sections 4/17, M0 skeleton)

#include "ggml-ocl.h"
#include "ggml-ocl-internal.h"

#include <memory>
#include <string>
#include <vector>

extern struct ggml_backend_device_i ggml_ocl_device_interface;

// ---------------------------------------------------------------------------
// device probing (DESIGN.md section 11.2)
// ---------------------------------------------------------------------------

static std::vector<ggml_backend_device> g_ggml_ocl_devices;
static std::vector<std::unique_ptr<ggml_ocl_device_context>> g_ggml_ocl_dev_ctxs;

static std::vector<ggml_backend_device> ggml_ocl_probe_devices(ggml_backend_reg * reg) {
    std::vector<ggml_backend_device> found;

    enum { NPLAT = 16, NDEV = 16 };

    cl_platform_id platform_ids[NPLAT];
    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(NPLAT, platform_ids, &n_platforms) != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-ocl: platform IDs not available.\n");
        return found;
    }

    struct dev_info {
        cl_platform_id platform;
        cl_device_id   id;
        cl_device_type type;
        char name[128];
    };
    std::vector<dev_info> devices;

    cl_platform_id default_platform = nullptr;
    cl_device_id   default_device   = nullptr;

    for (cl_uint i = 0; i < n_platforms; i++) {
        char pname[128] = {0}, pvendor[128] = {0};
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, sizeof(pvendor), pvendor, nullptr);

        cl_device_id device_ids[NDEV];
        cl_uint n_devices = 0;
        cl_int err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, NDEV, device_ids, &n_devices);
        if (err == CL_DEVICE_NOT_FOUND) {
            continue;
        }
        if (err != CL_SUCCESS) {
            continue;
        }

        for (cl_uint j = 0; j < n_devices; j++) {
            dev_info d = { platform_ids[i], device_ids[j], 0, {0} };
            clGetDeviceInfo(d.id, CL_DEVICE_TYPE, sizeof(d.type), &d.type, nullptr);
            clGetDeviceInfo(d.id, CL_DEVICE_NAME, sizeof(d.name), d.name, nullptr);
            devices.push_back(d);

            if (default_device == nullptr && d.type == CL_DEVICE_TYPE_GPU) {
                default_platform = platform_ids[i];
                default_device   = d.id;
            }
        }
    }

    if (devices.empty()) {
        GGML_LOG_ERROR("ggml-ocl: no OpenCL devices found.\n");
        return found;
    }

    // env overrides: GGML_OCL_PLATFORM / GGML_OCL_DEVICE (index or name substring)
    const char * user_platform = getenv("GGML_OCL_PLATFORM");
    const char * user_device   = getenv("GGML_OCL_DEVICE");

    cl_device_id selected = default_device;
    if (user_platform != nullptr || user_device != nullptr) {
        selected = nullptr;
        for (const dev_info & d : devices) {
            bool match = true;
            if (user_platform != nullptr && user_platform[0] != 0) {
                char pname[128] = {0};
                clGetPlatformInfo(d.platform, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
                match = match && strstr(pname, user_platform) != nullptr;
            }
            if (user_device != nullptr && user_device[0] != 0) {
                match = match && strstr(d.name, user_device) != nullptr;
            }
            if (match) {
                selected = d.id;
                default_platform = d.platform;
                break;
            }
        }
        if (selected == nullptr) {
            GGML_LOG_ERROR("ggml-ocl: no device matching GGML_OCL_PLATFORM/DEVICE found.\n");
            return found;
        }
    }

    GGML_ASSERT(selected != nullptr);

    // shared context for all devices of the selected platform
    std::vector<cl_device_id> platform_devices;
    for (const dev_info & d : devices) {
        if (d.platform == default_platform) {
            platform_devices.push_back(d.id);
        }
    }

    cl_int err;
    cl_context_properties props[] = {
        (intptr_t) CL_CONTEXT_PLATFORM, (intptr_t) default_platform, 0
    };
    cl_context shared_context = clCreateContext(props, platform_devices.size(),
                                                platform_devices.data(), nullptr, nullptr, &err);
    OCL_CHECK(err);

    for (const dev_info & d : devices) {
        if (d.platform != default_platform) {
            continue;
        }

        char pname[128] = {0};
        clGetPlatformInfo(d.platform, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);

        size_t global_mem = 0;
        clGetDeviceInfo(d.id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);

        auto dev_ctx = std::unique_ptr<ggml_ocl_device_context>(new ggml_ocl_device_context{
            /* .platform        = */ d.platform,
            /* .device          = */ d.id,
            /* .platform_name   = */ pname,
            /* .device_name     = */ d.name,
            /* .device_type     = */ d.type,
            /* .global_mem_size = */ global_mem,
            /* .context         = */ shared_context,
            /* .context_refs    = */ 0,
            /* .backend         = */ nullptr,
            /* .buffer_type     = */ {},
        });

        found.push_back(ggml_backend_device{
            /* .iface   = */ ggml_ocl_device_interface,
            /* .reg     = */ reg,
            /* .context = */ dev_ctx.get(),
        });

        g_ggml_ocl_dev_ctxs.push_back(std::move(dev_ctx));
    }

    return found;
}

// ---------------------------------------------------------------------------
// backend (stream)
// ---------------------------------------------------------------------------

static const char * ggml_ocl_backend_name(ggml_backend_t backend) {
    return "OCL";
    GGML_UNUSED(backend);
}

static void ggml_ocl_backend_free(ggml_backend_t backend) {
    ggml_ocl_backend * b = (ggml_ocl_backend *) backend->context;

    if (b->q_compute) {
        OCL_CHECK(clReleaseCommandQueue(b->q_compute));
    }
    if (b->q_copy) {
        OCL_CHECK(clReleaseCommandQueue(b->q_copy));
    }

    // context 为进程级共享 (dev_ctx 持有), 不随 backend 释放;
    // backend 可能被 sched 多次 init/free
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) backend->device->context;
    dev_ctx->backend = nullptr;

    delete b;
    delete backend;
}

static void ggml_ocl_backend_synchronize(ggml_backend_t backend) {
    ggml_ocl_backend * b = (ggml_ocl_backend *) backend->context;
    OCL_CHECK(clFinish(b->q_compute));
}

// M0: 空骨架 - 过滤视图类节点, 其余跳过 (supports_op 全 false, 不应有计算节点到达)
static ggml_status ggml_ocl_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_ocl_backend * b = (ggml_ocl_backend *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node)) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_NONE:
                continue;
            default:
                break;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        GGML_LOG_WARN("ggml-ocl: unsupported op skipped (%s)\n", ggml_op_name(node->op));
    }

    clFlush(b->q_compute);
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_ocl_backend_interface = {
    /* .get_name          = */ ggml_ocl_backend_name,
    /* .free              = */ ggml_ocl_backend_free,
    /* .set_tensor_async  = */ nullptr,
    /* .get_tensor_async  = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async  = */ nullptr,
    /* .synchronize       = */ ggml_ocl_backend_synchronize,
    /* .graph_plan_create = */ nullptr,
    /* .graph_plan_free   = */ nullptr,
    /* .graph_plan_update = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute     = */ ggml_ocl_backend_graph_compute,
    /* .event_record      = */ nullptr,
    /* .event_wait        = */ nullptr,
    /* .graph_optimize    = */ nullptr,
};

// ---------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------

static const char * ggml_ocl_device_get_name(ggml_backend_dev_t dev) {
    return "GPUOCL";
    GGML_UNUSED(dev);
}

static const char * ggml_ocl_device_get_description(ggml_backend_dev_t dev) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) dev->context;
    return dev_ctx->device_name.c_str();
}

static void ggml_ocl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) dev->context;

    // M0: 无记账, 用 global size 减 1 GiB 余量; S6 起改为池记账
    static const size_t margin = 1ull * 1024 * 1024 * 1024;
    *total = dev_ctx->global_mem_size;
    *free  = *total > margin ? *total - margin : 0;
}

static enum ggml_backend_dev_type ggml_ocl_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_ocl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_ocl_device_get_name(dev);
    props->description = ggml_ocl_device_get_description(dev);
    ggml_ocl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = ggml_ocl_device_get_type(dev);
    props->device_id = nullptr;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ggml_ocl_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) dev->context;

    ggml_ocl_backend * b = new ggml_ocl_backend{};
    b->caps.device = dev_ctx->device;
    b->caps.platform = dev_ctx->platform;
    ggml_ocl_caps_probe(&b->caps);
    ggml_ocl_caps_print(&b->caps);

    b->context = dev_ctx->context;
    OCL_CHECK((b->q_compute = clCreateCommandQueueWithProperties(b->context, dev_ctx->device, nullptr, nullptr), 0));
    OCL_CHECK((b->q_copy    = clCreateCommandQueueWithProperties(b->context, dev_ctx->device, nullptr, nullptr), 0));

    b->scratch_pool.init(b->context);

    dev_ctx->backend = b;
    dev_ctx->context_refs++;

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ nullptr,
        /* .iface   = */ ggml_ocl_backend_interface,
        /* .device  = */ dev,
        /* .context = */ b,
    };

    return backend;
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_ocl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) dev->context;

    dev_ctx->buffer_type = ggml_backend_buffer_type{
        /* .iface   = */ ggml_ocl_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };

    return &dev_ctx->buffer_type;
}

static bool ggml_ocl_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // 承载类 op 必须支持 (预分配张量如 KV cache 会落在本后端 buffer 上)
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        default:
            // M0: 计算类 op 全部回退 CPU; S8 起接入 op 注册表
            GGML_UNUSED(dev);
            return false;
    }
}

static bool ggml_ocl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (dev->iface.get_name != ggml_ocl_device_get_name ||
        buft->iface.get_name != ggml_ocl_buffer_type_get_name) {
        return false;
    }
    ggml_ocl_device_context * dev_ctx0 = (ggml_ocl_device_context *) dev->context;
    ggml_ocl_device_context * dev_ctx1 = (ggml_ocl_device_context *) buft->device->context;
    return dev_ctx0->context == dev_ctx1->context;
}

struct ggml_backend_device_i ggml_ocl_device_interface = {
    /* .get_name             = */ ggml_ocl_device_get_name,
    /* .get_description      = */ ggml_ocl_device_get_description,
    /* .get_memory           = */ ggml_ocl_device_get_memory,
    /* .get_type             = */ ggml_ocl_device_get_type,
    /* .get_props            = */ ggml_ocl_device_get_props,
    /* .init_backend         = */ ggml_ocl_device_init,
    /* .get_buffer_type      = */ ggml_ocl_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_ocl_device_supports_op,
    /* .supports_buft        = */ ggml_ocl_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---------------------------------------------------------------------------
// backend registry
// ---------------------------------------------------------------------------

static const char * ggml_ocl_reg_get_name(ggml_backend_reg_t reg) {
    return "OCL";
    GGML_UNUSED(reg);
}

static size_t ggml_ocl_reg_device_count(ggml_backend_reg_t reg) {
    return g_ggml_ocl_devices.size();
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_ocl_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < g_ggml_ocl_devices.size());
    return &g_ggml_ocl_devices[index];
    GGML_UNUSED(reg);
}

static struct ggml_backend_reg_i ggml_ocl_reg_interface = {
    /* .get_name         = */ ggml_ocl_reg_get_name,
    /* .device_count     = */ ggml_ocl_reg_device_count,
    /* .device_get       = */ ggml_ocl_reg_device_get,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_ocl_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);

    if (initialized) {
        return &reg;
    }
    initialized = true;

    g_ggml_ocl_devices = ggml_ocl_probe_devices(&reg);

    reg = ggml_backend_reg{
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_ocl_reg_interface,
        /* .context     = */ nullptr,
    };

    return &reg;
}

bool ggml_backend_is_ocl(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_ocl_backend_name;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ocl_reg)
