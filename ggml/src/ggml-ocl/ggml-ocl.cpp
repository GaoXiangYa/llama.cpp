// ggml-ocl backend: backend registration (DESIGN.md sections 4/17, M0 skeleton)

#include "ggml-ocl.h"
#include "ggml-ocl-internal.h"
#include "ggml.h"

#include <memory>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstring>

extern struct ggml_backend_device_i ggml_ocl_device_interface;

// ---------------------------------------------------------------------------
// device probing (DESIGN.md section 11.2)
// ---------------------------------------------------------------------------

static std::vector<ggml_backend_device> g_ggml_ocl_devices;
static std::vector<std::unique_ptr<ggml_ocl_device_context>> g_ggml_ocl_dev_ctxs;

// fp16 存储开关: 必须在权重上传之前定下来, 所以在设备探测阶段设置, 不能等到 backend init
static bool g_ggml_ocl_fp16_storage = false;

bool ggml_ocl_fp16_storage() {
    return g_ggml_ocl_fp16_storage;
}

void ggml_ocl_fp16_storage_set(bool enable) {
    g_ggml_ocl_fp16_storage = enable;
}

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
        char pname[128] = {0};
        char pvendor[128] = {0};
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

        size_t max_alloc = 0;
        clGetDeviceInfo(d.id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);

        cl_uint align_bits = 0;
        clGetDeviceInfo(d.id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits), &align_bits, nullptr);

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

        dev_ctx->max_alloc = max_alloc;
        dev_ctx->alignment = align_bits > 0 ? align_bits / 8 : 128;

        // fp16 存储: F32 张量在设备上以 half 存放。开关必须在权重上传前定下来,
        // 且进程内唯一 (多设备混用不同存储精度不在此支持)。
        {
            char ext[4096] = {0};
            clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, nullptr);
            const bool has_fp16 = strstr(ext, "cl_khr_fp16") != nullptr;

            const char * env  = getenv("GGML_OCL_FP16");
            const bool   want = env != nullptr && env[0] != 0 && atoi(env) != 0;

            if (want && !has_fp16) {
                GGML_LOG_ERROR("ggml-ocl: GGML_OCL_FP16=1 but device has no cl_khr_fp16, disabled.\n");
            }
            ggml_ocl_fp16_storage_set(want && has_fp16);
            GGML_LOG_INFO("ggml-ocl: fp16 storage: %s\n", ggml_ocl_fp16_storage() ? "on" : "off");
        }

        // 权重加载阶段 backend 尚未创建，需要一个设备级 copy queue 上传权重
        cl_int qerr = CL_SUCCESS;
        dev_ctx->q_load = clCreateCommandQueueWithProperties(shared_context, d.id, nullptr, &qerr);
        OCL_CHECK(qerr);

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
#ifdef GGML_OCL_PROFILING
    b->stats.print();
#endif

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

// ---------------------------------------------------------------------------
// 调试: 逐节点检测 NaN / Inf，并可选打印每个节点的数值范围
// 用法:
//   GGML_OCL_CHECK_NAN=1  -> 发现第一个 NaN 时 abort
//   GGML_OCL_DUMP_NODE=1  -> 打印每个 OCL 节点的 min/max/nan/inf
// ---------------------------------------------------------------------------

static bool ggml_ocl_debug_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && atoi(env) != 0;
}

static void ggml_ocl_check_node_nan(ggml_backend_t backend, ggml_tensor * node) {
    const bool check_nan = ggml_ocl_debug_env_enabled("GGML_OCL_CHECK_NAN");
    const bool dump_node = ggml_ocl_debug_env_enabled("GGML_OCL_DUMP_NODE");

    if (!check_nan && !dump_node) {
        return;
    }
    if (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_F16) {
        return;
    }

    // 先把计算队列同步到当前节点完成
    ggml_backend_synchronize(backend);

    std::vector<uint8_t> buf(ggml_nbytes(node));
    ggml_backend_tensor_get(node, buf.data(), 0, buf.size());

    size_t count_nan = 0;
    size_t count_inf = 0;
    double v_min =  INFINITY;
    double v_max = -INFINITY;

    auto scan_one = [&](float v) {
        if (isnan(v)) {
            count_nan++;
        } else if (isinf(v)) {
            count_inf++;
        } else {
            v_min = v < v_min ? v : v_min;
            v_max = v > v_max ? v : v_max;
        }
    };

    if (node->type == GGML_TYPE_F32) {
        const float * data = (const float *) buf.data();
        const size_t n = buf.size() / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            scan_one(data[i]);
        }
    } else {
        const ggml_fp16_t * data = (const ggml_fp16_t *) buf.data();
        const size_t n = buf.size() / sizeof(ggml_fp16_t);
        for (size_t i = 0; i < n; ++i) {
            scan_one(GGML_FP16_TO_FP32(data[i]));
        }
    }

    if (dump_node) {
        GGML_LOG_INFO("ggml-ocl: node op=%-16s name=%-32s type=%-6s ne=[%lld %lld %lld %lld] min=%g max=%g nan=%zu inf=%zu\n",
                      ggml_op_name(node->op), node->name ? node->name : "?",
                      ggml_type_name(node->type),
                      (long long) node->ne[0], (long long) node->ne[1],
                      (long long) node->ne[2], (long long) node->ne[3],
                      v_min, v_max, count_nan, count_inf);
    }

    if (check_nan && count_nan > 0) {
        GGML_LOG_ERROR("ggml-ocl: first NaN detected after op=%s name=%s type=%s ne=[%lld %lld %lld %lld]\n",
                       ggml_op_name(node->op), node->name ? node->name : "?",
                       ggml_type_name(node->type),
                       (long long) node->ne[0], (long long) node->ne[1],
                       (long long) node->ne[2], (long long) node->ne[3]);
        GGML_ABORT("ggml-ocl: NaN detected after %s (%s)\n", ggml_op_name(node->op), node->name ? node->name : "?");
    }
}

// M0: 空骨架 - 过滤视图类节点, 其余跳过 (supports_op 全 false, 不应有计算节点到达)
static ggml_status ggml_ocl_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_ocl_backend * b = (ggml_ocl_backend *) backend->context;

    // 等待 H2D 上传完成 (set_tensor 记录的事件)
    ocl_exec_wait_pending_copies(b);

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

        bool ok = ocl_op_dispatch(b, node);
        GGML_ASSERT(ok && "unsupported op in graph (supports_op should have filtered it)");

        // 调试模式: 每个节点后同步并检查 NaN，定位最先出错的算子
        ggml_ocl_check_node_nan(backend, node);
    }

    ocl_exec_flush(b);
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

    *total = dev_ctx->global_mem_size;
    // 记账值 (DESIGN.md 16.5); multi-buffer 拆分由 ggml-alloc 处理
    // 模型加载阶段 backend 可能还没创建，所以统一使用 dev_ctx->mem_allocated
    *free = *total > dev_ctx->mem_allocated ? *total - dev_ctx->mem_allocated : 0;
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
    cl_command_queue_properties queue_props = 0;
#ifdef GGML_OCL_PROFILING
    queue_props |= CL_QUEUE_PROFILING_ENABLE;
#endif

    auto create_queue = [&](cl_command_queue * q, cl_command_queue_properties props) -> cl_int {
        cl_int err = CL_SUCCESS;
        *q = clCreateCommandQueueWithProperties(b->context, dev_ctx->device, &props, &err);
        if (err == CL_SUCCESS) {
            return err;
        }

        // OpenCL 1.2 驱动可能只完整支持旧的 clCreateCommandQueue。
        // 新 API 失败时回退到旧 API，这样 CL_QUEUE_PROFILING_ENABLE 仍可用。
        cl_int err_old = CL_SUCCESS;
        cl_command_queue q_old = clCreateCommandQueue(b->context, dev_ctx->device, props, &err_old);
        if (err_old == CL_SUCCESS) {
            *q = q_old;
            return err_old;
        }

        return err;
    };

    cl_int qerr = create_queue(&b->q_compute, queue_props);
#ifdef GGML_OCL_PROFILING
    if (qerr != CL_SUCCESS && (queue_props & CL_QUEUE_PROFILING_ENABLE)) {
        GGML_LOG_WARN("ggml-ocl: profiling queue creation failed (%d), fallback to normal queue\n", qerr);
        queue_props &= ~CL_QUEUE_PROFILING_ENABLE;
        qerr = create_queue(&b->q_compute, queue_props);
    } else {
        b->profiling_enabled = true;
    }
#endif
    OCL_CHECK(qerr);
    GGML_ASSERT(b->q_compute != nullptr);

    qerr = create_queue(&b->q_copy, queue_props);
    OCL_CHECK(qerr);
    GGML_ASSERT(b->q_copy != nullptr);

    b->scratch_pool.init(b->context);

#ifdef GGML_OCL_PROFILING
    GGML_LOG_INFO("ggml-ocl: profiling %s\n", b->profiling_enabled ? "enabled" : "disabled (queue fallback)");
#endif

    // 启动期全量编译 kernel (进程级共享, 幂等); 推理期零编译
    b->kmgr = &dev_ctx->kmgr;
    b->kmgr->compile_all(b);

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
            break;
    }
    // 计算类 op 查注册表 (S8+)
    // 注意: 模型加载阶段 dev_ctx->backend 可能还是 nullptr，但此时
    // scheduler 仍需要 supports_op() 来判断权重能否放进 OCL buffer。
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) dev->context;
    return ocl_op_supports(dev_ctx->backend ? &dev_ctx->backend->caps : nullptr, op);
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
