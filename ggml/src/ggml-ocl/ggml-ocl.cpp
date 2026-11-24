// ggml-ocl backend: backend registration (DESIGN.md sections 4/17, M0 skeleton)

#include "ggml-ocl.h"

#include "ggml-ocl-internal.h"
#include "ggml.h"
#include "ops/ops.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

extern struct ggml_backend_device_i ggml_ocl_device_interface;

static std::vector<ggml_backend_device>                      g_ggml_ocl_devices;
static std::vector<std::unique_ptr<ggml_ocl_device_context>> g_ggml_ocl_dev_ctxs;

static std::vector<ggml_backend_device> ggml_ocl_probe_devices(ggml_backend_reg * reg) {
    std::vector<ggml_backend_device> found;

    enum { NPLAT = 16, NDEV = 16 };

    cl_platform_id platform_ids[NPLAT];
    cl_uint        n_platforms = 0;
    if (clGetPlatformIDs(NPLAT, platform_ids, &n_platforms) != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-ocl: platform IDs not available.\n");
        return found;
    }

    struct dev_info {
        cl_platform_id platform;
        cl_device_id   id;
        cl_device_type type;
        char           name[128];
    };

    std::vector<dev_info> devices;

    cl_platform_id default_platform = nullptr;
    cl_device_id   default_device   = nullptr;

    for (cl_uint i = 0; i < n_platforms; i++) {
        char pname[128]   = { 0 };
        char pvendor[128] = { 0 };
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, sizeof(pvendor), pvendor, nullptr);

        cl_device_id device_ids[NDEV];
        cl_uint      n_devices = 0;
        cl_int       err       = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, NDEV, device_ids, &n_devices);
        if (err == CL_DEVICE_NOT_FOUND) {
            continue;
        }
        if (err != CL_SUCCESS) {
            continue;
        }

        for (cl_uint j = 0; j < n_devices; j++) {
            dev_info d = { platform_ids[i], device_ids[j], 0, { 0 } };
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
                char pname[128] = { 0 };
                clGetPlatformInfo(d.platform, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
                match = match && strstr(pname, user_platform) != nullptr;
            }
            if (user_device != nullptr && user_device[0] != 0) {
                match = match && strstr(d.name, user_device) != nullptr;
            }
            if (match) {
                selected         = d.id;
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

    cl_int                err;
    cl_context_properties props[] = { (intptr_t) CL_CONTEXT_PLATFORM, (intptr_t) default_platform, 0 };
    cl_context            shared_context =
        clCreateContext(props, platform_devices.size(), platform_devices.data(), nullptr, nullptr, &err);
    OCL_CHECK(err);

    for (const dev_info & d : devices) {
        if (d.platform != default_platform) {
            continue;
        }

        size_t global_mem = 0;
        clGetDeviceInfo(d.id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);

        size_t max_alloc = 0;
        clGetDeviceInfo(d.id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);

        // 单块 cl_mem 过大时该设备 (Glenfly Arise) 会出现寻址错乱: 1.7B 权重整块 1018 MiB 时,
        // 偏移 540 MB 之后的层读到的权重是错的 (mul 用 0~6.5 的权重算出 6.4e4).
        // 限制单块上限, ggml-alloc 会按张量边界自动把大 buffer 拆成多个小 cl_mem.
        // 单个张量大于上限时仍会单独分配一整块, 不会失败.
        constexpr size_t MAX_ALLOC_CAP = 256ull * 1024 * 1024;
        if (max_alloc > MAX_ALLOC_CAP) {
            max_alloc = MAX_ALLOC_CAP;
        }

        // 调试/覆盖: GGML_OCL_MAX_ALLOC_MB 可继续压低上限
        if (const char * env = getenv("GGML_OCL_MAX_ALLOC_MB")) {
            const long mb = atol(env);
            if (mb > 0 && (size_t) mb * 1024 * 1024 < max_alloc) {
                max_alloc = (size_t) mb * 1024 * 1024;
            }
        }

        cl_uint align_bits = 0;
        clGetDeviceInfo(d.id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits), &align_bits, nullptr);

        auto dev_ctx             = std::unique_ptr<ggml_ocl_device_context>(new ggml_ocl_device_context());
        dev_ctx->device          = d.id;
        dev_ctx->device_name     = d.name;
        dev_ctx->global_mem_size = global_mem;
        dev_ctx->context         = shared_context;

        dev_ctx->max_alloc = max_alloc;
        dev_ctx->alignment = align_bits > 0 ? align_bits / 8 : 128;

        // fp16 存储是固定策略: 设备不支持则跳过, 由上层回退到 CPU
        {
            char ext[4096] = { 0 };
            clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, nullptr);
            if (strstr(ext, "cl_khr_fp16") == nullptr) {
                GGML_LOG_ERROR("ggml-ocl: device %s has no cl_khr_fp16, skipped\n", d.name);
                continue;
            }
        }

        cl_int qerr     = CL_SUCCESS;
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

    // 释放队列前排空: 否则在途拷贝可能被丢弃
    ocl_exec_wait_pending_copies(b);
    if (b->q_copy) {
        OCL_CHECK(clFinish(b->q_copy));
    }
    if (b->q_compute) {
        OCL_CHECK(clFinish(b->q_compute));
    }

    if (b->q_compute) {
        OCL_CHECK(clReleaseCommandQueue(b->q_compute));
    }
    if (b->q_copy) {
        OCL_CHECK(clReleaseCommandQueue(b->q_copy));
    }
#ifdef GGML_OCL_PROFILING
    ocl_prof_flush(b);
    b->stats.print();
#endif

    // context 为进程级共享 (dev_ctx 持有), 不随 backend 释放;
    // backend 可能被 sched 多次 init/free
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) backend->device->context;
    dev_ctx->backend                  = nullptr;

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

static void ggml_ocl_dump_tensor_stats(ggml_backend_t backend, ggml_tensor * node, const char * tag) {
    if (node == nullptr) {
        return;
    }
    if (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_F16) {
        return;
    }
    ggml_ocl_tensor_extra * ex = (ggml_ocl_tensor_extra *) node->extra;

    // 先把计算队列同步到当前节点完成
    ggml_backend_synchronize(backend);

    std::vector<uint8_t> buf(ggml_nbytes(node));
    ggml_backend_tensor_get(node, buf.data(), 0, buf.size());

    size_t count_nan = 0;
    size_t count_inf = 0;
    double v_min     = INFINITY;
    double v_max     = -INFINITY;

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
        const size_t  n    = buf.size() / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            scan_one(data[i]);
        }
    } else {
        const ggml_fp16_t * data = (const ggml_fp16_t *) buf.data();
        const size_t        n    = buf.size() / sizeof(ggml_fp16_t);
        for (size_t i = 0; i < n; ++i) {
            scan_one(GGML_FP16_TO_FP32(data[i]));
        }
    }

    GGML_LOG_INFO(
        "ggml-ocl: %-5s op=%-10s name=%-34s ne=[%lld %lld %lld %lld] min=%g max=%g nan=%zu inf=%zu dev=%p off=%llu "
        "bufsz=%zu\n",
        tag, ggml_op_name(node->op), node->name ? node->name : "?", (long long) node->ne[0], (long long) node->ne[1],
        (long long) node->ne[2], (long long) node->ne[3], v_min, v_max, count_nan, count_inf,
        ex ? (void *) ex->data_device : nullptr, ex ? (unsigned long long) ex->offset : 0ULL,
        node->buffer ? ggml_backend_buffer_get_size(node->buffer) : 0);
}

static void ggml_ocl_check_node_nan(ggml_backend_t backend, ggml_tensor * node) {
    const bool check_nan = ggml_ocl_debug_env_enabled("GGML_OCL_CHECK_NAN");
    const bool dump_node = ggml_ocl_debug_env_enabled("GGML_OCL_DUMP_NODE");

    if (!check_nan && !dump_node) {
        return;
    }
    ggml_ocl_dump_tensor_stats(backend, node, dump_node ? "node" : "nan");
    if (check_nan) {
        // NaN 检查本身由 dump 打印, 这里只在发现时 abort
    }
}

// 打印图里所有叶子张量 (模型权重等) 的数值与显存位置, 用于核对上传
static void ggml_ocl_dump_graph_leaves(ggml_backend_t backend, ggml_cgraph * cgraph) {
    if (!ggml_ocl_debug_env_enabled("GGML_OCL_DUMP_LEAF")) {
        return;
    }
    static std::vector<const ggml_tensor *> dumped;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = cgraph->nodes[i]->src[j];
            if (src == nullptr || src->buffer == nullptr || src->name[0] == 0) {
                continue;
            }
            bool seen = false;
            for (const ggml_tensor * d : dumped) {
                if (d == src) {
                    seen = true;
                    break;
                }
            }
            if (seen) {
                continue;
            }
            dumped.push_back(src);
            ggml_ocl_dump_tensor_stats(backend, src, "leaf");
        }
    }
}

static ggml_status ggml_ocl_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_ocl_backend * b = (ggml_ocl_backend *) backend->context;

    ocl_exec_wait_pending_copies(b);
    ggml_ocl_dump_graph_leaves(backend, cgraph);

#ifdef GGML_OCL_PROFILING
    const int64_t prof_t0 = ggml_time_us();
#endif
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

        const int n = ops::try_fuse(b, cgraph, i);
        if (n != 0) {
            if (ops::fuse_debug()) {
                GGML_LOG_INFO("ggml-ocl: fused %d nodes at %d (%s)\n", n + 1, i, node->name);
            }
            i += n;
            continue;
        }

        // 调试: dump 指定节点在执行前的 src/dst 数值 (GGML_OCL_DUMP_BEFORE=<名字子串>)
        const char * before = getenv("GGML_OCL_DUMP_BEFORE");
        if (before != nullptr && before[0] != 0 && strstr(node->name, before) != nullptr) {
            GGML_LOG_INFO("ggml-ocl: === before node %s (%s) ===\n", node->name, ggml_op_name(node->op));
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = node->src[j];
                if (src != nullptr && src->buffer != nullptr) {
                    ggml_ocl_dump_tensor_stats(backend, src, "pre");
                }
            }
            ggml_ocl_dump_tensor_stats(backend, node, "pre");
            for (int j = 0; j < 2; j++) {
                ggml_tensor * t = node->src[j];
                if (t == nullptr) {
                    continue;
                }
                ggml_ocl_tensor_extra * e = (ggml_ocl_tensor_extra *) t->extra;
                GGML_LOG_INFO(
                    "ggml-ocl: arg%d name=%s extra=%p dev=%p off=%llu nb0=%llu nb1=%llu ne=[%lld %lld] view_offs=%zu\n",
                    j, t->name, (void *) e, e ? (void *) e->data_device : nullptr,
                    e ? (unsigned long long) ocl_dev_offset(t, e) : 0ULL, (unsigned long long) ocl_nb(t, t->nb[0]),
                    (unsigned long long) ocl_nb(t, t->nb[1]), (long long) t->ne[0], (long long) t->ne[1], t->view_offs);
            }
            {
                ggml_ocl_tensor_extra * e = (ggml_ocl_tensor_extra *) node->extra;
                GGML_LOG_INFO("ggml-ocl: argd name=%s extra=%p dev=%p off=%llu nb0=%llu nb1=%llu ne=[%lld %lld]\n",
                              node->name, (void *) e, e ? (void *) e->data_device : nullptr,
                              e ? (unsigned long long) ocl_dev_offset(node, e) : 0ULL,
                              (unsigned long long) ocl_nb(node, node->nb[0]),
                              (unsigned long long) ocl_nb(node, node->nb[1]), (long long) node->ne[0],
                              (long long) node->ne[1]);
            }
        }

        bool ok = ocl_op_dispatch(b, node);
        GGML_ASSERT(ok && "unsupported op in graph (supports_op should have filtered it)");

        if (before != nullptr && before[0] != 0 && strstr(node->name, before) != nullptr) {
            GGML_LOG_INFO("ggml-ocl: === after node %s (%s) ===\n", node->name, ggml_op_name(node->op));
            ggml_ocl_dump_tensor_stats(backend, node, "post");
        }

        ggml_ocl_check_node_nan(backend, node);
    }

#ifdef GGML_OCL_PROFILING
    b->stats.host_ns += (cl_ulong) (ggml_time_us() - prof_t0) * 1000ULL;
#endif

    ocl_exec_flush(b);

#ifdef GGML_OCL_PROFILING
    ocl_prof_flush(b);
#endif
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
    *free  = *total > dev_ctx->mem_allocated ? *total - dev_ctx->mem_allocated : 0;
}

static enum ggml_backend_dev_type ggml_ocl_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_ocl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_ocl_device_get_name(dev);
    props->description = ggml_ocl_device_get_description(dev);
    ggml_ocl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type      = ggml_ocl_device_get_type(dev);
    props->device_id = nullptr;

    props->caps.async                = false;
    props->caps.host_buffer          = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events               = false;
}

static ggml_backend_t ggml_ocl_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) dev->context;

    ggml_ocl_backend * b = new ggml_ocl_backend{};
    b->caps.device       = dev_ctx->device;
    ggml_ocl_caps_probe(&b->caps);
    ggml_ocl_caps_print(&b->caps);

    b->context                              = dev_ctx->context;
    cl_command_queue_properties queue_props = 0;
#ifdef GGML_OCL_PROFILING
    queue_props |= CL_QUEUE_PROFILING_ENABLE;
#endif

    auto create_queue = [&](cl_command_queue * q, cl_command_queue_properties props) -> cl_int {
        cl_int err = CL_SUCCESS;
        *q         = clCreateCommandQueueWithProperties(b->context, dev_ctx->device, &props, &err);
        if (err == CL_SUCCESS) {
            return err;
        }

        cl_int           err_old = CL_SUCCESS;
        cl_command_queue q_old   = clCreateCommandQueue(b->context, dev_ctx->device, props, &err_old);
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

#ifdef GGML_OCL_PROFILING
    GGML_LOG_INFO("ggml-ocl: profiling %s\n", b->profiling_enabled ? "enabled" : "disabled (queue fallback)");
#endif

    // 启动期全量编译 kernel (进程级共享, 幂等); 推理期零编译
    b->kmgr = &dev_ctx->kmgr;
    b->kmgr->compile_all(b);

    dev_ctx->backend = b;

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

    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) dev->context;
    return ocl_op_supports(dev_ctx->backend ? &dev_ctx->backend->caps : nullptr, op);
}

static bool ggml_ocl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (dev->iface.get_name != ggml_ocl_device_get_name || buft->iface.get_name != ggml_ocl_buffer_type_get_name) {
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
    static std::mutex           mutex;
    static ggml_backend_reg     reg;
    static bool                 initialized = false;
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
