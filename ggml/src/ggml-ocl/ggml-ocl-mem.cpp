// ggml-ocl backend: memory management (DESIGN.md section 16)

#include "ggml-ocl.h"
#include "ggml-ocl-internal.h"

// ---------------------------------------------------------------------------
// extra pool: 每个 tensor 一份 extra (buffer reset 时回收复用)
// ---------------------------------------------------------------------------

ggml_ocl_tensor_extra * ocl_extra_pool::alloc() {
    if (!free_.empty()) {
        ggml_ocl_tensor_extra * extra = free_.back();
        free_.pop_back();
        extra->reset();
        return extra;
    }
    ggml_ocl_tensor_extra * extra = new ggml_ocl_tensor_extra();
    all_.push_back(extra);
    return extra;
}

void ocl_extra_pool::reset() {
    for (ggml_ocl_tensor_extra * extra : all_) {
        extra->reset();
        free_.push_back(extra);
    }
}

void ocl_extra_pool::free_all() {
    for (ggml_ocl_tensor_extra * extra : all_) {
        delete extra;
    }
    all_.clear();
    free_.clear();
}

// ---------------------------------------------------------------------------
// buffer (DESIGN.md 16.2, AoS only for M0)
// ---------------------------------------------------------------------------

struct ggml_ocl_buffer_ctx {
    cl_mem mem = nullptr;
    size_t size = 0;
    ocl_extra_pool extras;

    explicit ggml_ocl_buffer_ctx(cl_mem mem_, size_t size_) : mem(mem_), size(size_) {}
    ~ggml_ocl_buffer_ctx() {
        if (mem) {
            clReleaseMemObject(mem);
        }
        extras.free_all();
    }
};

// 记录拷贝事件, graph_compute 开头统一等待 (上限 32 个)
static void ocl_buffer_track_copy_event(ggml_ocl_backend * backend, cl_event evt) {
    backend->pending_copy_events.push_back(evt);
    if (backend->pending_copy_events.size() >= 32) {
        ocl_exec_wait_pending_copies(backend);
    }
}

static ggml_ocl_device_context * ggml_ocl_buffer_dev_ctx(ggml_backend_buffer_t buffer) {
    return (ggml_ocl_device_context *) buffer->buft->device->context;
}

static cl_command_queue ggml_ocl_buffer_copy_queue(ggml_backend_buffer_t buffer) {
    ggml_ocl_device_context * dev_ctx = ggml_ocl_buffer_dev_ctx(buffer);

    if (dev_ctx->backend != nullptr) {
        return dev_ctx->backend->q_copy;
    }

    // 模型加载阶段 backend 还没创建，使用设备级 copy queue
    if (dev_ctx->q_load == nullptr) {
        cl_int err = CL_SUCCESS;
        dev_ctx->q_load = clCreateCommandQueueWithProperties(dev_ctx->context, dev_ctx->device, nullptr, &err);
        OCL_CHECK(err);
    }
    return dev_ctx->q_load;
}

static void ggml_ocl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_ocl_device_context * dev_ctx = ggml_ocl_buffer_dev_ctx(buffer);
    dev_ctx->mem_allocated -= buffer->size;
    delete (ggml_ocl_buffer_ctx *) buffer->context;
}

static void * ggml_ocl_buffer_get_base(ggml_backend_buffer_t buffer) {
    // 必须是 alignment 对齐的假地址: ggml_tallocr_new 用 aligned_offset(base, 0, align) 定起点,
    // 基址不对齐会让每个 buffer 白白少掉 (align - base%align) 字节, 最后一个张量放不下.
    ggml_ocl_device_context * dev_ctx = ggml_ocl_buffer_dev_ctx(buffer);
    return (void *) (uintptr_t) dev_ctx->alignment;
}

static enum ggml_status ggml_ocl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_ocl_buffer_ctx * ctx = (ggml_ocl_buffer_ctx *) buffer->context;

    if (tensor->view_src != nullptr) {
        // view: reuse parent's extra, offset computed at runtime (DESIGN.md 16.3)
        ggml_ocl_tensor_extra * view_extra = (ggml_ocl_tensor_extra *) tensor->view_src->extra;
        GGML_ASSERT(view_extra != nullptr);
        tensor->extra = view_extra;
    } else {
        size_t offset = (size_t) ((char *) tensor->data - (char *) ggml_ocl_buffer_get_base(buffer));
        ggml_ocl_tensor_extra * extra = ctx->extras.alloc();
        extra->offset = offset;
        extra->data_device = ctx->mem;
        tensor->extra = extra;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_ocl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                       const void * data, size_t offset, size_t size) {
    ggml_ocl_device_context * dev_ctx = ggml_ocl_buffer_dev_ctx(buffer);
    ggml_ocl_backend *        backend  = dev_ctx->backend;
    ggml_ocl_tensor_extra *   extra    = (ggml_ocl_tensor_extra *) tensor->extra;
    GGML_ASSERT(extra != nullptr);

    std::vector<ggml_fp16_t> tmp;
    bool                     converted = false;
    if (ocl_f16_packed(tensor)) {
        GGML_ASSERT(offset % sizeof(float) == 0 && size % sizeof(float) == 0);
        const size_t n = size / sizeof(float);
        tmp.resize(n);
        ggml_fp32_to_fp16_row((const float *) data, tmp.data(), n);
        data      = tmp.data();
        size      = n * sizeof(ggml_fp16_t);
        converted = true;
    }

    // Q4_1 weights are repacked into two planes while uploading
    if (ocl_q4_1_repacked(tensor)) {
        GGML_ASSERT(offset % 20 == 0 && size % 20 == 0 && "Q4_1 set_tensor must be block aligned");

        const size_t nblk   = size / 20;
        const size_t b0     = offset / 20;
        const size_t qbytes = ocl_q4_1_quants_bytes(tensor);

        std::vector<char> q(nblk * 16);
        std::vector<char> d(nblk * 4);
        const char *      src = (const char *) data;
        for (size_t k = 0; k < nblk; k++) {
            memcpy(q.data() + k * 16, src + k * 20 + 4, 16);  // 16 quant bytes
            memcpy(d.data() + k * 4, src + k * 20, 4);        // d, m
        }

        const cl_ulong     base = ocl_dev_offset(tensor, extra);
        cl_command_queue   cq   = ggml_ocl_buffer_copy_queue(buffer);
        OCL_CHECK(clEnqueueWriteBuffer(cq, extra->data_device, CL_TRUE, base + b0 * 16, nblk * 16, q.data(), 0,
                                       nullptr, nullptr));
        OCL_CHECK(clEnqueueWriteBuffer(cq, extra->data_device, CL_TRUE, base + qbytes + b0 * 4, nblk * 4, d.data(), 0,
                                       nullptr, nullptr));
        return;
    }

    cl_ulong    eff_offset = ocl_dev_offset(tensor, extra) + ocl_dev_bytes(tensor, offset);
    cl_event    evt        = nullptr;
    cl_event *  evt_ptr    = backend ? &evt : nullptr;
    cl_bool     blocking   = (backend && !converted) ? CL_FALSE : CL_TRUE;

    OCL_CHECK(clEnqueueWriteBuffer(ggml_ocl_buffer_copy_queue(buffer), extra->data_device, blocking,
                                   eff_offset, size, data, 0, nullptr, evt_ptr));
    if (evt_ptr != nullptr) {
        backend->pending_copy_events.push_back(evt);
        if (backend->pending_copy_events.size() >= 32) {
            ocl_exec_wait_pending_copies(backend);
        }
    }
}

static void ggml_ocl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                       void * data, size_t offset, size_t size) {
    ggml_ocl_tensor_extra * extra = (ggml_ocl_tensor_extra *) tensor->extra;
    GGML_ASSERT(extra != nullptr);

    cl_command_queue q          = ggml_ocl_buffer_copy_queue(buffer);
    cl_ulong         eff_offset = ocl_dev_offset(tensor, extra) + ocl_dev_bytes(tensor, offset);
    const size_t     dev_size   = ocl_dev_bytes(tensor, size);

    // undo the two-plane repack when the caller wants the GGUF layout back
    if (ocl_q4_1_repacked(tensor)) {
        GGML_ASSERT(offset % 20 == 0 && size % 20 == 0 && "Q4_1 get_tensor must be block aligned");

        const size_t nblk   = size / 20;
        const size_t b0     = offset / 20;
        const size_t qbytes = ocl_q4_1_quants_bytes(tensor);

        std::vector<char> qb(nblk * 16);
        std::vector<char> db(nblk * 4);
        OCL_CHECK(clEnqueueReadBuffer(q, extra->data_device, CL_TRUE, eff_offset + b0 * 16, nblk * 16, qb.data(), 0,
                                      nullptr, nullptr));
        OCL_CHECK(clEnqueueReadBuffer(q, extra->data_device, CL_TRUE, eff_offset + qbytes + b0 * 4, nblk * 4, db.data(),
                                      0, nullptr, nullptr));

        char * dst = (char *) data;
        for (size_t k = 0; k < nblk; k++) {
            memcpy(dst + k * 20, db.data() + k * 4, 4);
            memcpy(dst + k * 20 + 4, qb.data() + k * 16, 16);
        }
        return;
    }

    if (ocl_f16_packed(tensor)) {
        GGML_ASSERT(size % sizeof(float) == 0);
        std::vector<ggml_fp16_t> tmp(dev_size / sizeof(ggml_fp16_t));
        OCL_CHECK(clEnqueueReadBuffer(q, extra->data_device, CL_TRUE,
                                      eff_offset, dev_size, tmp.data(), 0, nullptr, nullptr));
        if (!tmp.empty()) {
            ggml_fp16_to_fp32_row(tmp.data(), (float *) data, tmp.size());
        }
    } else {
        OCL_CHECK(clEnqueueReadBuffer(q, extra->data_device, CL_TRUE,
                                      eff_offset, size, data, 0, nullptr, nullptr));
    }
}

static void ggml_ocl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                          uint8_t value, size_t offset, size_t size) {
    ggml_ocl_device_context * dev_ctx = ggml_ocl_buffer_dev_ctx(buffer);
    ggml_ocl_tensor_extra *   extra   = (ggml_ocl_tensor_extra *) tensor->extra;
    GGML_ASSERT(extra != nullptr);

    cl_command_queue q = ggml_ocl_buffer_copy_queue(buffer);
    cl_ulong eff_offset = ocl_dev_offset(tensor, extra) + ocl_dev_bytes(tensor, offset);
    size_t   dev_size   = ocl_dev_bytes(tensor, size);

    if (ocl_f16_packed(tensor) && value != 0) {
        GGML_ABORT("ggml-ocl: memset_tensor with value %u on fp16-stored F32 tensor", value);
    }

    cl_event evt = nullptr;
    OCL_CHECK(clEnqueueFillBuffer(q, extra->data_device, &value, sizeof(value),
                                  eff_offset, dev_size, 0, nullptr, dev_ctx->backend ? &evt : nullptr));
    if (dev_ctx->backend != nullptr) {
        ocl_buffer_track_copy_event(dev_ctx->backend, evt);
    } else {
        OCL_CHECK(clFinish(q));
    }
}

static bool ggml_ocl_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_ocl_device_context * dev_ctx = ggml_ocl_buffer_dev_ctx(buffer);
    if (src->buffer->buft->device == nullptr || src->buffer->buft->device->context != dev_ctx) {
        return false;
    }

    ggml_ocl_tensor_extra * src_extra = (ggml_ocl_tensor_extra *) src->extra;
    ggml_ocl_tensor_extra * dst_extra = (ggml_ocl_tensor_extra *) dst->extra;
    GGML_ASSERT(src_extra != nullptr && dst_extra != nullptr);
    GGML_ASSERT(ocl_f16_packed(src) == ocl_f16_packed(dst) && "mixed storage in cpy_tensor");

    cl_command_queue q       = ggml_ocl_buffer_copy_queue(buffer);
    cl_ulong         src_off = ocl_dev_offset(src, src_extra);
    cl_ulong         dst_off = ocl_dev_offset(dst, dst_extra);
    cl_event evt = nullptr;
    OCL_CHECK(clEnqueueCopyBuffer(q, src_extra->data_device, dst_extra->data_device,
                                  src_off, dst_off, ocl_dev_bytes(src, ggml_nbytes(src)), 0, nullptr,
                                  dev_ctx->backend ? &evt : nullptr));
    if (dev_ctx->backend != nullptr) {
        ocl_buffer_track_copy_event(dev_ctx->backend, evt);
    } else {
        OCL_CHECK(clFinish(q));
    }
    return true;
}

static void ggml_ocl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_ocl_buffer_ctx * ctx = (ggml_ocl_buffer_ctx *) buffer->context;
    cl_command_queue          q       = ggml_ocl_buffer_copy_queue(buffer);
    OCL_CHECK(clEnqueueFillBuffer(q, ctx->mem, &value, sizeof(value),
                                  0, ctx->size, 0, nullptr, nullptr));
    // 整块清零只在初始化/重置时发生, 直接等完, 避免与第一次计算竞争
    OCL_CHECK(clFinish(q));
}

static void ggml_ocl_buffer_reset(ggml_backend_buffer_t buffer) {
    // 不能清空 extra 池: ggml-alloc 每次 compute 都会调用 reset, 但只对 data==NULL 的张量
    // 重新 init_tensor, 已分配张量的 extra 若被清掉就成为悬空/复用状态, 读到的 cl_mem 和
    // offset 属于别的张量 (表现为个别层突然算出几万的垃圾值).
    // extra 里的 offset 在同一 buffer 内是不变的, 保持不动即可.
    GGML_UNUSED(buffer);
}

static ggml_backend_buffer_i ggml_ocl_buffer_interface = {
    /* .free_buffer     = */ ggml_ocl_buffer_free_buffer,
    /* .get_base        = */ ggml_ocl_buffer_get_base,
    /* .init_tensor     = */ ggml_ocl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_ocl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_ocl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_ocl_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_ocl_buffer_cpy_tensor,
    /* .clear           = */ ggml_ocl_buffer_clear,
    /* .reset           = */ ggml_ocl_buffer_reset,
};

// ---------------------------------------------------------------------------
// buffer type (DESIGN.md 16.2)
// ---------------------------------------------------------------------------

const char * ggml_ocl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "OCL";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_ocl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) buft->device->context;
    GGML_ASSERT(dev_ctx->context != nullptr);

    // clCreateBuffer returns -61 for size 0
    size = MAX(size, (size_t) 1);

    // size > max_alloc 由 ggml-alloc 自动拆分为 multi-buffer (ggml-alloc.c),
    // 此处单次分配恒 <= caps.max_alloc
    cl_int err;
    cl_mem mem = clCreateBuffer(dev_ctx->context, CL_MEM_READ_WRITE, size, nullptr, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_INFO("ggml-ocl: failed to allocate %.2f MiB\n", size / 1024.0 / 1024.0);
        return nullptr;
    }

    GGML_LOG_DEBUG("ggml-ocl: allocated buffer of %.2f MiB (total %.2f MiB)\n",
                   size / 1024.0 / 1024.0,
                   (dev_ctx->mem_allocated + size) / 1024.0 / 1024.0);
    dev_ctx->mem_allocated += size;

    ggml_ocl_buffer_ctx * ctx = new ggml_ocl_buffer_ctx(mem, size);
    return ggml_backend_buffer_init(buft, ggml_ocl_buffer_interface, ctx, size);
}

static size_t ggml_ocl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) buft->device->context;
    return dev_ctx->alignment;
}

static size_t ggml_ocl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) buft->device->context;
    return dev_ctx->max_alloc;
}

static size_t ggml_ocl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

struct ggml_backend_buffer_type_i ggml_ocl_buffer_type_interface = {
    /* .get_name         = */ ggml_ocl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_ocl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_ocl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_ocl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_ocl_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};
