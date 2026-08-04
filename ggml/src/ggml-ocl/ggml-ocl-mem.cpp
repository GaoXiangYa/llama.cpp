// ggml-ocl backend: memory management (DESIGN.md section 16)
// M0 scope: pools + buffer interface (AoS direct). SoA-DM layout conversion,
// subbuffer 池细化, 显存记账 arrive in S6.

#include "ggml-ocl.h"
#include "ggml-ocl-internal.h"

// ---------------------------------------------------------------------------
// pools (DESIGN.md 16.1)
// ---------------------------------------------------------------------------

void ocl_pool::init(cl_context ctx) {
    context_ = ctx;
}

cl_mem ocl_pool::alloc(size_t size) {
    std::lock_guard<std::mutex> lock(m_);
    size_t bucket = 1;
    while (bucket < size) {
        bucket <<= 1;
    }

    auto it = free_.find(bucket);
    if (it != free_.end() && !it->second.empty()) {
        cl_mem mem = it->second.back();
        it->second.pop_back();
        return mem;
    }

    cl_int err;
    cl_mem mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, bucket, nullptr, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-ocl: pool alloc %zu B failed (%d)\n", bucket, err);
        return nullptr;
    }
    reserved_ += bucket;
    high_water_ = MAX(high_water_, reserved_);
    return mem;
}

void ocl_pool::release(cl_mem mem, size_t size) {
    std::lock_guard<std::mutex> lock(m_);
    size_t bucket = 1;
    while (bucket < size) {
        bucket <<= 1;
    }
    free_[bucket].push_back(mem);
}

void ocl_pool::trim() {
    std::lock_guard<std::mutex> lock(m_);
    if (reserved_ <= high_water_ / 2) {
        return;
    }
    // release the oldest (front) entries of every bucket
    for (auto & kv : free_) {
        for (cl_mem mem : kv.second) {
            clReleaseMemObject(mem);
            reserved_ -= kv.first;
        }
        kv.second.clear();
    }
}

void ocl_pool::clear() {
    std::lock_guard<std::mutex> lock(m_);
    for (auto & kv : free_) {
        for (cl_mem mem : kv.second) {
            clReleaseMemObject(mem);
        }
        kv.second.clear();
    }
    reserved_ = 0;
    high_water_ = 0;
}

cl_mem ocl_subpool::alloc(cl_mem parent, size_t origin, size_t size) {
    std::lock_guard<std::mutex> lock(m_);
    auto key = std::make_pair(parent, size);
    auto it = free_.find(key);
    if (it != free_.end() && !it->second.empty()) {
        cl_mem sub = it->second.back();
        it->second.pop_back();
        return sub;
    }

    cl_int err;
    cl_buffer_region region = { origin, size };
    cl_mem sub = clCreateSubBuffer(parent, CL_MEM_READ_WRITE,
                                   CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-ocl: subbuffer alloc %zu B failed (%d)\n", size, err);
        return nullptr;
    }
    return sub;
}

void ocl_subpool::release(cl_mem sub, cl_mem parent, size_t size) {
    std::lock_guard<std::mutex> lock(m_);
    free_[std::make_pair(parent, size)].push_back(sub);
}

void ocl_subpool::clear() {
    std::lock_guard<std::mutex> lock(m_);
    for (auto & kv : free_) {
        for (cl_mem sub : kv.second) {
            clReleaseMemObject(sub);
        }
    }
    free_.clear();
}

ggml_ocl_tensor_extra * ocl_extra_pool::alloc() {
    if (!free_.empty()) {
        ggml_ocl_tensor_extra * extra = free_.back();
        free_.pop_back();
        extra->reset();
        return extra;
    }
    return new ggml_ocl_tensor_extra();
}

void ocl_extra_pool::free_all() {
    for (ggml_ocl_tensor_extra * extra : free_) {
        delete extra;
    }
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

static ggml_ocl_backend * ggml_ocl_buffer_backend(ggml_backend_buffer_t buffer) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) buffer->buft->device->context;
    return dev_ctx->backend;
}

static void ggml_ocl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_ocl_buffer_ctx *) buffer->context;
}

static void * ggml_ocl_buffer_get_base(ggml_backend_buffer_t buffer) {
    // fake base for offset arithmetic; tensor->data = base + offset
    ggml_ocl_backend * backend = ggml_ocl_buffer_backend(buffer);
    return (void *) (uintptr_t) backend->caps.alignment;
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
        extra->aux_device = nullptr;
        extra->layout = LAYOUT_AOS;
        tensor->extra = extra;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_ocl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                       const void * data, size_t offset, size_t size) {
    ggml_ocl_backend * backend = ggml_ocl_buffer_backend(buffer);
    ggml_ocl_tensor_extra * extra = (ggml_ocl_tensor_extra *) tensor->extra;
    GGML_ASSERT(extra != nullptr);
    GGML_ASSERT(extra->layout == LAYOUT_AOS && "SoA-DM arrives in S6");

    cl_ulong eff_offset = extra->offset + tensor->view_offs + offset;
    OCL_CHECK(clEnqueueWriteBuffer(backend->q_copy, extra->data_device, CL_FALSE,
                                   eff_offset, size, data, 0, nullptr, nullptr));
}

static void ggml_ocl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                       void * data, size_t offset, size_t size) {
    ggml_ocl_backend * backend = ggml_ocl_buffer_backend(buffer);
    ggml_ocl_tensor_extra * extra = (ggml_ocl_tensor_extra *) tensor->extra;
    GGML_ASSERT(extra != nullptr);

    cl_ulong eff_offset = extra->offset + tensor->view_offs + offset;
    OCL_CHECK(clEnqueueReadBuffer(backend->q_copy, extra->data_device, CL_TRUE,
                                  eff_offset, size, data, 0, nullptr, nullptr));
}

static void ggml_ocl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_ocl_backend * backend = ggml_ocl_buffer_backend(buffer);
    ggml_ocl_buffer_ctx * ctx = (ggml_ocl_buffer_ctx *) buffer->context;
    OCL_CHECK(clEnqueueFillBuffer(backend->q_copy, ctx->mem, &value, sizeof(value),
                                  0, ctx->size, 0, nullptr, nullptr));
}

static void ggml_ocl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
}

static ggml_backend_buffer_i ggml_ocl_buffer_interface = {
    /* .free_buffer     = */ ggml_ocl_buffer_free_buffer,
    /* .get_base        = */ ggml_ocl_buffer_get_base,
    /* .init_tensor     = */ ggml_ocl_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_ocl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_ocl_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
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
    ggml_ocl_backend * backend = dev_ctx->backend;
    GGML_ASSERT(backend != nullptr);

    // clCreateBuffer returns -61 for size 0
    size = MAX(size, (size_t) 1);

    cl_int err;
    cl_mem mem = clCreateBuffer(backend->context, CL_MEM_READ_WRITE, size, nullptr, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_INFO("ggml-ocl: failed to allocate %.2f MiB\n", size / 1024.0 / 1024.0);
        return nullptr;
    }

    ggml_ocl_buffer_ctx * ctx = new ggml_ocl_buffer_ctx(mem, size);
    return ggml_backend_buffer_init(buft, ggml_ocl_buffer_interface, ctx, size);
}

static size_t ggml_ocl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) buft->device->context;
    return dev_ctx->backend->caps.alignment;
}

static size_t ggml_ocl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_ocl_device_context * dev_ctx = (ggml_ocl_device_context *) buft->device->context;
    return dev_ctx->backend->caps.max_alloc;
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
