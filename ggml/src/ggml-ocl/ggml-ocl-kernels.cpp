#include "ggml-ocl-internal.h"

#include <cstring>

struct ocl_kernel_src {
    const char * id;
    const char * src;
};

// 由 CMake 根据 GGML_OCL_KERNELS 生成 (kernels/embed_kernel.py)
#include "ocl_kernels.inc"

static std::string ocl_kernel_compile_opts(const ggml_ocl_caps * caps) {
    return std::string("-cl-std=CL") + std::to_string(caps->ocl_c_major) + "." +
           std::to_string(caps->ocl_c_minor) + " -cl-mad-enable";
}

// ---------------------------------------------------------------------------
// ocl_kernel_mgr
// ---------------------------------------------------------------------------

void ocl_kernel_mgr::compile_all(ggml_ocl_backend * backend) {
    std::lock_guard<std::mutex> lock(m_);
    if (!entries_.empty()) {
        return;
    }

    const ggml_ocl_caps * caps = &backend->caps;
    const int64_t         t0   = ggml_time_us();
    int                   n_ok = 0;

    for (const ocl_kernel_src & def : g_kernel_srcs) {
        ocl_kernel_entry entry;
        entry.src_id       = def.id;
        entry.compile_opts = ocl_kernel_compile_opts(caps);

        const char * src_cstr = def.src;
        const size_t src_len  = strlen(def.src);

        cl_int     err;
        cl_program program = clCreateProgramWithSource(backend->context, 1, &src_cstr, &src_len, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("ggml-ocl: create program %s failed (%d)\n", def.id, err);
            entry.state          = KS_FAILED;
            entries_[def.id] = std::move(entry);
            continue;
        }

        err = clBuildProgram(program, 1, &caps->device, entry.compile_opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char   log[4096] = { 0 };
            size_t log_size  = 0;
            clGetProgramBuildInfo(program, caps->device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, &log_size);
            GGML_LOG_ERROR("ggml-ocl: build %s failed (%d):\n%s\n", def.id, err, log);
            clReleaseProgram(program);
            entry.state          = KS_FAILED;
            entries_[def.id] = std::move(entry);
            continue;
        }

        cl_uint n_kernels = 0;
        err               = clCreateKernelsInProgram(program, 0, nullptr, &n_kernels);
        if (err != CL_SUCCESS || n_kernels == 0) {
            GGML_LOG_ERROR("ggml-ocl: no kernels found in %s (%d)\n", def.id, err);
            clReleaseProgram(program);
            entry.state          = KS_FAILED;
            entries_[def.id] = std::move(entry);
            continue;
        }

        entry.kernels.resize(n_kernels);
        clCreateKernelsInProgram(program, n_kernels, entry.kernels.data(), nullptr);
        for (cl_uint i = 0; i < n_kernels; i++) {
            char fn_name[128] = { 0 };
            clGetKernelInfo(entry.kernels[i], CL_KERNEL_FUNCTION_NAME, sizeof(fn_name) - 1, fn_name, nullptr);
            entry.kernel_index[fn_name] = (int) i;

            cl_ulong private_mem = 0;
            size_t   wg_limit    = 0;
            clGetKernelWorkGroupInfo(entry.kernels[i], caps->device, CL_KERNEL_PRIVATE_MEM_SIZE,
                                     sizeof(private_mem), &private_mem, nullptr);
            clGetKernelWorkGroupInfo(entry.kernels[i], caps->device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(wg_limit), &wg_limit, nullptr);
            GGML_LOG_INFO("ggml-ocl: kernel %-28s private_mem %6llu B  wg_limit %zu\n",
                          fn_name, (unsigned long long) private_mem, wg_limit);
        }

        entry.program        = program;
        entry.state          = KS_OK;
        entries_[def.id] = std::move(entry);
        n_ok++;
    }

    const int n_total = sizeof(g_kernel_srcs) / sizeof(g_kernel_srcs[0]);
    GGML_LOG_INFO("ggml-ocl: loaded %d/%d kernel sources in %.2f s\n", n_ok, n_total, (ggml_time_us() - t0) / 1e6);
}

cl_kernel ocl_kernel_mgr::get(const char * src_id, const char * fn_name) {
    auto it = entries_.find(src_id);
    if (it == entries_.end()) {
        GGML_LOG_ERROR("ggml-ocl: kernel source '%s' was never compiled (missing from GGML_OCL_KERNELS?)\n", src_id);
        return nullptr;
    }
    if (it->second.state != KS_OK) {
        GGML_LOG_ERROR("ggml-ocl: kernel source '%s' failed to build\n", src_id);
        return nullptr;
    }
    const ocl_kernel_entry & entry = it->second;
    auto                     fit   = entry.kernel_index.find(fn_name);
    if (fit == entry.kernel_index.end()) {
        GGML_LOG_ERROR("ggml-ocl: kernel '%s' not found in source '%s'\n", fn_name, src_id);
        return nullptr;
    }
    return entry.kernels[fit->second];
}
