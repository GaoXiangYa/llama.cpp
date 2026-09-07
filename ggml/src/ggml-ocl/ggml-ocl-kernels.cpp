#include "ggml-ocl-internal.h"

#include <cstring>

enum ocl_kernel_class { OCL_KERNEL_GENERIC, OCL_KERNEL_PERF };

struct ocl_kernel_def {
    const char *     src_id;
    ocl_kernel_class cls;
};

static const ocl_kernel_def g_kernel_defs[] = {
    { "add/add",           OCL_KERNEL_GENERIC },
    { "set_rows/set_rows", OCL_KERNEL_GENERIC },
    { "get_rows/get_rows", OCL_KERNEL_GENERIC },
    { "mul_mat/gemv",       OCL_KERNEL_GENERIC },
    { "mul_mat/gemv_q4_1", OCL_KERNEL_GENERIC },
    { "mul_mat/gemm",       OCL_KERNEL_GENERIC },
};

static const std::string & ocl_kernel_source(const char * src_id) {
    if (strcmp(src_id, "add/add") == 0) {
        static const std::string src{
#include "add/add.cl.h"
        };
        return src;
    }
    if (strcmp(src_id, "set_rows/set_rows") == 0) {
        static const std::string src{
#include "set_rows/set_rows.cl.h"
        };
        return src;
    }
    if (strcmp(src_id, "get_rows/get_rows") == 0) {
        static const std::string src{
#include "get_rows/get_rows.cl.h"
        };
        return src;
    }
    if (strcmp(src_id, "mul_mat/gemv") == 0) {
        static const std::string src{
#include "mul_mat/gemv.cl.h"
        };
        return src;
    }
    if (strcmp(src_id, "mul_mat/gemv_q4_1") == 0) {
        static const std::string src{
#include "mul_mat/gemv_q4_1.cl.h"
        };
        return src;
    }
    if (strcmp(src_id, "mul_mat/gemm") == 0) {
        static const std::string src{
#include "mul_mat/gemm.cl.h"
        };
        return src;
    }
    GGML_ABORT("ggml-ocl: unknown kernel source '%s'", src_id);
}

static std::string ocl_kernel_compile_opts(const ggml_ocl_caps * caps, ocl_kernel_class cls) {
    std::string opts = std::string("-cl-std=CL") + std::to_string(caps->ocl_c_major) + "." +
                       std::to_string(caps->ocl_c_minor) + " -cl-mad-enable";
    if (cls == OCL_KERNEL_PERF) {
        opts += " -cl-fast-relaxed-math";
    }
    return opts;
}

// ---------------------------------------------------------------------------
// ocl_kernel_mgr
// ---------------------------------------------------------------------------

void ocl_kernel_mgr::compile_all(ggml_ocl_backend * backend) {
    std::lock_guard<std::mutex> lock(m_);
    if (!entries_.empty()) {
        return;  // 幂等: 进程内只编译一次
    }

    const ggml_ocl_caps * caps = &backend->caps;
    const int64_t         t0   = ggml_time_us();
    int                   n_ok = 0;

    for (const ocl_kernel_def & def : g_kernel_defs) {
        ocl_kernel_entry entry;
        entry.src_id       = def.src_id;
        entry.compile_opts = ocl_kernel_compile_opts(caps, def.cls);

        const std::string & src      = ocl_kernel_source(def.src_id);
        const char *        src_cstr = src.c_str();
        const size_t        src_len  = src.size();

        cl_int     err;
        cl_program program = clCreateProgramWithSource(backend->context, 1, &src_cstr, &src_len, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("ggml-ocl: create program %s failed (%d)\n", def.src_id, err);
            entry.state          = KS_FAILED;
            entries_[def.src_id] = std::move(entry);
            continue;
        }

        err = clBuildProgram(program, 1, &caps->device, entry.compile_opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char   log[4096] = { 0 };
            size_t log_size  = 0;
            clGetProgramBuildInfo(program, caps->device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, &log_size);
            GGML_LOG_ERROR("ggml-ocl: build %s failed (%d):\n%s\n", def.src_id, err, log);
            clReleaseProgram(program);
            entry.state          = KS_FAILED;
            entries_[def.src_id] = std::move(entry);
            continue;
        }

        cl_uint n_kernels = 0;
        err               = clCreateKernelsInProgram(program, 0, nullptr, &n_kernels);
        if (err != CL_SUCCESS || n_kernels == 0) {
            GGML_LOG_ERROR("ggml-ocl: no kernels found in %s (%d)\n", def.src_id, err);
            clReleaseProgram(program);
            entry.state          = KS_FAILED;
            entries_[def.src_id] = std::move(entry);
            continue;
        }

        entry.kernels.resize(n_kernels);
        clCreateKernelsInProgram(program, n_kernels, entry.kernels.data(), nullptr);
        for (cl_uint i = 0; i < n_kernels; i++) {
            char fn_name[128] = { 0 };
            clGetKernelInfo(entry.kernels[i], CL_KERNEL_FUNCTION_NAME, sizeof(fn_name) - 1, fn_name, nullptr);
            entry.kernel_index[fn_name] = (int) i;
        }

        entry.program        = program;
        entry.state          = KS_OK;
        entries_[def.src_id] = std::move(entry);
        n_ok++;
    }

    const int n_total = sizeof(g_kernel_defs) / sizeof(g_kernel_defs[0]);
    GGML_LOG_INFO("ggml-ocl: loaded %d/%d kernel sources in %.2f s\n", n_ok, n_total, (ggml_time_us() - t0) / 1e6);
}

cl_kernel ocl_kernel_mgr::get(const char * src_id, const char * fn_name) {
    auto it = entries_.find(src_id);
    if (it == entries_.end() || it->second.state != KS_OK) {
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

bool ocl_kernel_mgr::is_ready(const char * src_id) const {
    auto it = entries_.find(src_id);
    return it != entries_.end() && it->second.state == KS_OK;
}
