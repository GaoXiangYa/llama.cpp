// ggml-ocl backend: execution infrastructure (DESIGN.md section 17)
// S7 范围: ocl_kernel_call 封装 + 队列工具 + profiling;
// graph_compute 完整流程在 S8 与 ops 注册表一起落地.

#include "ggml-ocl-internal.h"

// ---------------------------------------------------------------------------
// ocl_kernel_call (DESIGN.md 17.1)
// ---------------------------------------------------------------------------

void ocl_kernel_call::enqueue(ggml_ocl_backend * backend, cl_event * evt_out) {
    GGML_ASSERT(kernel != nullptr);

    for (int i = 0; i < arg_idx; i++) {
        OCL_CHECK(clSetKernelArg(kernel, i, arg_sizes[i], &args[i]));
    }

#ifdef GGML_OCL_PROFILING
    cl_event evt = nullptr;
    OCL_CHECK(clEnqueueNDRangeKernel(backend->q_compute, kernel, ndims, nullptr,
                                     global, local, 0, nullptr, &evt));
    // profiling 模式: 同步结算, 统计 (op 名, kernel 名, 耗时)
    OCL_CHECK(clWaitForEvents(1, &evt));
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(end),   &end,   nullptr);
    clReleaseEvent(evt);
    backend->stats.record(op_name ? op_name : "?", "?", (cl_ulong)(end - start));
    if (evt_out) {
        *evt_out = nullptr;
    }
#else
    OCL_CHECK(clEnqueueNDRangeKernel(backend->q_compute, kernel, ndims, nullptr,
                                     global, local, 0, nullptr, evt_out));
#endif
}

// ---------------------------------------------------------------------------
// 队列工具 (DESIGN.md 17.3)
// ---------------------------------------------------------------------------

void ocl_exec_wait_pending_copies(ggml_ocl_backend * backend) {
    if (!backend->pending_copy_events.empty()) {
        OCL_CHECK(clWaitForEvents(backend->pending_copy_events.size(),
                                  backend->pending_copy_events.data()));
        for (cl_event evt : backend->pending_copy_events) {
            clReleaseEvent(evt);
        }
        backend->pending_copy_events.clear();
    }
}

void ocl_exec_flush(ggml_ocl_backend * backend) {
    OCL_CHECK(clFlush(backend->q_compute));
}

// ---------------------------------------------------------------------------
// profiling (DESIGN.md 17.4)
// ---------------------------------------------------------------------------

#ifdef GGML_OCL_PROFILING
void ocl_stats::record(const char * op, const char * kernel, cl_ulong ns) {
    for (auto & s : kernels) {
        if (s.op_name == op && s.kernel_name == kernel) {
            s.count++;
            s.total_ns += ns;
            return;
        }
    }
    kernels.push_back({ op, kernel, 1, ns });
}

void ocl_stats::print() const {
    GGML_LOG_INFO("ggml-ocl: profiling summary (%zu kernels):\n", kernels.size());
    for (const auto & s : kernels) {
        GGML_LOG_INFO("  %-24s %-20s %6d calls %10.2f ms total %8.2f us/call\n",
                      s.op_name.c_str(), s.kernel_name.c_str(), s.count,
                      s.total_ns / 1e6, s.total_ns / 1e3 / MAX(s.count, 1));
    }
}
#endif // GGML_OCL_PROFILING
