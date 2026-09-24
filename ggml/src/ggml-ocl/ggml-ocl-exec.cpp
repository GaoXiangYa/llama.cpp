#include "ggml-ocl-internal.h"


void ocl_kernel_call::enqueue(ggml_ocl_backend * backend, cl_event * evt_out) {
    GGML_ASSERT(kernel != nullptr);

    for (int i = 0; i < arg_idx; i++) {
        OCL_CHECK(clSetKernelArg(kernel, i, arg_sizes[i], &args[i]));
    }

#ifdef GGML_OCL_PROFILING
    if (backend->profiling_enabled) {
        cl_event evt = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(backend->q_compute, kernel, ndims, nullptr, global, local, 0, nullptr, &evt));
        OCL_CHECK(clWaitForEvents(1, &evt));
        cl_ulong start = 0;
        cl_ulong end = 0;
        OCL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
        OCL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
        clReleaseEvent(evt);

        char   kname[128] = { 0 };
        size_t kname_len  = 0;
        if (clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kname) - 1, kname, &kname_len) != CL_SUCCESS ||
            kname[0] == 0) {
            snprintf(kname, sizeof(kname), "?");
        }
        backend->stats.record(op_name ? op_name : "?", kname, (cl_ulong) (end - start));
        if (evt_out) {
            *evt_out = nullptr;
        }
        return;
    }
#endif
    OCL_CHECK(clEnqueueNDRangeKernel(backend->q_compute, kernel, ndims, nullptr, global, local, 0, nullptr, evt_out));
}

void ocl_exec_wait_pending_copies(ggml_ocl_backend * backend) {
    if (!backend->pending_copy_events.empty()) {
        OCL_CHECK(clWaitForEvents(backend->pending_copy_events.size(), backend->pending_copy_events.data()));
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
        GGML_LOG_INFO("  %-24s %-20s %6d calls %10.2f ms total %8.2f us/call\n", s.op_name.c_str(),
                      s.kernel_name.c_str(), s.count, s.total_ns / 1e6, s.total_ns / 1e3 / MAX(s.count, 1));
    }
}
#endif  // GGML_OCL_PROFILING
