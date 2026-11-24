#include "ggml-ocl-internal.h"

#include <algorithm>


void ocl_kernel_call::enqueue(ggml_ocl_backend * backend, cl_event * evt_out) {
    GGML_ASSERT(kernel != nullptr);

    for (int i = 0; i < arg_idx; i++) {
        OCL_CHECK(clSetKernelArg(kernel, i, arg_sizes[i], arg_is_local[i] ? nullptr : (const void *) &args[i]));
    }

#ifdef GGML_OCL_PROFILING
    if (backend->profiling_enabled) {
        cl_event evt = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(backend->q_compute, kernel, ndims, nullptr, global, local, 0, nullptr, &evt));
        backend->pending_prof.push_back({ evt, kernel, op_name });
        backend->stats.launches++;
        // safety cap, a normal graph stays far below this
        if (backend->pending_prof.size() >= 8192) {
            ocl_prof_flush(backend);
        }
        if (evt_out) {
            *evt_out = nullptr;
        }
        return;
    }
#endif
    OCL_CHECK(clEnqueueNDRangeKernel(backend->q_compute, kernel, ndims, nullptr, global, local, 0, nullptr, evt_out));
}

void ocl_pack_binary_args(ocl_kernel_call &  call,
                          const ggml_tensor * s0,
                          const ggml_tensor * s1,
                          const ggml_tensor * dst,
                          bool                strides_u64) {
    ggml_ocl_tensor_extra * e0 = (ggml_ocl_tensor_extra *) s0->extra;
    ggml_ocl_tensor_extra * e1 = (ggml_ocl_tensor_extra *) s1->extra;
    ggml_ocl_tensor_extra * ed = (ggml_ocl_tensor_extra *) dst->extra;
    GGML_ASSERT(e0 != nullptr && e1 != nullptr && ed != nullptr);

    auto nb = [&](const ggml_tensor * t, size_t v) {
        if (strides_u64) {
            call.arg_u64(ocl_nb64(t, v));
        } else {
            call.arg_i32(ocl_nb(t, v));
        }
    };

    call.arg_cl_mem(e0->data_device);
    call.arg_u64(ocl_dev_offset(s0, e0));
    call.arg_cl_mem(e1->data_device);
    call.arg_u64(ocl_dev_offset(s1, e1));
    call.arg_cl_mem(ed->data_device);
    call.arg_u64(ocl_dev_offset(dst, ed));

    call.arg_i32((cl_int) s0->ne[0]);
    call.arg_i32((cl_int) s0->ne[1]);
    call.arg_i32((cl_int) s0->ne[2]);
    call.arg_i32((cl_int) s0->ne[3]);
    nb(s0, s0->nb[0]);
    nb(s0, s0->nb[1]);
    nb(s0, s0->nb[2]);
    nb(s0, s0->nb[3]);

    call.arg_i32((cl_int) s1->ne[0]);
    call.arg_i32((cl_int) s1->ne[1]);
    call.arg_i32((cl_int) s1->ne[2]);
    call.arg_i32((cl_int) s1->ne[3]);
    nb(s1, s1->nb[0]);
    nb(s1, s1->nb[1]);
    nb(s1, s1->nb[2]);
    nb(s1, s1->nb[3]);

    call.arg_i32((cl_int) dst->ne[0]);
    call.arg_i32((cl_int) dst->ne[1]);
    call.arg_i32((cl_int) dst->ne[2]);
    call.arg_i32((cl_int) dst->ne[3]);
    nb(dst, dst->nb[0]);
    nb(dst, dst->nb[1]);
    nb(dst, dst->nb[2]);
    nb(dst, dst->nb[3]);
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
const char * ocl_stats::name_for(cl_kernel k) {
    for (const auto & e : kernel_names) {
        if (e.k == k) {
            return e.name.c_str();
        }
    }
    char   kname[128] = { 0 };
    size_t kname_len  = 0;
    if (clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(kname) - 1, kname, &kname_len) != CL_SUCCESS || kname[0] == 0) {
        snprintf(kname, sizeof(kname), "?");
    }
    kernel_names.push_back({ k, kname });
    return kernel_names.back().name.c_str();
}

void ocl_stats::record(const char * op, cl_kernel k, cl_ulong ns) {
    const char * kernel = name_for(k);
    for (auto & s : kernels) {
        if (s.op_name == op && s.kernel_name == kernel) {
            s.count++;
            s.total_ns += ns;
            return;
        }
    }
    kernels.push_back({ op, kernel, 1, ns });
}

void ocl_prof_flush(ggml_ocl_backend * backend) {
    std::vector<ocl_prof_entry> & pend = backend->pending_prof;
    if (pend.empty()) {
        return;
    }

    std::vector<cl_event> evts;
    evts.reserve(pend.size());
    for (const auto & e : pend) {
        evts.push_back(e.evt);
    }
    OCL_CHECK(clWaitForEvents((cl_uint) evts.size(), evts.data()));

    cl_ulong span_start = ~0ull;
    cl_ulong span_end   = 0;
    for (const auto & e : pend) {
        cl_ulong start = 0;
        cl_ulong end   = 0;
        OCL_CHECK(clGetEventProfilingInfo(e.evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
        OCL_CHECK(clGetEventProfilingInfo(e.evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
        backend->stats.record(e.op_name ? e.op_name : "?", e.kernel, end - start);
        span_start = start < span_start ? start : span_start;
        span_end   = end   > span_end   ? end   : span_end;
        clReleaseEvent(e.evt);
    }

    backend->stats.graph_gpu_ns += span_end - span_start;
    backend->stats.graphs++;
    pend.clear();
}

void ocl_stats::print() const {
    GGML_LOG_INFO("ggml-ocl: profiling summary (%zu kernels, %d graphs, %d launches):\n", kernels.size(), graphs,
                  launches);

    std::vector<const ocl_kernel_stat *> sorted;
    sorted.reserve(kernels.size());
    for (const auto & s : kernels) {
        sorted.push_back(&s);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const ocl_kernel_stat * a, const ocl_kernel_stat * b) { return a->total_ns > b->total_ns; });

    for (const auto * s : sorted) {
        GGML_LOG_INFO("  %-24s %-20s %6d calls %10.2f ms total %8.2f us/call\n", s->op_name.c_str(),
                      s->kernel_name.c_str(), s->count, s->total_ns / 1e6, s->total_ns / 1e3 / MAX(s->count, 1));
    }

    const double n = MAX(graphs, 1);
    GGML_LOG_INFO("ggml-ocl: per graph: launches %.1f, host enqueue %.3f ms, gpu busy %.3f ms\n",
                  (double) launches / n, host_ns / 1e6 / n, graph_gpu_ns / 1e6 / n);
}
#endif  // GGML_OCL_PROFILING
