#pragma once
// ggml-ocl backend: internal shared definitions (DESIGN.md section 12)

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"

#define CL_TARGET_OPENCL_VERSION GGML_OCL_TARGET_VERSION
#include <CL/cl.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// forward declarations
// ---------------------------------------------------------------------------

struct ggml_ocl_backend;

// ---------------------------------------------------------------------------
// kernel manager (DESIGN.md section 15)
// ---------------------------------------------------------------------------

enum ocl_kernel_state { KS_OK, KS_FAILED };

struct ocl_kernel_entry {
    const char *               src_id = nullptr;  // 源名, 如 "gemv_q4_1"
    std::string                compile_opts;      // 按类别的编译选项
    ocl_kernel_state           state   = KS_OK;
    cl_program                 program = nullptr;
    std::vector<cl_kernel>     kernels;       // 同一源内的多个 kernel 函数
    std::map<std::string, int> kernel_index;  // fn 名 -> kernels 下标
};

class ocl_kernel_mgr {
    std::mutex                              m_;  // 仅启动期需要; 推理期单线程读
    std::map<std::string, ocl_kernel_entry> entries_;
  public:
    // 启动期: 编译全部 kernel (backend 创建时调用一次, 幂等)
    void      compile_all(ggml_ocl_backend * backend);
    // 推理期: 纯查表, 零编译; 返回 nullptr 表示该源/函数不可用
    cl_kernel get(const char * src_id, const char * fn_name);
};

// ---------------------------------------------------------------------------
// device context (DESIGN.md section 11.2; created in ggml-ocl.cpp)
// ---------------------------------------------------------------------------

struct ggml_ocl_device_context {
    cl_device_id             device          = nullptr;
    std::string              device_name;
    size_t                   global_mem_size = 0;
    cl_context               context         = nullptr;  // shared per platform
    ggml_ocl_backend *       backend         = nullptr;  // set on init_backend
    ggml_backend_buffer_type buffer_type     = {};

    // model load 阶段 (init_backend 之前) 也要能分配权重 buffer / 上传 tensor
    cl_command_queue         q_load          = nullptr;
    size_t                   max_alloc       = 0;
    size_t                   alignment       = 128;

    ~ggml_ocl_device_context() {
        if (q_load) {
            clReleaseCommandQueue(q_load);
        }
    }

    // 进程级共享 kernel 管理器: backend 可能被 sched 多次 init/free,
    // 启动期只编译一次, 避免每次 init 重复编译
    ocl_kernel_mgr kmgr;

    // 显存记账 (DESIGN.md 16.5): 已分配 cl_mem 总量, 供 get_memory 使用
    size_t mem_allocated = 0;
};

// ---------------------------------------------------------------------------
// mem module cross-module symbols (ggml-ocl-mem.cpp)
// ---------------------------------------------------------------------------

const char *                             ggml_ocl_buffer_type_get_name(ggml_backend_buffer_type_t buft);
extern struct ggml_backend_buffer_type_i ggml_ocl_buffer_type_interface;

// ---------------------------------------------------------------------------
// error handling (DESIGN.md section 11.3)
// ---------------------------------------------------------------------------

#define OCL_CHECK(err)                                                                          \
    do {                                                                                        \
        cl_int err_ = (err);                                                                    \
        if (err_ != CL_SUCCESS) {                                                               \
            GGML_LOG_ERROR("ggml-ocl: %s error %d at %s:%d\n", #err, err_, __FILE__, __LINE__); \
            GGML_ASSERT(0);                                                                     \
        }                                                                                       \
    } while (0)

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ---------------------------------------------------------------------------
// caps (DESIGN.md sections 5.1 / 13)
// ---------------------------------------------------------------------------

struct ggml_ocl_caps {
    cl_device_id device             = nullptr;
    char         device_name[128]   = { 0 };
    char         driver_version[64] = { 0 };

    int ocl_c_major = 1;
    int ocl_c_minor = 2;
};

// caps module (ggml-ocl-caps.cpp)
void ggml_ocl_caps_probe(ggml_ocl_caps * caps);
void ggml_ocl_caps_print(const ggml_ocl_caps * caps);

// ---------------------------------------------------------------------------
// tensor extra (DESIGN.md section 16.3: view 语义)
// ---------------------------------------------------------------------------

struct ggml_ocl_tensor_extra {
    cl_mem   data_device = nullptr;  // 张量所在的 device buffer
    cl_ulong offset      = 0;        // 在 buffer 内的偏移 (不含 view_offs)

    void reset() {
        data_device = nullptr;
        offset      = 0;
    }
};

// ---------------------------------------------------------------------------
// fp16 存储: F32 张量在设备上以 half 存放, ggml 侧仍报 F32。
// 只影响 type == F32 的张量; Q4_1 / F16 / I32 等原样存放。
// 设备上数据紧跟在 slot 起始处, 所以 extra->offset 不用折半;
// 只有 view_offs / 步长 / 访问长度需要折半。设备必须先确认支持 cl_khr_fp16。
// ---------------------------------------------------------------------------

static inline bool ocl_f16_packed(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32;
}

// 设备上的绝对字节偏移: half 存放时 view_offs 以 F32 字节计, 要折半
static inline cl_ulong ocl_dev_offset(const ggml_tensor * t, const ggml_ocl_tensor_extra * e) {
    return e->offset + (ocl_f16_packed(t) ? (t->view_offs >> 1) : t->view_offs);
}

// 设备访问的字节数: host 侧的 size / offset 以 F32 字节计, half 存放时要折半
static inline size_t ocl_dev_bytes(const ggml_tensor * t, size_t n) {
    return ocl_f16_packed(t) ? n / 2 : n;
}

// 传给 kernel 的步长: half 存放时 F32 张量的 nb[] 减半
static inline cl_int ocl_nb(const ggml_tensor * t, size_t nb) {
    return (cl_int) (ocl_f16_packed(t) ? (nb >> 1) : nb);
}

// 同上, 给步长参数是 ulong 的 kernel 用
static inline cl_ulong ocl_nb64(const ggml_tensor * t, size_t nb) {
    return (cl_ulong) (ocl_f16_packed(t) ? (nb >> 1) : nb);
}

// extra 对象池: 从 buffer ctx 内分配/回收 extra
class ocl_extra_pool {
    std::vector<ggml_ocl_tensor_extra *> free_;  // 可复用
    std::vector<ggml_ocl_tensor_extra *> all_;   // 全部已分配 (reset/free 用)
  public:
    ggml_ocl_tensor_extra * alloc();
    void                    reset();     // 全部归还 free_ (buffer reset 时)
    void                    free_all();  // buffer 释放时统一回收
};

// ---------------------------------------------------------------------------
// op registry (DESIGN.md section 19.1: 通用, 覆盖全部 GGML_OP_*)
// ---------------------------------------------------------------------------

struct ocl_op {
    enum ggml_op op       = GGML_OP_NONE;
    int          unary_op = 0;  // 仅 op==GGML_OP_UNARY 时有效 (0 = 全部)
    bool (*supports)(const ggml_ocl_caps * caps, const ggml_tensor * t)                                  = nullptr;
    bool (*run)(ggml_ocl_backend * b, const ggml_tensor * s0, const ggml_tensor * s1, ggml_tensor * dst) = nullptr;
};

// ops module (ggml-ocl-ops.cpp)
bool ocl_op_dispatch(ggml_ocl_backend * b, ggml_tensor * node);
bool ocl_op_supports(const ggml_ocl_caps * caps, const ggml_tensor * node);

// ---------------------------------------------------------------------------
// profiling stats (DESIGN.md section 17.4)
// ---------------------------------------------------------------------------

#ifdef GGML_OCL_PROFILING
struct ocl_kernel_stat {
    std::string op_name;
    std::string kernel_name;
    int         count    = 0;
    cl_ulong    total_ns = 0;
};

struct ocl_stats {
    std::vector<ocl_kernel_stat> kernels;
    void                         record(const char * op, const char * kernel, cl_ulong ns);
    void                         print() const;
};
#endif  // GGML_OCL_PROFILING

// ---------------------------------------------------------------------------
// kernel call wrapper (DESIGN.md section 17.1; 实现与队列工具在 ggml-ocl-exec.cpp)
// 参数顺序约定 (DESIGN.md 21.2): src0, off0, src1, off1, dst, offd, 形状..., 步长...
// ---------------------------------------------------------------------------

struct ocl_kernel_call {
    cl_kernel kernel    = nullptr;
    cl_uint   ndims     = 1;
    size_t    global[3] = { 0, 0, 1 };
    size_t    local[3]  = { 0, 0, 1 };

    const char * op_name = nullptr;  // profiling 用 (可空)

    union ocl_arg {
        cl_mem   mem;
        cl_ulong u64;
        cl_int   i32;
        cl_float f32;
        cl_int4  i32x4;
    };

    ocl_arg args[64]      = {};  // 上限 64 参数 (gemv_q4_1 需 33)
    size_t  arg_sizes[64] = {};
    int     arg_idx       = 0;

    void arg_cl_mem(cl_mem v) {
        args[arg_idx].mem  = v;
        arg_sizes[arg_idx] = sizeof(cl_mem);
        arg_idx++;
    }

    void arg_u64(cl_ulong v) {
        args[arg_idx].u64  = v;
        arg_sizes[arg_idx] = sizeof(cl_ulong);
        arg_idx++;
    }

    void arg_i32(cl_int v) {
        args[arg_idx].i32  = v;
        arg_sizes[arg_idx] = sizeof(cl_int);
        arg_idx++;
    }

    void arg_i32x4(cl_int4 v) {
        args[arg_idx].i32x4 = v;
        arg_sizes[arg_idx] = sizeof(int32_t) * 4;
        arg_idx++;
    }

    void arg_f32(cl_float v) {
        args[arg_idx].f32  = v;
        arg_sizes[arg_idx] = sizeof(cl_float);
        arg_idx++;
    }

    // 提交到 compute 队列; profiling 开启时同步结算耗时
    void enqueue(ggml_ocl_backend * backend, cl_event * evt_out = nullptr);
};

// ---------------------------------------------------------------------------
// exec 队列工具 (DESIGN.md sections 17.2/17.3)
// ---------------------------------------------------------------------------

void ocl_exec_wait_pending_copies(ggml_ocl_backend * backend);
void ocl_exec_flush(ggml_ocl_backend * backend);

// ---------------------------------------------------------------------------
// backend main structure (DESIGN.md section 12.2)
// ---------------------------------------------------------------------------

struct ggml_ocl_backend {
    ggml_ocl_caps    caps;
    cl_context       context   = nullptr;
    cl_command_queue q_compute = nullptr;  // in-order 计算队列
    cl_command_queue q_copy    = nullptr;  // in-order 拷贝队列

    ocl_kernel_mgr * kmgr = nullptr;       // 进程级共享 (指向 dev_ctx->kmgr)

    bool profiling_enabled = false;        // 实际创建 profiling queue 是否成功

    // 显存记账 (DESIGN.md 16.5): 已分配 buffer 总字节, 供 get_memory
    size_t mem_allocated = 0;

    // 跨队列事件: copy 队列最近事件列表, 供 compute 首节点等待
    std::vector<cl_event> pending_copy_events;

#ifdef GGML_OCL_PROFILING
    ocl_stats stats;
#endif
};
