#pragma once
// ggml-ocl backend: internal shared definitions (DESIGN.md section 12)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define CL_TARGET_OPENCL_VERSION GGML_OCL_TARGET_VERSION
#include <CL/cl.h>

// OpenCL 2.1 device info (absent from some header revisions)
#ifndef CL_DEVICE_SUBGROUP_SIZES
#define CL_DEVICE_SUBGROUP_SIZES 0x1025
#endif

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
// device context (DESIGN.md section 11.2; created in ggml-ocl.cpp)
// ---------------------------------------------------------------------------

struct ggml_ocl_device_context {
    cl_platform_id platform = nullptr;
    cl_device_id   device   = nullptr;
    std::string    platform_name;
    std::string    device_name;
    cl_device_type device_type = 0;
    size_t         global_mem_size = 0;
    cl_context     context = nullptr;       // shared per platform
    int            context_refs = 0;
    ggml_ocl_backend * backend = nullptr;   // set on init_backend
    ggml_backend_buffer_type buffer_type = {};
};

// ---------------------------------------------------------------------------
// mem module cross-module symbols (ggml-ocl-mem.cpp)
// ---------------------------------------------------------------------------

const char * ggml_ocl_buffer_type_get_name(ggml_backend_buffer_type_t buft);
extern struct ggml_backend_buffer_type_i ggml_ocl_buffer_type_interface;

// ---------------------------------------------------------------------------
// constants (DESIGN.md section 14: inline constants, no config files)
// ---------------------------------------------------------------------------

// DRAM bandwidth of the target GPU, used for budget/utilization estimates only
#define GGML_OCL_DRAM_BW_MBPS 42700

// "any type / any dim" sentinels for rule matching
#define OCL_ANY_TYPE ((ggml_type) -1)
#define OCL_ANY_DIM  (-1)

// ---------------------------------------------------------------------------
// error handling (DESIGN.md section 11.3)
// ---------------------------------------------------------------------------

#define OCL_CHECK(err)                                                      \
    do {                                                                    \
        cl_int err_ = (err);                                                \
        if (err_ != CL_SUCCESS) {                                           \
            GGML_LOG_ERROR("ggml-ocl: %s error %d at %s:%d\n",              \
                #err, err_, __FILE__, __LINE__);                            \
            GGML_ASSERT(0);                                                 \
        }                                                                   \
    } while (0)

#undef  MIN
#undef  MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

static inline size_t ocl_align(size_t v, size_t a) {
    GGML_ASSERT(a && (a & (a - 1)) == 0);
    return (v + a - 1) & ~(a - 1);
}

// ---------------------------------------------------------------------------
// weight layout (DESIGN.md section 12.3)
// ---------------------------------------------------------------------------

enum ocl_weight_layout {
    LAYOUT_AOS = 0,    // 原样 (默认)
    LAYOUT_SOA_DM = 1, // d/m 平面 + q 平面分离 (Q4_1 候选, 实测后开)
};

// ---------------------------------------------------------------------------
// caps (DESIGN.md sections 5.1 / 13)
// ---------------------------------------------------------------------------

struct ggml_ocl_caps {
    cl_platform_id platform = nullptr;
    cl_device_id   device   = nullptr;
    char device_name[128]      = {0};
    char driver_version[64]    = {0};

    int ocl_c_major = 1;
    int ocl_c_minor = 2;

    uint32_t vendor_id = 0;        // 自研 GPU vendor id
    int chip_gen      = 0;         // 代际枚举 (vendor 层解析)
    int compiler_major = -1;       // 自研编译器版本契约
    int compiler_minor = -1;

    bool fp16              = false;
    bool dp4a              = false;
    bool has_matrix_unit   = false;
    bool images            = false;   // 已确认支持; 默认仍走纯 buffer, 纹理路径实测
    bool image1d_buffer    = false;
    bool subgroups         = false;
    bool subgroup_shuffle  = false;

    size_t wave_size = 64;          // 已确认
    size_t max_wg    = 1024;        // 待确认
    size_t local_mem = 8192;        // 已确认: 4 x 1024 x 2 B
    size_t max_alloc = 0;
    size_t image_max_buffer = 0;
    size_t alignment = 128;

    size_t global_mem = 12ull * 1024 * 1024 * 1024;  // 已确认: 最大 12 GB
    size_t dram_bandwidth_mbps = GGML_OCL_DRAM_BW_MBPS;

    bool out_of_order_queues = false;   // 已确认: 不支持
    bool svm = false;
};

// caps module (ggml-ocl-caps.cpp)
void ggml_ocl_caps_probe(ggml_ocl_caps * caps);
int  ggml_ocl_parse_chip(const char * device_name);
void ggml_ocl_caps_print(const ggml_ocl_caps * caps);

// ---------------------------------------------------------------------------
// tensor extra (DESIGN.md section 16.3: view 语义)
// ---------------------------------------------------------------------------

struct ggml_ocl_tensor_extra {
    cl_mem   data_device = nullptr;  // 主平面 (AoS: 全部数据; SoA-DM: q 平面)
    cl_mem   aux_device  = nullptr;  // SoA-DM: d/m 平面 (AoS: null)
    cl_ulong offset = 0;             // 在 buffer 内的偏移 (不含 view_offs)
    uint32_t layout = LAYOUT_AOS;    // 当前布局

    void reset() {
        data_device = nullptr;
        aux_device  = nullptr;
        offset      = 0;
        layout      = LAYOUT_AOS;
    }
};

// ---------------------------------------------------------------------------
// pools (DESIGN.md section 16.1)
// ---------------------------------------------------------------------------

// scratch/temp 池: size-class 分桶 LRU
class ocl_pool {
    cl_context context_ = nullptr;
    std::map<size_t, std::vector<cl_mem>> free_;   // 按 size class 分桶 (2 的幂)
    size_t reserved_ = 0;
    size_t high_water_ = 0;
    std::mutex m_;
public:
    void init(cl_context ctx);
    cl_mem alloc(size_t size);        // size 向上对齐到桶
    void release(cl_mem mem, size_t size);
    void trim();                      // reserved_ > high_water_*2 时释放最旧批次
    size_t reserved() const { return reserved_; }
    void clear();
};

// subbuffer 池: 布局转换/转置中间件高频使用
class ocl_subpool {
    std::map<std::pair<cl_mem, size_t>, std::vector<cl_mem>> free_;
    std::mutex m_;
public:
    cl_mem alloc(cl_mem parent, size_t origin, size_t size);
    void release(cl_mem sub, cl_mem parent, size_t size);
    void clear();
};

// extra 对象池: 从 buffer ctx 内分配/回收 extra
class ocl_extra_pool {
    std::vector<ggml_ocl_tensor_extra *> free_;
public:
    ggml_ocl_tensor_extra * alloc();
    void free_all();                  // buffer 释放时统一回收
};

// ---------------------------------------------------------------------------
// kernel manager (DESIGN.md section 15)
// ---------------------------------------------------------------------------

enum ocl_kernel_state { KS_OK, KS_FAILED };

struct ocl_kernel_entry {
    const char * src_id = nullptr;      // 源名, 如 "gemv_q4_1"
    std::string compile_opts;           // 按类别的编译选项
    ocl_kernel_state state = KS_OK;
    cl_program program = nullptr;
    std::vector<cl_kernel> kernels;     // 同一源内的多个 kernel 函数 (按 fn 名索引)
};

class ocl_kernel_mgr {
    std::mutex m_;                      // 仅启动期需要; 推理期单线程读
    std::map<std::string, ocl_kernel_entry> entries_;
public:
    // 启动期: 编译全部 kernel (backend 创建时调用一次)
    void compile_all(ggml_ocl_backend * backend);
    // 推理期: 纯查表, 零编译; 返回 nullptr 表示该源不可用
    cl_kernel get(const char * src_id, const char * fn_name);
    bool is_ready(const char * src_id) const;
};

// ---------------------------------------------------------------------------
// rule table (DESIGN.md sections 7.1 / 19.1: 仅 GEMM 家族内部使用)
// ---------------------------------------------------------------------------

struct ocl_rule {
    const char * name = nullptr;         // 规则名
    ggml_type src0t = OCL_ANY_TYPE;      // 0 = 不限
    ggml_type src1t = OCL_ANY_TYPE;
    int min_ne00 = 0, min_ne01 = 0, min_ne11 = 0;
    int dk = 0, dv = 0, gqa = 0;         // 注意力特化 (0 = 不限)
    const char * kernel = nullptr;       // kernel 函数名
    int nth0 = 64, nth1 = 1, ndst = 4;   // wavefront 数 x 行数
    int layout = LAYOUT_AOS;
    const char * env_override = nullptr; // 调试开关 (GGML_OCL_DISABLE_<name>)
};

// ---------------------------------------------------------------------------
// op registry (DESIGN.md section 19.1: 通用, 覆盖全部 GGML_OP_*)
// ---------------------------------------------------------------------------

struct ocl_op {
    enum ggml_op op = GGML_OP_NONE;
    int unary_op = 0;                    // 仅 op==GGML_OP_UNARY 时有效 (0 = 全部)
    bool (*supports)(const ggml_ocl_caps * caps, const ggml_tensor * t) = nullptr;
    bool (*run)(ggml_ocl_backend * b, const ggml_tensor * s0,
                const ggml_tensor * s1, ggml_tensor * dst) = nullptr;
};

// ---------------------------------------------------------------------------
// vendor (DESIGN.md section 20; 定义在 vendor/vendor.h, 此处前向声明)
// ---------------------------------------------------------------------------

struct ggml_ocl_vendor;

// ---------------------------------------------------------------------------
// profiling stats (DESIGN.md section 17.4)
// ---------------------------------------------------------------------------

#ifdef GGML_OCL_PROFILING
struct ocl_kernel_stat {
    std::string op_name;
    std::string kernel_name;
    int count = 0;
    cl_ulong total_ns = 0;
};
struct ocl_stats {
    std::vector<ocl_kernel_stat> kernels;
    void record(const char * op, const char * kernel, cl_ulong ns);
    void print() const;
};
#endif // GGML_OCL_PROFILING

// ---------------------------------------------------------------------------
// backend main structure (DESIGN.md section 12.2)
// ---------------------------------------------------------------------------

struct ggml_ocl_backend {
    ggml_ocl_caps caps;
    cl_context context = nullptr;
    cl_command_queue q_compute = nullptr;   // in-order 计算队列
    cl_command_queue q_copy    = nullptr;   // in-order 拷贝队列

    ocl_kernel_mgr kmgr;
    ocl_pool scratch_pool;
    ocl_subpool sub_pool;

    const ggml_ocl_vendor * vendor = nullptr;

    // 跨队列事件: copy 队列最近事件列表, 供 compute 首节点等待
    std::vector<cl_event> pending_copy_events;

#ifdef GGML_OCL_PROFILING
    ocl_stats stats;
#endif
};
