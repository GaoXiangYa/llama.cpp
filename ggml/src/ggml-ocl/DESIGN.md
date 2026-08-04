# ggml-ocl 后端重构设计文档

| 项 | 内容 |
|---|---|
| 版本 | v0.6 (review 草稿) |
| 状态 | 待评审 |
| 目标硬件 | 自研桌面 GPU (Glenfly) |
| 首个模型 | Qwen3-1.7B (GGML_TYPE_Q4_1) |
| 参考实现 | ggml/src/ggml-opencl (仅作学习参考，kernel 全部自研) |
| 变更记录 | v0.2: 按已确认硬件参数调整 (wavefront 64 / fp16 有 / dp4a 无 / 矩阵单元无 / 带宽 42.7 GB/s / in-order / PCIe 3.0); v0.3: 显存 12 GB / local mem 8 KB 每 CU / image 支持已确认, fp32-fp16 TFLOPS 仍未提供; v0.4: 新增第二部分模块详细设计 (第 11-24 节); v0.5: 删除 config 模块 (配置内联化), 明确两级注册表 (通用 op 注册表 + GEMM 家族规则表); v0.6: kernel 管理改为借鉴 ggml-opencl load_cl_kernels 过程 - 启动时全量编译, 推理期零编译 (撤销离线编译方案) |

---

## 1. 背景与目标

### 1.1 背景

- 在 ggml backend 框架下为自研桌面 GPU 实现 OpenCL 后端，目录 `ggml/src/ggml-ocl/`。
- 自研 GPU 提供 OpenCL 接口，具备自研 ICD 与 OpenCL C 编译器（可定义厂商扩展与版本契约）。
- ggml-opencl 是为移动端 Adreno GPU 设计的后端：特化深度高、工程缺陷多。本设计**保留其架构思路，重写其实现**，kernel 全部由团队自研，ggml-opencl 的 kernel 仅作学习参考。

### 1.2 硬件基线（已确认，2026-08-04）

| 参数 | 值 | 设计影响 |
|---|---|---|
| wavefront (subgroup) | 64 | 所有 kernel 以 64 线程 wavefront 为映射单位 |
| fp16 | 支持 | GEMM 走 fp16 路径 |
| dp4a 整数点积 | 不支持 | 无量化 GEMM 捷径，量化权重靠即时解量化 |
| 矩阵单元 | 无 | GEMM 仅通用 fp16/fp32 FMA |
| DRAM 带宽 | 42.7 GB/s | **decode 性能的唯一硬约束，全设计围绕它展开** |
| out-of-order 队列 | 不支持 | in-order 单计算队列，执行模型简化（第 5.2 节） |
| PCIe | 3.0 | 全量 offload 无压力；权重加载约 100ms |
| 显存容量 | 最大 12 GB | 64k 上下文 f16 KV / 128k 上下文 q8_0 KV（第 8.1 节预算） |
| local mem 每 CU | 8 KB (4x1024x2 B) | **tiling 上限硬约束**，GEMM/FA 以寄存器分块为主（第 6.3/6.4 节） |
| image (纹理) | 支持 | FA/权重读取可实测 image1d 路径（默认仍纯 buffer） |

### 1.3 带宽现实（先设预期，再谈优化）

42.7 GB/s 属于移动级带宽。Qwen3-1.7B Q4_1 权重约 **1.06 GB**（5 bits/weight，含 d/m 开销），decode 每 token 必须读完全部权重：

```
理论上限 (t/s) = 42.7 GB/s / (权重 1.06 GB + KV 读取量)
```

| 上下文 | KV f16 读取 | 每 token 总读取 | 理论上限 | 70% 利用率目标 |
|---|---|---|---|---|
| 4k | 0.47 GB | 1.53 GB | 27.9 t/s | ~19 t/s |
| 8k | 0.94 GB | 2.00 GB | 21.3 t/s | ~15 t/s |
| 32k | 3.76 GB | 4.82 GB | 8.9 t/s | ~6 t/s |
| 32k + KV q8_0 | 1.88 GB | 2.94 GB | 14.5 t/s | ~10 t/s |

结论（写进需求的三个事实）：
1. **decode 是纯带宽游戏**：计算量 2x1.7e9 FLOPs/token，在 25ms 预算内仅需 ~0.14 TFLOPS 持续算力，计算永远不会是瓶颈；
2. **上下文越长 KV 越贵**：32k 时 KV 读取是权重的 3.5 倍，t/s 断崖式下降；**KV 量化 (q8_0) 是长上下文的最关键收益项**；
3. 性能目标一律以带宽预算为基准，任何超出预算的数字都不现实；
4. **显存 12 GB 预算**：权重 1.06 GB + 激活/scratch 约 1.5 GB 后，f16 KV 可支撑约 64k 上下文，q8_0 KV 可支撑约 128k（第 8.1 节）。

### 1.4 里程碑目标

| 里程碑目标 | 说明 |
|---|---|
| M2 (首个性能里程碑) | Qwen3-1.7B Q4_1 端到端推理，4k 上下文 f16 KV decode 达到带宽预算 70% (~19 t/s) |
| 后续 | 全量化类型、KV q8_0、更多模型 (MoE、多模态)、长上下文优化 |
| 非目标 (v1) | 移动端优化、训练算子、split 多卡、超过带宽预算的性能追求 |

### 1.5 与 ggml-opencl 的定位差异

| 维度 | ggml-opencl (Adreno 移动) | ggml-ocl (自研桌面, 低带宽) |
|---|---|---|
| 约束 | 带宽极小、编译器弱、L2 无调度 | 带宽 42.7 GB/s（移动级）、编译器自研可控 |
| 策略 | 纹理缓存 (image1d)、SoA+转置等特技 | 带宽利用率为王 + KV 量化；特技按实测启用 |
| 重点 | kernel 内省到极致 | 带宽预算内做对，主机开销从简 |
| 假设 | 硬编码 wave=64、Adreno 行为 | 能力模型查询（wave=64 已确认，仍走 caps） |

---

## 2. ggml-opencl 经验：取长补短对照表

### 2.1 借鉴（取长）

| 借鉴点 | ggml-opencl 位置 | 本设计落地 |
|---|---|---|
| 能力查询式初始化（平台/设备/扩展/编译器版本） | ggml-opencl.cpp:5809 | caps 能力模型 (第 6.1 节) |
| kernel 嵌入二进制 + 启动时全量编译 (load_cl_kernels) | ggml-opencl.cpp:1262, CMakeLists | kernel 管理器 (第 15 节): 启动编译全部, 推理期零编译 |
| subgroup (wavefront) 编程 + 寄存器分块 | kernels/rms_norm.cl, mul_mv_q4_0_f32.cl | 自研 kernel 设计原则 (第 7.1 节) |
| 算子融合 (rms_norm+mul, norm+mul+add, MoE combine) | :6952 | 融合调度 (第 8.2 节) |
| 权重 SoA 拆分 (scales/quants 分离, subbuffer 零拷贝) | :7961 | 权重布局策略 (第 6.4 节) |
| 按 (dk,dv) 手调 tile 表 | fa_tune.h | 编译期常量表, kernel 内按 shape 判断 (第 14 节) |
| 完整 profiling 基础设施 (cl_event 全时段) | :350 | 保留并默认内置 |
| 优雅降级 (支持 op 才接管, 否则 CPU 回退) | :24434 | 渐进启用策略 (第 10 节) |

### 2.2 避免（补短）

| ggml-opencl 缺陷 | 后果 | 本设计对策 |
|---|---|---|
| 单文件 24,792 行 | 无法并行开发/评审 | 按算子模块化 (第 5 节) |
| mul_mat 2300 行 if-else 迷宫 + 20+ 环境变量 | 不可维护、不可调 | 规则表驱动 dispatch (第 8.1 节) |
| 每节点全同步 | 无异步重叠 | 依赖由 in-order 提交顺序保证，仅跨队列/跨后端同步 (第 6.2 节) |
| 无内存池 (每次 clCreateBuffer/SubBuffer) | 分配开销高 | size-class 池 + subbuffer 池 (第 6.3 节) |
| 厂商代码 #ifdef 焊死 | 多代际难扩展 | 厂商插件接口 (第 5.4 节) |
| 手调常量散落 + 20+ 环境变量 | 换硬件重调、排障困难 | 常量内联 + 少量调试 env 开关 (第 14 节) |
| CI 只编译不测试, 测试用例 #if 0 | 回归风险高 | 真机 CI + kernel harness (第 9.3/9.4 节) |

---

## 3. 硬件能力清单

### 3.1 已确认

| 参数 | 值 | 影响 |
|---|---|---|
| wavefront 尺寸 | 64 | kernel 映射单位 |
| fp16 | 支持 | GEMM fp16 路径可行 |
| dp4a | 不支持 | 无量化 GEMM 捷径 |
| 矩阵单元 | 无 | 无张量核心路径 |
| DRAM 带宽 | 42.7 GB/s | 性能预算的唯一硬约束 |
| out-of-order 队列 | 不支持 | in-order 执行模型 |
| PCIe | 3.0 | offload/H2D 无压力 |
| 显存容量 | 最大 12 GB | 上下文支撑上限 (第 8.1 节) |
| local mem 每 CU | 8 KB | GEMM/FA tiling 硬约束 |
| image 支持 | 支持 | 纹理路径可实测 (默认纯 buffer) |

### 3.2 待确认（M1 前补齐）

| 参数 | 影响 | 缺省假设 |
|---|---|---|
| fp16 计算吞吐 (TFLOPS) | prefill GEMM 上界 | 仍未提供; 按 fp32 的 2x 假设, 实测修正 |
| fp32 计算吞吐 (TFLOPS) | prefill 兜底路径 | 仍未提供; 需实测 |
| 最大 workgroup 尺寸 | 并行度上限 | 按 1024 假设 |
| L2 容量与行为 | 权重布局 (AoS vs SoA) 选择 | 按小 L2 假设 (不做布局赌注) |
| image1d_buffer 具体限制 | FA 纹理路径可行性 | 已确认 image 支持; 运行时查 IMAGE_MAX_BUFFER_SIZE |
| SVM / zero-copy | H2D 策略 | 按不支持假设 |
| 自研编译器行为契约 | 编译选项/精度 | M1 用保守选项，harness 验证 |

**缺省全部按保守假设设计**，通用路径先行，实测后逐项开启。

---

## 4. 总体架构

### 4.1 分层

```
llama.cpp 应用层 (llama-cli/server)
        |
        v
ggml backend API (ggml-backend.h: device/buffer/graph_compute/supports_op)
        |
        v
ggml-ocl host runtime      <- 设备/内存/kernel 编译/调度/融合 (本设计核心)
        |
        v
vendor 层 (gf-arise)       <- 代际特化: 专用 kernel 注册、布局策略、规则覆盖
        |
        v
kernels/ (自研 .cl)        <- 全部重写, 嵌入式编译
```

### 4.2 目录结构

```
ggml/src/ggml-ocl/
├── CMakeLists.txt
├── ggml-ocl.h                      # 对外声明 (后端注册)
├── ggml-ocl.cpp                    # 后端注册、device/buffer 接口、graph_compute
├── ggml-ocl-caps.cpp               # 设备探测 + 能力模型 (第 6.1 节)
├── ggml-ocl-mem.cpp                # 内存池、subbuffer 池、权重布局转换 (第 6.3/6.4 节)
├── ggml-ocl-kernels.cpp            # kernel 二进制加载/回退编译 (第 15 节)
├── ggml-ocl-exec.cpp               # 队列管理、跨队列/跨后端同步、graph 执行 (第 6.2 节)
├── ggml-ocl-fusion.cpp             # 图级融合识别 (第 8.2 节)
├── ops/
│   ├── op-gemv.cpp                 # decode GEMV 调度 (规则表)
│   ├── op-gemm.cpp                 # prefill GEMM 调度
│   ├── op-mul-mat-id.cpp           # MoE
│   ├── op-flash-attn.cpp           # FA (decode/prefill)
│   ├── op-norm.cpp                 # rms_norm/norm/group_norm (+融合)
│   ├── op-rope.cpp / op-softmax.cpp / op-cpy.cpp / op-other.cpp
├── vendor/
│   ├── vendor.h                    # 厂商插件接口 (第 5.4 节)
│   └── gf-arise.cpp                # 自研 GPU 特化注册
├── kernels/                        # 自研 .cl (embed 嵌入, 启动时全量编译, 第 15/21 节)
│   ├── gemv/  gemm/  flash_attn/  norm/  misc/
└── DESIGN.md                       # 本文档
```

### 4.3 模块职责边界

- 通用层 (ggml-ocl*.cpp + ops/)：**不包含任何厂商 if/def**，只依赖 caps 能力位。
- vendor 层：通过注册表提供"接管点"，通用层先查 vendor 表，未接管走通用路径。
- ops/ 内每个算子文件独立编译单元，单人可维护、可独立评审。

### 4.4 厂商插件接口

```cpp
// vendor/vendor.h
struct ggml_ocl_vendor {
    const char * name;
    bool (*match)(const ggml_ocl_caps & caps);            // vendor_id + chip_gen
    void (*init)(ggml_ocl_backend *);                     // 注册 kernel/布局策略/规则
    // 接管点: 返回 true 表示已处理
    bool (*mul_mat)(ggml_ocl_backend *, const ggml_tensor * s0,
                    const ggml_tensor * s1, ggml_tensor * dst);
    bool (*mul_mat_id)(...);
    bool (*flash_attn)(...);
    bool (*weight_convert)(...);                          // 上传时布局转换
};

// 注册顺序即优先级; 末尾 nullptr 表示通用路径
static const ggml_ocl_vendor * g_ocl_vendors[] = { &gf_arise_vendor, nullptr };
```

新代际适配 = 新增一个 vendor 条目，通用层零改动。矩阵单元若未来加入，同样以 vendor 注册独立 GEMM 规则接入。

---

## 5. 运行时设计

### 5.1 能力模型 (caps)

初始化时一次性查询，全后端只读共享（已确认项已标注）：

```cpp
struct ggml_ocl_caps {
    cl_platform_id platform; cl_device_id device;
    int ocl_c_major, ocl_c_minor;
    uint32_t vendor_id;            // 自研 GPU vendor id
    int chip_gen;                  // 代际枚举 (由 vendor 层解析)
    int compiler_major, compiler_minor;   // 自研编译器版本契约
    bool fp16;                     // (已确认: true)
    bool dp4a;                     // (已确认: false)
    bool has_matrix_unit;          // (已确认: false)
    bool images, image1d_buffer;   // (已确认: images=true; 默认纯 buffer, 纹理路径实测)
    bool subgroups, subgroup_shuffle;
    size_t wave_size;              // (已确认: 64)
    size_t max_wg, local_mem, max_alloc, image_max_buffer, alignment;  // (local_mem 已确认: 8192)
    size_t global_mem, l2_size, dram_bandwidth_mbps;  // (global_mem 已确认: 12 GB; 带宽已确认: 42700)
    bool out_of_order_queues;      // (已确认: false)
    bool svm;                      // (缺省: false)
};
```

版本契约示例：自研编译器 >= 2.1 保证 fp16 融合质量、fast-math 数值行为可控，caps 用版本位判断，不做字符串解析。

### 5.2 执行模型: in-order 单计算队列（已确认硬件后的简化设计）

**硬件事实**：无 out-of-order 队列 -> 同队列内 kernel 按提交顺序执行，依赖天然由顺序保证。

```
graph_compute 伪代码:

for node in graph:
    if (node 无计算标志 || 是 VIEW/PERMUTE 等) continue;
    if (src 属于其他后端) sync_with_other_backends();   // 唯一跨后端同步点
    if (可融合) 发融合 kernel (吞并后续 1-2 节点); else 发单算子 kernel;
    // 不追踪 per-node event: in-order 队列保证顺序

图尾: clFlush(queue_compute);  // 不阻塞
宿主需要读回/图边界才 clFinish;
```

- **双队列**：`queue_compute` (in-order) + `queue_copy` (H2D/D2H)。跨队列依赖用 event 衔接（每批上传一个 event，等待其完成后计算队列才消费）。
- **重叠机会**：权重上传（copy 队列）与首批计算并行；H2D 激活上传与上一节点计算并行。kernel 之间无重叠（in-order 限制，但图依赖本来就是串行的，无实际损失）。
- **每 token 一次 clFlush**，不逐节点 clFinish。
- **host 开销优先级下调**：GPU 每 token 耗时 25ms+（4k 上下文），host 1-2ms 占比 <10%，不再是首要瓶颈。融合仍保留（减节点、减排障面），但不为 host 开销做过度工程。
- 跨后端同步保留：仅在节点 src 属于其他后端时触发。

### 5.3 内存管理

- **用户 buffer**（权重/KV）：每个 alloc 一个 cl_mem，不做池（生命周期长）。
- **scratch/temp 池**：size-class 分桶 LRU，`high_water * 2` 触发 trim。
- **subbuffer 池**：权重 SoA 拆分、转置中间件高频 create/release subbuffer，池化复用（ggml-opencl 此处无池，是高频开销点）。
- **host staging**：PCIe 3.0 下 H2D 延迟可控；activation 每步上传量小（KB 级），直接同步写或 async copy 均可，实测决定；SVM/zero-copy 缺省不用。
- **alloc_size 计算**：量化张量预留 SoA 对齐 slack（参照 ggml-opencl :10687 的 4*alignment 做法，但按实际组件数精确计算）。

### 5.4 权重布局策略（可插拔，默认 AoS）

上传 (set_tensor) 时按 vendor 策略选择布局，kernel 与布局解耦：

```
enum weight_layout { LAYOUT_AOS, LAYOUT_SOA_DM, LAYOUT_SOA_TRANSPOSED };
weight_layout vendor_pick_layout(caps, ggml_type);
```

- **Q4_1 布局分析**（20 字节/块 = half2 dm + 16B nibbles，见附录 B）：AoS 行内块粒度非 16B 对齐（20 非 16 倍数），向量化受限。SoA-DM（d/m 平面 + q 平面分离）恢复 16B 对齐且核内可 half2 向量化。
- **决策依据**：低带宽 GPU 上 L2 行为未知，**默认 AoS 先跑通**，实测加载效率后决定是否开启 SoA-DM（若 AoS 已达预算利用率，SoA 不开启，控制复杂度）。
- 转换 kernel 自研（参考 ggml-opencl :7961 的 subbuffer 零拷贝思路，不双份分配）。

### 5.5 kernel 编译管理（启动时全量编译，推理期零编译）

**目标**：借鉴 ggml-opencl 的 `load_cl_kernels`（ggml-opencl.cpp:1262）过程——**backend 创建（程序启动）时一次性编译全部 kernel**，此后推理过程不做任何 kernel 编译，`get` 为纯查表。

```
启动期 (backend 创建, 一次性):
  load_cl_kernels() 等价物: 遍历静态 kernel 注册表 (全部 .cl 源)
    -> 每个源: clCreateProgramWithSource + clBuildProgram (按类别编译选项)
    -> 提取全部 kernel 函数 (clCreateKernel)
    -> 失败: 记日志 + 标记不可用 (规则匹配自动跳过, 不 abort)

推理期:
  get(src_id, fn) -> 查表返回 cl_kernel, 零编译
```

- **kernel 源嵌入二进制**（沿用 ggml-opencl 的 embed_kernel.py 机制，CMake 调 Python 生成 `.cl.h`），无运行时文件依赖；
- **kernel 变体数量必须收敛**：启动编译集是静态有限的。FA 不做 ggml-opencl 式 per-(dk,dv) 宏编译（否则变体数爆炸、启动编译时间不可控），改为**运行期参数化**（dk/dv/tile 作 kernel 参数传入），固定 fa_decode + fa_prefill 两个源；
- 启动编译耗时计入后端初始化（约 1-3 秒量级，按 kernel 数实测），可接受——与推理期零编译的收益交换；
- 编译选项按 kernel 类别固定（第 15.5 节），运行期不变化。

---

## 6. Kernel 设计（自研重写核心）

### 6.1 设计原则

1. **wavefront 为中心**：wavefront = 64 线程（已确认）。每个 wavefront 固定处理 N_DST 行（寄存器分块），行内 stride 循环；local size 一律取 64 的倍数；
2. **16B 对齐向量加载**：全部 kernel 的 global 访问按 float4/uint4 粒度设计；布局不满足时先转换；
3. **local mem 约束 (8 KB/CU 已确认)**：GEMM/FA 的 tiling 以寄存器分块为主，local mem 只做小尺寸单/双缓冲 (f16 64x32 tile = 4 KB)；GEMV 零 local mem 保占用率；双缓冲占用 >= 4 KB 时占用率降至 1-2 wavefront/CU，需以高 ILP 与寄存器预取补偿延迟（实测为准）；
4. **解量化最小指令化**：Q4_1 用 nibble 提取 + 乘加，禁止逐元素分支；
5. **decode/prefill 分离**：GEMV 与 GEMM 是两套 kernel，不共用；
6. **fp32 累加，规约顺序确定**：结果与 CPU 参考的误差在 harness 容差内；
7. **量化类型参数化**：以 block traits (QK/QR/组件布局) 模板化，新增量化类型是增量的（Q4_0 只需改 traits 与解量化内联函数）。

### 6.2 GEMV (decode 命脉)

**瓶颈定性**：纯 DRAM 带宽瓶颈（第 1.3 节已给预算表）。计算量 ~3.4 GFLOPs/token，在 25ms 预算内仅需 ~0.14 TFLOPS 持续算力 -> **kernel 优化重心是加载效率与占用率（latency hiding），不是 ALU 技巧**。

**kernel 设计**：
- 网格：每 wavefront (64 线程) 处理 N_DST 行（默认 4，调参），遍历 K 方向；
- 加载：每线程每次迭代 16B（Q4_1 = 8 quants），一个 wavefront 一次覆盖 1 KB 连续权重（64 x 16B）-> 追求 DRAM 饱和需足够的 wavefront 并行度（Qwen3 各投影行数 2048-151936，天然充足）；
- 计算：nibble 提取 -> fp32 乘加；d 与 m 分别累加，末次合并（减少每元素乘法）；
- 规约：wavefront 内 reduce_add，一行一个结果；
- 形状特化（先做 Qwen3-1.7B 的形状）: ne00=2048（隐藏层），行数 N=11008/2048/151936 三类（FFN up/down、attn proj、embedding 输出）各做参数特化，用规则表接入；
- **权重必须保持 Q4_1 原样读取**（1.06 GB 已是最小字节数），不做 f16 化（2.1 GB 直接腰斩带宽预算）。

### 6.3 GEMM (prefill)

- **路径唯一化**：dp4a 不支持、矩阵单元无 -> 只剩 **fp16 FMA 路径**（激活转 f16 + 权重即时解量化到 f16，寄存器/local mem 内完成，**权重不上传为 f16**，保持 1.06 GB 读取量）；
- **tiling 设计（8 KB local mem 约束下）**：以 BM=64, BN=64 输出 tile 为起点，每线程 8x8 f32 微块（64 累加寄存器）；A 激活 f16 tile 64x32 = 4 KB 双缓冲（共 8 KB，恰满）；B 权重按 Q4_1 原样从 global 直读进寄存器即时解量化（0.625 B/元素，带宽友好，不占 local mem）；若实测占用率不足（1 wavefront/CU 时延迟隐藏差），降 BK 至 16（2 KB x 2 = 4 KB，恢复 2 wavefront/CU）或以高 ILP/预取补偿；
- 即时解量化开销：解量化到 f16 的指令成本由 fp16 FMA 的吞吐提升覆盖，harness 实测验证（决策点）；
- split-K 支持长 prompt prefill（K 维度拆分 + 部分和归约到 global）；
- prefill 上界未知项：fp16 计算吞吐（TFLOPS 仍未提供），预算公式 `prefill 时间 = max(带宽下界 25ms, 计算时间)`。

### 6.4 Flash Attention

- **decode FA 是必须项，不是可选项**：非 FA 路径（QK^T -> softmax -> PV）对 KV 读两遍，FA 只读一遍。以 8k 上下文 f16 KV 计，非 FA 每 token 多读 0.94 GB -> 上限从 21.3 t/s 掉到 14 t/s。**低带宽硬件上 FA 是决定性优化**。
- v1 范围：f16 KV，decode kernel（单 query 行 x 全 KV，wavefront 并行扫 KV，**零 local mem**，纯寄存器累加 + global 直读，占用率最大化）+ prefill kernel（BM x BN tile + online softmax）；
- **prefill tile 约束（8 KB local mem）**：S/Q/K/V 全放 local 不现实，推荐 BM=16-32 x BN=16-32，K/V 以 f16 单缓冲小 tile 或 **image1d 纹理路径**承载（image 已确认支持，与纯 buffer 实测对比后定默认）；S 与 P 计算以寄存器完成；
- KV 缓存 head-major 布局由 llama.cpp 图给出（permuted K），kernel 按此布局设计连续读取；
- **KV 量化 (q8_0) 为 M3 高优先项**：llama.cpp 支持 `-ctk/-ctv q8_0`，KV 读取减半，32k 上下文 decode 上限从 8.9 提到 14.5 t/s。需实现 q8_0 K/V 的 FA decode kernel（核内解量化）；
- 与 "gemm(QK) + softmax + gemm(PV)" 基线实测对比确认收益（预期 FA 显著胜出）；
- 不做量化 Q 侧（q4_0/q8_0 Q 变体），按需后续加。

### 6.5 其他算子

| 算子 | 设计 |
|---|---|
| rms_norm | 融合 mul（llama.cpp 图中 rms_norm 后必接 mul），wavefront 归约（参考 ggml-opencl rms_norm.cl 结构） |
| rope | neox 风格（Qwen3），向量化旋转，freq 预计算上传 |
| soft_max | 行内归约 + scale/max_bias 参数 |
| cpy/dup | 格式转换矩阵（f32/f16/量化 互相转换），作为权重上传与 FA 前处理的底层 |
| add/mul/scale 等 elementwise | 通用 4 元素向量 kernel，支持广播 |
| get_rows/set_rows | embedding 查表（GEMV 之外的另一带宽点，数据量小） |

### 6.6 数值与精度

- 全部中间累加 fp32；Q4_1 解量化公式 `x = d*q + m`（q 为 4-bit [0,15]），与 ggml CPU 参考逐位对齐验证；
- fast-math 只用于有 harness 数值验证的 kernel；
- bf16 模型权重按 f16 存储于设备（沿用 ggml-opencl :18479 约定）。

---

## 7. 调度设计

### 7.1 规则表驱动 dispatch

mul_mat/FA 的所有特化分支收敛为数据表（新形状特化 = 加一行）：

```cpp
struct ocl_rule {
    const char * name;
    ggml_type src0t, src1t;          // 0 = any
    int min_ne00, min_ne01, min_ne11;
    int dk, dv, gqa;                 // 注意力特化 (0 = 不限)
    const char * kernel;             // kernel 名
    int nth0, nth1, ndst;            // wavefront 数 x 行数
    int layout;                      // AoS/SoA-DM/...
    const char * env_override;       // 调试开关 (默认 nullptr)
};

static const ocl_rule g_gemv_rules[] = {
    { "q4_1_dense_2048", GGML_TYPE_Q4_1, GGML_TYPE_F32, 2048, 64, 0, 0,0,0,
      "gemv_q4_1_n2048", 64, 4, LAYOUT_AOS, nullptr },
    { "q4_1_generic",     GGML_TYPE_Q4_1, GGML_TYPE_F32, 0, 0, 0, 0,0,0,
      "gemv_q4_1", 64, 4, LAYOUT_AOS, nullptr },
    ...
};

const ocl_rule * ocl_match_rule(caps, rules, s0, s1);
// 失败: kernel 未编译成功/能力位不满足 -> 自动跳过 -> 下一规则或 CPU 回退
```

- 规则表同时服务 `supports_op` 与运行时 dispatch（同一匹配函数）；
- 调参（nth/ndst/layout）是**编译期常量**（规则表条目内的值），修改即改代码重编；kernel 内的自适应参数（如 BK 降级）在 kernel 内按形状运行时判断；
- `env_override` 仅保留为调试开关（`GGML_OCL_DISABLE_<rule>` 形式），不构成配置体系；**不使用任何配置文件**（第 14 节）。

### 7.2 算子融合

- rms_norm + mul（必做，图里成对出现）；
- norm/group_norm + mul + add（LLaVA 等视觉模型需要）；
- MoE combine 链（后续 MoE 模型）；
- 融合在 graph_compute 内做模式匹配（可关闭，排障用）。

### 7.3 CPU 回退

- `supports_op` 未声明支持的算子由 llama.cpp 调度器自动落到 CPU backend；
- 渐进启用策略：M1 只开已自研 kernel 的算子，其余全部回退 -> 任何时刻主分支可运行、可验证。

---

## 8. 性能工程与验证

### 8.1 性能预算（以 42.7 GB/s 为准）

| 指标 | 公式 | v1 目标 |
|---|---|---|
| decode 带宽利用率 | (t/s x 每 token 读取量) / 42.7 GB/s | >= 70% |
| decode t/s (4k ctx, f16 KV) | 预算上限 27.9 t/s | >= 19 t/s |
| decode t/s (32k ctx, q8_0 KV) | 预算上限 14.5 t/s | >= 10 t/s (M3) |
| prefill | max(带宽下界 25ms, 计算时间) | 达到两者较大者的 80% |
| host 开销 | 每 token host 时间 | <= 2 ms (优先级低) |
| 图执行重叠 | copy 队列与 compute 并行 | 权重加载与首批计算重叠 |

**显存预算 (12 GB, 已确认)**：

| 项 | 占用 |
|---|---|
| 权重 (Q4_1) | 1.06 GB |
| 激活 + scratch/池 | 预留 ~1.5 GB |
| KV f16 | 112 KB/token -> 64k 上下文约 7.2 GB（合计 ~9.8 GB，可容纳） |
| KV q8_0 | 56 KB/token -> 128k 上下文约 7.2 GB（合计 ~9.8 GB，可容纳） |

结论：12 GB 下 f16 KV 支持到约 64k 上下文，q8_0 KV 支持到约 128k；超出部分由 llama.cpp 自动回退 CPU 卸载，或提示用户降低上下文。

基准工具：`llama-bench`（pp/tg 对比 CPU-only 与 ggml-ocl），`test-backend-ops`（正确性）。

### 8.2 调参（无配置文件，内联常量）

- 所有调参项（规则表 nth/ndst、GEMM BM/BN/BK、FA tile、融合开关）为**代码内编译期常量**，集中在各模块顶部常量区（一处维护）；
- 决策逻辑在 kernel/调度处按实际 shape 运行时判断（如 BK 按 ne00 降级），**不读取任何外部配置**；
- 调试开关仅保留 `GGML_OCL_*` 环境变量（如 `GGML_OCL_DISABLE_FUSION=1`），用于排障，不承载调参职责；
- autotune 不作为 v1 目标：带宽瓶颈下参数空间小（主要就是 ndst/BK），harness 实测 + 常量修改即可收敛；若后续需要，harness 输出 CSV 人工导入常量，仍不引入运行时配置。

### 8.3 kernel 开发 harness（自研 kernel 的快速迭代工具）

独立于 llama.cpp 的 kernel 单测程序（建议自研，参考 test-backend-ops 的驱动方式）：

```
用法: ocl-kernel-test <kernel名> [--shape=...] [--seed=N]
流程: 生成随机输入(权重+激活) -> CPU 参考实现计算 -> 设备 kernel 执行
      -> 比较 (绝对/相对容差, 逐元素与整体) -> 打印带宽/耗时
```

- 每个 kernel 提交前必跑：**数值验证** + **简单计时**（带宽利用率直接可见，与预算表对照）；
- 参考实现直接复用 ggml CPU 的 quantize/dequantize/dot 逻辑（保证数值口径一致）；
- 这是"kernel 全部自研"路线下质量与效率的基石，优先开发。

### 8.4 回归与 CI

- test-backend-ops 全量算子回归（后端无关，直接可用），真机 runner 挂自研 GPU；
- 每 PR: kernel harness 全量 + test-backend-ops + llama-bench 关键 shape；
- 每日: 完整模型链路（Qwen3-1.7B）perplexity 与生成正确性对照；
- 开启 ggml-opencl 中被 `#if 0` 的 OpenCL 用例作为补充（tests/test-backend-ops.cpp:8823 附近）。

---

## 9. 里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| M0 骨架 | 目录/模块拆分, 空实现注册, 通用路径全回退 CPU; 补齐第 3.2 节硬件参数 | llama-cli 可跑（纯 CPU 行为不劣化） |
| M1 通用 kernel | 全部算子自研通用版 (AoS 布局, 保守编译) | test-backend-ops 全绿, Qwen3-1.7B 可生成 |
| M2 Q4_1 decode 性能 | GEMV 特化 (Qwen3 形状), FA decode (f16 KV), 双队列重叠 | 4k ctx f16 KV decode >= 18-19 t/s (预算 70%), host <= 2ms |
| M3 prefill + 长上下文 | GEMM fp16 路径 + split-K, FA prefill, KV q8_0 FA decode | prefill 达带宽下界 80%; 32k ctx + q8 KV decode >= 10 t/s |
| M4 扩展 | 全量化类型, 常量调参收敛, 真机 CI, MoE | 多模型多量化回归稳定 |

每个里程碑保持主分支可运行；**先功能后性能，性能改动一律带回归数据（带宽利用率）**。

---

## 10. 风险与开放问题

| # | 风险/问题 | 影响 | 对策 |
|---|---|---|---|
| 1 | 低带宽 (42.7 GB/s) 限制长上下文 t/s | 32k 上下文 decode 上限仅 9-15 t/s | 需求层面明确预期; KV q8_0 优先; 建议产品按上下文分级给出性能指标 |
| 2 | fp16/fp32 计算吞吐未提供 | prefill 上界未定 | M1 前测 TFLOPS; prefill 预算公式已留位 |
| 3 | local mem 仅 8 KB/CU | GEMM/FA tiling 受限, 占用率可能 1-2 wavefront/CU | 寄存器分块为主; GEMV 零 local mem; harness 实测调整 |
| 4 | 自研编译器 fast-math 数值行为 | 精度回归 | harness 数值验证门禁 |
| 5 | 驱动 enqueue 开销高 | host 时间超预算 | 低优先级; 融合 + 每 token 一次 flush; 驱动侧优化需求 |
| 6 | Q4_1 实用性（业界多用 Q4_0/Q4_K） | 后续迁移成本 | traits 参数化, Q4_0 增量接入 (且 Q4_0 0.96GB 更省带宽) |
| 7 | L2 行为未知; image 已支持但收益未测 | SoA/image 优化方向未定 | 默认 AoS + 纯 buffer, image1d 列入 FA 实测项 |
| 8 | llama.cpp 上游接口演进 | 维护成本 | 仅用稳定 backend API, 不做 hack |

---

---

# 第二部分 模块详细设计

## 11. 模块依赖与总体时序

### 11.1 模块依赖图

```
caps (13)  <- 无依赖, 最先开发
kernels (15) <- caps, 嵌入源 (autogenerated/*.cl.h, 由 embed 脚本生成)
mem (16)    <- caps, kernels (转换 kernel)
exec (17)   <- caps, kernels, mem, fusion, ops, vendor
fusion (18) <- 无 (编译期常量开关)
ops (19)    <- caps, kernels, mem
vendor (20) <- caps, kernels
ggml-ocl.cpp (后端注册) <- 全部
harness (23) <- kernels, mem (独立于 llama.cpp 运行)
```

### 11.2 初始化时序

```
ggml_backend_opencl_reg()                     [ggml-ocl.cpp]
  -> ggml_ocl_probe_devices()                 枚举平台/设备, 过滤不支持项, 选默认设备
  -> ggml_ocl_caps_probe(&caps)               [13] 一次性查询
  -> vendor 匹配: ggml_ocl_vendor_match()     [20] 首个 match 生效
  -> ggml_ocl_backend_create(dev, caps, vendor)
       -> cl_context / q_compute / q_copy     [17]
       -> 池初始化                             [16]
       -> kmgr.compile_all(): 启动期全量编译所有 kernel (embed 源, 一次性)
          成功 -> 就绪 (推理期零编译); 个别失败 -> 标记不可用, 规则跳过 [15]

buffer 生命周期:
  alloc_buffer()                              [16] clCreateBuffer
  init_tensor()                               [16] 分配 extra; view 复用 parent extra
  set_tensor()                                [16] 布局转换 (AoS 直写 / SoA-DM 转换 kernel)

每 token 计算:
  graph_compute()                             [17]
    -> 融合扫描 (吞并节点)                    [18]
    -> 跨后端同步 (仅 src 属其他后端时)
    -> vendor 接管尝试                        [20]
    -> op run()                               [19] 规则表匹配
    -> kmgr.get_kernel() (查表, 命中即返回)   [15]
    -> 统一 enqueue 封装                      [17]
  -> clFlush; 图边界才 clFinish
```

### 11.3 错误处理总则（贯穿所有模块）

| 错误类型 | 处理 |
|---|---|
| 启动期 kernel 编译失败 | 该源标记不可用, 规则匹配自动跳过, 落到下一规则或 CPU 回退 (不 abort) |
| 驱动查询失败 (非关键) | 记 warn 日志 + 使用缺省值 |
| 驱动 enqueue/参数错误 | GGML_ASSERT (运行期出现 = 程序 bug) |
| 显存分配失败 | 返回 null, 上层 (llama.cpp) 触发 CPU 回退 |

## 12. 公共定义 (ggml-ocl.h / ggml-ocl-internal.h)

### 12.1 ggml-ocl.h (对外)

```cpp
// 仅暴露后端注册入口 (与 ggml-opencl.h 对齐)
ggml_backend_reg_t ggml_backend_opencl_reg(void);
```

### 12.2 ggml-ocl-internal.h (模块间共享, 全部内部模块 include)

```cpp
#pragma once
#include "ggml-backend-impl.h"
#include <CL/cl.h>

// ---------- caps (完整字段, 探测见 13 节) ----------
struct ggml_ocl_caps {
    cl_platform_id platform; cl_device_id device;
    char device_name[128]; char driver_version[64];
    int ocl_c_major, ocl_c_minor;
    uint32_t vendor_id;              // 自研 GPU vendor id
    int chip_gen;                    // 代际枚举
    int compiler_major, compiler_minor;
    bool fp16, dp4a, has_matrix_unit;
    bool images, image1d_buffer;
    bool subgroups, subgroup_shuffle;
    size_t wave_size;                // 64
    size_t max_wg, local_mem;        // 8192
    size_t max_alloc, image_max_buffer, alignment;
    size_t global_mem;               // 12 GB
    size_t dram_bandwidth_mbps;      // 42700 (常量, 见 13.3 节)
    bool out_of_order_queues;        // false
    bool svm;
};

// ---------- tensor extra (view 语义见 16.4 节) ----------
struct ggml_ocl_tensor_extra {
    cl_mem   data_device;            // 主平面 (AoS: 全部数据; SoA-DM: q 平面)
    cl_mem   aux_device;             // SoA-DM: d/m 平面 (AoS: null)
    cl_ulong offset;                 // 在 buffer 内的偏移 (不含 view_offs)
    uint32_t layout;                 // LAYOUT_AOS / LAYOUT_SOA_DM
};

// ---------- buffer ctx ----------
struct ggml_ocl_buffer_ctx {
    cl_mem mem; size_t size;
    ocl_extra_pool extras;           // extra 对象池 (见 16.3 节)
};

// ---------- backend 主结构 ----------
struct ggml_ocl_backend {
    ggml_ocl_caps caps;
    cl_context context;
    cl_command_queue q_compute;      // in-order 计算队列
    cl_command_queue q_copy;         // in-order 拷贝队列
    ocl_kernel_mgr kmgr;             // 15 节
    ocl_pool scratch_pool;           // 16 节
    ocl_subpool sub_pool;
    const ggml_ocl_vendor * vendor;  // 20 节
    // 跨队列事件: copy 队列最近事件列表, 供 compute 首节点等待
    std::vector<cl_event> pending_copy_events;
    // 统计 (GGML_OCL_PROFILING 时启用)
    ocl_stats stats;
};

// ---------- 规则/算子注册表 (见 19 节) ----------
struct ocl_rule { ... };             // 同 7.1 节定义
struct ocl_op { ... };               // 同 19.1 节定义

// ---------- 辅助宏 ----------
#define OCL_CHECK(err) ...           // 等价 ggml-opencl 的 CL_CHECK (enqueue 类错误 assert)
static inline size_t ocl_align(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }
```

### 12.3 布局枚举

```cpp
enum ocl_weight_layout {
    LAYOUT_AOS = 0,    // 原样 (默认)
    LAYOUT_SOA_DM = 1, // d/m 平面 + q 平面分离 (Q4_1 候选, 实测后开)
};
```

## 13. caps 模块详细设计 (ggml-ocl-caps.cpp)

### 13.1 函数签名

```cpp
void ggml_ocl_caps_probe(ggml_ocl_caps * caps);        // 全字段查询
int  ggml_ocl_parse_chip(const char * device_name);    // 代际识别 (自研命名规则)
void ggml_ocl_caps_print(const ggml_ocl_caps * caps);  // 启动日志
```

### 13.2 字段来源表（探测实现清单）

| 字段 | CL API | 失败缺省 |
|---|---|---|
| device_name / driver_version | CL_DEVICE_NAME / CL_DRIVER_VERSION | "unknown" |
| ocl_c_major/minor | CL_DEVICE_OPENCL_C_VERSION | 1.2 |
| vendor_id | CL_DEVICE_VENDOR_ID | 0 |
| fp16 | 扩展串含 cl_khr_fp16 | false |
| dp4a | 扩展串含 cl_khr_integer_dot_product | false |
| images / image1d_buffer | CL_DEVICE_IMAGE_SUPPORT / 扩展串 | false |
| subgroups | CL_DEVICE_SUBGROUP_SIZES (2.1+) | 用 wave_size=64 假设 |
| wave_size | CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 探测 kernel 或设备信息 | 64 |
| max_wg / local_mem / max_alloc | CL_DEVICE_MAX_WORK_GROUP_SIZE / LOCAL_MEM_SIZE / MAX_MEM_ALLOC_SIZE | 1024 / 8192 / 1 GB |
| alignment | CL_DEVICE_MEM_BASE_ADDR_ALIGN | 128 |
| global_mem | CL_DEVICE_GLOBAL_MEM_SIZE | 12 GB |
| 带宽 | 无法直接查询 -> caps 内编译期常量 42700 (见 13.3) | 42700 |

### 13.3 要点

- **带宽为编译期常量**：`caps.dram_bandwidth_mbps = 42700` 直接写入 caps 模块常量区（不读任何配置）；该值仅用于性能预算/利用率估算与日志，不影响正确性，换硬件时改常量重编即可；
- 编译器版本解析：自研编译器版本串格式与硬件团队约定，一次性解析进 compiler_major/minor，不做字符串模式匹配；
- 探测结果全量打印到日志（`llama-cli --list-devices` 可看到），便于现场排障；
- 所有探测失败不 abort。

## 14. 配置策略（无配置文件，内联化）

**原则**：不引入任何配置文件与解析器。所有可调参数是代码内编译期常量；运行时决策在 kernel/调度处按实际 shape 判断。

### 14.1 参数归属表（替代原 config 各节）

| 参数 | 归属 | 形式 |
|---|---|---|
| DRAM 带宽 | caps 模块常量区 | `#define GGML_OCL_DRAM_BW_MBPS 42700` |
| GEMV ndst/nth | op-gemv 规则表条目 | 编译期常量 (规则表字段) |
| GEMM BM/BN/BK | op-gemm 常量区 | 编译期常量; BK 按 ne00 运行时降级判断在 kernel 内 |
| FA tile/nth | op-flash-attn 常量区 | 编译期常量 |
| 融合开关 | fusion 模块常量区 | `static const bool fusion_enabled = true;` |
| 权重布局 (AoS/SoA-DM) | mem 模块常量区 | 编译期常量, 实测后改值重编 |
| 规则级调试开关 | 各规则表 env_override | `GGML_OCL_DISABLE_<rule>` 环境变量 (仅排障) |

### 14.2 决策内联示例（kernel 内按 shape 灵活判断）

```c
// 例 1: GEMM 中 BK 按 ne00 降级 (local mem 8 KB 约束)
int bk = (ne00 >= 4096) ? 16 : 32;      // 大 K 降 BK 保占用率

// 例 2: GEMV 中按行数选微块宽度
int ndst = (ne01 >= 4096) ? 8 : 4;      // 行数充足时加大每 wavefront 行数

// 例 3: FA 中按 n_kv 选 kernel 变体 (decode 固定 q1, prefill 按长度分档)
if (n_q == 1) { fa_decode(...); } else if (n_kv < 2048) { fa_prefill_small(...); }
```

### 14.3 环境变量（仅调试，非配置体系）

- `GGML_OCL_DISABLE_FUSION=1`：关闭融合（对照实验/排障）；
- `GGML_OCL_DISABLE_<rule>=1`：禁用特定规则（规则表 env_override 字段）；
- `GGML_OCL_LOG=2`：详细日志（kernel 选择、时序、带宽利用率）；
- 全部为布尔/整型开关，不承载参数值，不影响默认行为。

### 14.3 要点

- 未知 key / 非法值：warn + 默认值，不阻断启动；
## 15. kernels 模块详细设计 (ggml-ocl-kernels.cpp)

### 15.1 设计目标（对齐 ggml-opencl 的 load_cl_kernels 过程）

- **启动期全量编译**：backend 创建时（程序启动）遍历静态 kernel 注册表，一次性 `clBuildProgram` 编译全部 kernel 源；
- **推理期零编译**：`get` 为纯查表，不触发任何编译；
- 个别源编译失败：标记不可用，规则匹配自动跳过（落 CPU 回退），不阻断启动；
- **kernel 变体数量收敛**（关键约束）：启动编译集必须是静态有限集合。FA 不做 per-(dk,dv) 宏编译变体，改为运行期参数化（dk/dv/tile 作为 kernel 参数），固定 2 个 FA 源（decode/prefill）——否则启动编译时间不可控。

### 15.2 数据结构

```cpp
enum ocl_kernel_state { KS_OK, KS_FAILED };
// KS_OK    : 启动期编译成功, kernel 句柄可用
// KS_FAILED: 启动期编译失败 (该源不可用, 规则匹配跳过)

struct ocl_kernel_entry {
    const char * src_id;                // 源名, 如 "gemv_q4_1"
    std::string compile_opts;           // 按类别的编译选项 (第 15.4 节)
    ocl_kernel_state state = KS_OK;
    cl_program program = nullptr;
    std::vector<cl_kernel> kernels;     // 同一源内的多个 kernel 函数 (按 fn 名索引)
};

class ocl_kernel_mgr {
    std::mutex m;                       // 仅启动期需要; 推理期单线程读
    std::map<std::string, ocl_kernel_entry> entries_;
public:
    // 启动期: 编译全部 kernel (backend 创建时调用一次, 见 15.3)
    void compile_all(ggml_ocl_backend * b);
    // 推理期: 纯查表, 零编译; 返回 nullptr 表示该源不可用
    cl_kernel get(const char * src_id, const char * fn_name);
    bool is_ready(const char * src_id) const;
};
```

### 15.3 启动期编译流程（compile_all）

```
compile_all():
  for (src_id, 类别) in 静态注册表:          // 注册表 = kernels/ 全部 .cl, 编译期列出
    opts = 类别编译选项 (第 15.4 节)
    src  = 嵌入源 (autogenerated/<src_id>.cl.h)   // embed_kernel.py 生成
    program = clCreateProgramWithSource + clBuildProgram(opts)   // 非 fatal
    成功 -> 提取全部 kernel 函数 (clCreateKernel) -> state = OK
    失败 -> 日志 (含 build log, 前 20 行) + state = FAILED, 继续下一个源

  GGML_LOG_INFO("ggml_opencl: loaded N/M kernel sources in %.2f s")
```

- 注册表为编译期静态数组（`{ src_id, 类别 }`），新增 kernel = 加一行 + 加 .cl 文件；
- 编译耗时计入启动（目标 < 3 秒，M1 实测；若超限，把 FA/GEMM 等大源拆小或降级选项）。

### 15.4 编译选项（按类别固定）

| 类别 | 选项 | 说明 |
|---|---|---|
| 全部 | `-cl-std=CL2.0 -cl-mad-enable` | 基线 |
| gemv/gemm/fa (性能类) | + `-cl-fast-relaxed-math` | 逐 kernel 经 harness 数值验证后加入 |
| 需要 fp16 的 | 源内 `#pragma OPENCL EXTENSION cl_khr_fp16` | 不传编译选项 |

编译选项表为 kmgr 内静态常量（改选项 = 改代码重编），运行期不变化。

### 15.5 推理期 get（零编译）

```
get(src_id, fn):
  entry = entries_[src_id]
  if entry.state == FAILED -> return nullptr      // 调用方规则匹配跳过
  return entry.kernels[fn 索引]                    // 纯查表, 无任何编译/加锁开销
```

## 16. mem 模块详细设计 (ggml-ocl-mem.cpp)

### 16.1 池设计

```cpp
class ocl_pool {           // scratch/temp 池
    std::map<size_t, std::vector<cl_mem>> free_;   // 按 size class 分桶
    size_t reserved_ = 0, high_water_ = 0;
public:
    cl_mem alloc(size_t size);        // size 向上对齐到桶 (2 的幂)
    void release(cl_mem m, size_t size);
    void trim();                      // reserved_ > high_water_*2 时释放最旧批次
    size_t reserved() const;
};

class ocl_subpool {        // subbuffer 池 (布局转换/转置中间件高频使用)
    // 键: (parent cl_mem, size); 释放的 subbuffer 按键回收复用
    std::map<std::pair<cl_mem, size_t>, std::vector<cl_mem>> free_;
public:
    cl_mem alloc(cl_mem parent, size_t origin, size_t size);
    void release(cl_mem sub);
    void clear();
};
```

要点：分配均按 `max(alignment, 16)` 对齐；池线程安全（低频操作，mutex 足够）。

### 16.2 buffer 接口实现清单

| 接口 | 实现 |
|---|---|
| alloc_buffer | clCreateBuffer (size>=1 防 -61)；ctx 创建 (含 extra 池) |
| init_tensor | 见 16.3 |
| set_tensor | 见 16.4 (布局转换) |
| get_tensor | 读回；SoA-DM 时先经转换 kernel 还原 AoS 再读 |
| cpy_tensor | clEnqueueCopyBuffer (同 context 内) |
| get_alloc_size | ggml_nbytes + SoA 对齐 slack (按组件数精确计算) |
| get_alignment | caps.alignment |

### 16.3 extra 分配与视图语义（正确性关键，ggml-opencl PR#7640 的教训）

```cpp
// ocl_extra_pool: 从 buffer ctx 内分配 extra 对象
ggml_ocl_tensor_extra * ocl_extra_alloc(ggml_ocl_buffer_ctx * bctx);
void ocl_extra_free_all(ggml_ocl_buffer_ctx * bctx);   // buffer 释放时统一回收

// init_tensor 规则:
//  - 非 view: 从池分配新 extra, 记录 offset = (tensor->data - buffer base)
//  - view: 复用 parent 的 extra (不新分配, 不写 offset)
//  - 运行期有效偏移: eff_offset = extra->offset + tensor->view_offs
//    任何使用处 (kernel 参数) 运行时计算, 绝不在 init 时固化
```

### 16.4 set_tensor 布局转换流程

```
set_tensor(buffer, tensor, data, offset, size):

  AoS 路径 (默认):
    clEnqueueWriteBuffer(q_copy, data_device, offset 处写 size 字节)

  SoA-DM 路径 (q4_1 等, 编译期常量 layout_q4_1 == SOA_DM 时):
    1. 写临时 AoS buffer (q_copy)
    2. 从 extra 池取 SoA-DM extra (或复用已分配)
    3. cvt 转换 kernel: AoS -> (q 平面 + d/m 平面), 两个 clCreateSubBuffer 别名
    4. 更新 tensor->extra 指向 SoA-DM extra (q = data_device, d/m = aux_device)
    5. 释放临时 buffer (回 scratch 池)

  约束: 仅对 非 view 且 连续 的主张量做 SoA (view 的 parent 拥有布局, 沿用 ggml-opencl 规则)
```

### 16.5 显存记账

```cpp
// 供 ggml_backend_opencl_device_get_memory 使用
// total = caps.global_mem (12 GB); free = total - 已分配 buffer 合计 - 池预留
// 无可靠查询 API, 按记账值 (与 8.1 节预算联动, 超出 90% 时日志提示)
```

## 17. exec 模块详细设计 (ggml-ocl-exec.cpp)

### 17.1 kernel 调用封装（消灭 ggml-opencl 式 20 行样板）

```cpp
struct ocl_kernel_call {
    cl_kernel kernel;
    size_t global[3] = {0,0,1}, local[3] = {0,0,1};
    int arg_idx = 0;

    void arg_cl_mem(cl_mem m);
    void arg_u64(cl_ulong v);
    void arg_i32(cl_int v);
    void arg_f32(cl_float v);

    void enqueue(ggml_ocl_backend * b,
                 cl_uint ndims,
                 cl_event * evt_out = nullptr);   // 内部 OCL_CHECK + clEnqueueNDRangeKernel
};
// 参数顺序约定 (全后端统一, 见 21 节):
//   0: src0 cl_mem, 1: src0 offset, 2: src1 cl_mem, 3: src1 offset,
//   4: dst cl_mem, 5: dst offset, 之后形状/步长/标量
```

### 17.2 graph_compute 细化流程

```cpp
ggml_status ocl_exec_graph(ggml_ocl_backend * b, ggml_cgraph * cgraph) {
    for (i = 0; i < cgraph->n_nodes; i++) {
        node = cgraph->nodes[i];
        if (ggml_is_empty(node)) continue;
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        switch (node->op) {
            case GGML_OP_RESHAPE: case GGML_OP_TRANSPOSE:
            case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_NONE:
                continue;                          // 视图类, 无计算
            default: break;
        }

        // 1) 融合扫描 (18 节): 命中则吞并后续 1-2 节点并跳指针
        if (fusion_enabled) {        // 编译期常量 (fusion.cpp 定义)
            if (ocl_fusion_try(b, cgraph, &i)) continue;
        }

        // 2) 跨后端同步: 仅当 src 属于其他后端
        for (s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s] && node->src[s]->buffer &&
                node->src[s]->buffer->buft != b->buft_self) {
                ocl_exec_sync_other(b); break;
            }
        }

        // 3) vendor 接管 (20 节)
        if (b->vendor && b->vendor->dispatch &&
            b->vendor->dispatch(b, node)) continue;

        // 4) op 注册表 (19 节): 查表 -> run -> 规则匹配 -> enqueue
        if (!ocl_op_dispatch(b, node)) {
            GGML_ASSERT(false && "op dispatch failed (supports_op 应已拦截)");
        }
    }

    // 图尾: 仅 flush; host 读回或图边界由调用方 clFinish
    clFlush(b->q_compute);
    return GGML_STATUS_SUCCESS;
}
```

### 17.3 跨队列与跨后端同步

- **copy -> compute**：set_tensor 的上传在 q_copy；graph_compute 首节点 enqueue 前，若 pending_copy_events 非空，则以其为 wait 事件（`clEnqueueNDRangeKernel(wait_list)`），随后清空；
- **compute -> copy**：get_tensor 读回时 clFinish(q_compute)（低频操作）；
- **跨后端**：`ocl_exec_sync_other()` 内部 clFinish(q_compute)（与 ggml-opencl 相同语义，但只在确有跨后端 src 时触发）。

### 17.4 profiling (编译宏 GGML_OCL_PROFILING)

```cpp
// 每 kernel 调用记录: (op 名, 次数, 总耗时, 平均耗时, 数据量 -> 带宽利用率估算)
// 汇总表在 backend free 时打印, 或写文件 ggml-ocl-profile.txt
// 默认 OFF; CMake 选项 GGML_OCL_PROFILING=ON 开启
```

## 18. fusion 模块详细设计 (ggml-ocl-fusion.cpp)

### 18.1 模式表

| 模式 | 吞并节点 | 融合 kernel | 条件 |
|---|---|---|---|
| rms_norm + mul | 2 个节点 | `rms_norm_mul` | rms_norm 的 dst 是 mul 的 src0；mul src1 为权重向量 (ne0 == ne00)；连续 |
| norm + mul + add | 3 个节点 | `norm_mul_add` | 同上 + add 广播合法 |
| group_norm + mul + add | 3 个节点 | `group_norm_mul_add` | 视觉模型, M4 |
| MoE combine 链 | k 个 VIEW/ADD | `moe_combine` | MoE 模型, M4 |

### 18.2 接口

```cpp
// 尝试对 cgraph->nodes[i] 起点的模式做融合; 命中返回 true 且 *i 已跳到模式末尾
bool ocl_fusion_try(ggml_ocl_backend * b, ggml_cgraph * cgraph, int * i);
```

实现要点：
- 模式匹配在 exec 循环内做（节点已过滤视图类）；
- 融合 kernel 参数 = 各被吞节点的 src/dst 拼接（rms_norm_mul: src0, off0, src1(权重), off1, dst, offd, 形状, eps）；
- 融合后节点计数下降 -> host 开销与排障面同步下降；
- 融合开关为编译期常量（`fusion.cpp` 内 `static const bool fusion_enabled = true;`），排障时改常量重编或 `GGML_OCL_DISABLE_FUSION=1` 临时关闭。

## 19. ops 模块详细设计 (ops/)

### 19.1 两级注册表（先明确层级关系）

dispatch 由**两层表**组成，职责不同：

```
第 1 层: ocl_op 注册表 (通用, 算子级)
   作用: 把 ggml-opencl 的巨型 switch 变成表驱动
   覆盖: 全部 GGML_OP_* (MUL_MAT / RMS_NORM / ROPE / SOFT_MAX / CPY / ADD ... 每个算子一个条目)
   位置: 所有 op 文件贡献条目, 汇总在 ggml-ocl-ops.cpp (或 ops/ 内注册表文件)

第 2 层: ocl_rule 规则表 (仅特化密集算子内部使用, 按需引入)
   作用: 消化单算子内部的形状/类型级特化分支
   使用方: 只有 MUL_MAT (gemv/gemm 家族)、MUL_MAT_ID (MoE)、FLASH_ATTN_EXT
          这类特化分支多的算子才建规则表; rms_norm/rope 等简单算子不需要
   位置: op-gemv.cpp / op-gemm.cpp / op-flash-attn.cpp 文件内 static 表
```

**回答"注册表是通用的还是 GEMM 特供的"**：`ocl_op` 注册表是**通用**的（所有算子）；`ocl_rule` 规则表是 **GEMM 家族（MUL_MAT/MUL_MAT_ID/FA）特供**的内部机制，其他算子不用。

```cpp
struct ocl_op {
    enum ggml_op op;               // GGML_OP_* (UNARY 用 GGML_OP_UNARY + unary 子表)
    int unary_op;                  // 仅 op==GGML_OP_UNARY 时有效 (0 = 全部)
    bool (*supports)(const ggml_ocl_caps *, const ggml_tensor *);
    bool (*run)(ggml_ocl_backend *, const ggml_tensor * s0,
                const ggml_tensor * s1, ggml_tensor * dst);
};

extern const ocl_op * ggml_ocl_ops[];   // 注册表, 由各 op 文件贡献
bool ocl_op_dispatch(ggml_ocl_backend *, ggml_tensor * node);
```

- `supports_op` 接口（llama.cpp 调度器调用）内部：先查 vendor，再遍历注册表查 `supports`（轻量，类型级判断；规则/编译级检查延迟到 run）；
- `run` 内部做规则表匹配（7.1 节结构，仅 MUL_MAT 家族），失败路径：
  1. 规则匹配不到 -> 通用规则（`*_generic`）；
  2. 通用规则 kernel 不可用（加载失败）-> 由 `supports` 层拦截（调度器回退 CPU）；
  3. 均失败 -> GGML_ASSERT（运行期不一致 = bug）。

### 19.2 算子明细

#### op-gemv.cpp (decode 命脉)

| 项 | 内容 |
|---|---|
| 触发 | MUL_MAT, src0 量化, ne11 == 1 且 batch 小 (decode) |
| kernel | `gemv_q4_1` (通用), `gemv_q4_1_n2048` (Qwen3 K=2048 特化) |
| 规则表 | `{ "q4_1_dense_2048", Q4_1, F32, min_ne00=2048, min_ne01=64, ..., "gemv_q4_1_n2048", nth=64, ndst=4 }` 等 4-6 条 |
| 参数 | r2/r3 广播在 kernel 内处理（沿用 ggml-opencl 约定），减少 enqueue 次数 |
| 数值 | fp32 累加, d/m 分累加末次合并 |

kernel 原型草案 (`kernels/gemv/gemv_q4_1.cl`)：

```c
kernel void gemv_q4_1_n2048(
    global const uchar * src0, ulong off0,   // Q4_1 权重 [2048, N]
    global const float  * src1, ulong off1,  // 激活 [2048, 1]
    global float        * dst,   ulong offd, // 输出 [N, 1]
    int ne01, int ne12, int ne13, int r2, int r3)
{
    // 每 wavefront (64 线程) 处理 ndst=4 行;
    // 每线程 16B 加载 (8 quants), 寄存器解量化, fp32 累加;
    // sub_group_reduce_add 归约, lane 0 写一行结果
}
```

#### op-gemm.cpp (prefill)

| 项 | 内容 |
|---|---|
| 触发 | MUL_MAT, ne11 > 1 或 batch 大 |
| kernel | `gemm_f16`（激活读 f32 -> 寄存器转 f16；权重 Q4_1 即时解量化 f16；fp16 FMA；fp32 累加） |
| tiling | BM=64, BN=64, BK=32 (A 双缓冲 4 KB x2), 每线程 8x8 微块 (见 6.3 节) |
| split-K | BK 之上再切 K 段, 部分和写 scratch, 二次 kernel 归约 (M3) |
| 规则表 | `{ "q4_1_gemm", Q4_1, F32, 0,0,0, ..., "gemm_f16", bm=64, bn=64, bk=32 }` |

#### op-flash-attn.cpp

| kernel | 触发 | 设计 |
|---|---|---|
| `fa_decode_f16` | FLASH_ATTN_EXT, n_q==1, KV f16 | 每 wavefront 一个 head；q 存寄存器；并行扫 KV (global 直读, 零 local mem)；在线 max/sum 累加 |
| `fa_prefill_f16` | FLASH_ATTN_EXT, n_q>1, KV f16 | BM=16-32 x BN=16-32；K/V 小单缓冲或 image1d 路径 (实测)；S/P 寄存器计算 |
| `fa_decode_q8` | 同上, KV q8_0 (M3) | 核内解量化, KV 读取减半 |

统一入口 `op_fa_run`：按 (n_q, KV 类型, dk/dv) 选 kernel；mask/sinks 参数透传；与 gemm+softmax 基线的对比数据驱动默认路径。

#### 其余算子

| 文件 | 算子 | kernel | 要点 |
|---|---|---|---|
| op-norm.cpp | RMS_NORM (+MUL 融合) | `rms_norm`, `rms_norm_mul` | wavefront 归约, 参考 ggml-opencl rms_norm.cl 结构 |
| op-norm.cpp | NORM/GROUP_NORM/L2_NORM | `norm`, `group_norm` | 视觉模型, M4 |
| op-rope.cpp | ROPE | `rope_neox` | 向量化旋转 (float2), freq 预计算上传缓存 |
| op-softmax.cpp | SOFT_MAX | `softmax` | 行归约 + scale/max_bias |
| op-cpy.cpp | CPY/DUP/SET | `cpy` | 格式转换矩阵 (f32/f16/q4_1 等组合表) |
| op-eltwise.cpp | ADD/MUL/SUB/DIV/SCALE | `add`,`mul`,`sub`,`div`,`scale` | 4 元素向量化 + 广播 |
| op-eltwise.cpp | UNARY (SILU/GELU/RELU/...) | `silu`,`gelu`,... | 表驱动 unary 映射 |
| op-rows.cpp | GET_ROWS/SET_ROWS | `get_rows`,`set_rows` | embedding 查表 |
| op-other.cpp | 其余 | - | v1 不实现, supports=false, CPU 回退 |

### 19.3 支持条件汇总（supports_op 判定, 各 op 文件内实现）

- 全 op 前提：src/dst 的 buffer 属于本后端（extra 非空）；
- MUL_MAT：src0 类型在已实现集合 (Q4_1 -> Q4_0 -> Q4_K -> ...)，src1 为 F32/F16，形状合法；
- FLASH_ATTN_EXT：KV 为 F16 或 Q8_0 (M3)，dk/dv 在支持集合 (128 起步, 96/192/256 按需)；
- 未列出的 op：返回 false（自动 CPU 回退）。

## 20. vendor 模块详细设计 (vendor/)

### 20.1 vendor.h（接口定型）

```cpp
struct ggml_ocl_vendor {
    const char * name;
    bool (*match)(const ggml_ocl_caps * caps);
    void (*init)(ggml_ocl_backend * b);      // 注册布局策略/规则覆盖/专用 kernel
    bool (*dispatch)(ggml_ocl_backend * b, const ggml_tensor * node);  // 可选全局接管
    bool (*weight_convert)(ggml_ocl_backend * b, const ggml_tensor * t,
                           cl_mem tmp, cl_mem * q, cl_mem * aux);      // 可选布局接管
};

const ggml_ocl_vendor * ggml_ocl_vendor_match(const ggml_ocl_caps * caps);
// 注册顺序即优先级; 末尾内置 "generic" (全 false), 保证总有 vendor
```

### 20.2 gf-arise.cpp 骨架

```cpp
static bool gf_match(const ggml_ocl_caps * c) {
    return c->vendor_id == GF_VENDOR_ID;     // 自研 GPU vendor id
}
static void gf_init(ggml_ocl_backend * b) {
    // v1: 无特化, 布局策略默认 AoS (编译期常量, 实测后改值重编)
}
static const ggml_ocl_vendor gf_arise_vendor = {
    "gf-arise", gf_match, gf_init, nullptr, nullptr,
};
```

- v1 全部接管点返回 false/不注册 -> 走通用路径（先功能后性能）；
- 未来特化（如专用 GEMM 变体、布局强制 SoA）以新 vendor 函数或规则覆盖接入，不动通用层。

## 21. kernels/ 组织与编写规范

### 21.1 目录与命名

```
kernels/
├── gemv/gemv_q4_1.cl          # 含 gemv_q4_1, gemv_q4_1_n2048 等函数
├── gemm/gemm_f16.cl
├── flash_attn/fa_f16.cl       # fa_decode_f16, fa_prefill_f16
├── norm/rms_norm.cl           # rms_norm, rms_norm_mul
├── rope/rope_neox.cl
├── softmax/softmax.cl
├── cpy/cpy.cl                 # cpy_f32_f16, cpy_f16_f32, cpy_to_q4_1 等
├── eltwise/eltwise.cl         # add/mul/sub/div/scale/silu/gelu/...
└── rows/get_rows.cl           # get_rows, set_rows
```

命名规则：文件 = 逻辑组；kernel 函数名 = 注册名（src_id 用文件名）；`.cl` 由 embed 脚本生成 `.cl.h`（沿用 ggml-opencl 的 embed_kernel.py 机制）。

### 21.2 kernel 编写规范（10 条）

1. **参数顺序统一**：`(src0, off0, src1, off1, dst, offd, 形状..., 步长..., 标量...)`；所有 buffer 参数为 `global uchar*` + `ulong` 偏移，kernel 内首行做偏移换算；
2. **wavefront 对齐**：`local size` 恒为 64 倍数；主循环以 `get_local_id(0)` 起步、`get_local_size(0)` 步进；
3. **向量化**：global 访问按 16B（`float4/uint4/half8`）对齐设计；无法对齐的场景先布局转换，不做非对齐加载赌注；
4. **寄存器分块**：GEMV 每线程 N_DST 行累加；GEMM 每线程 8x8 f32 微块；禁止逐元素 global 读；
5. **local mem 纪律**：声明处注明占用（如 `// local: 4 KB x2 double buffer`）；合计不得超过 caps.local_mem 的 3/4；
6. **fp32 累加**：量化点积、softmax、norm 归约一律 fp32；fp16 仅用于中间激活；
7. **规约确定**：`sub_group_reduce_add` + 局部数组两段式，规约顺序固定，harness 容差内与 CPU 参考一致；
8. **分支**：内层循环禁止数据相关分支；边界用循环拆分（主循环 + 尾部标量循环）；
9. **头注释模板**：目的 / 输入输出 / 对齐要求 / 占用 / 对应设计文档节号；
10. **新 kernel 门禁**：必须过 harness 数值 + 计时（23 节）才可提交。

### 21.3 第一批评审清单（M1-M2 交付）

| kernel | 归属 | 优先级 |
|---|---|---|
| gemv_q4_1 (+ n2048 特化) | op-gemv | P0 (decode 命脉) |
| fa_decode_f16 | op-flash-attn | P0 (KV 只读一遍) |
| rms_norm_mul | op-norm | P0 (每层必用) |
| cpy (f32/f16/q4_1) | op-cpy | P0 (上传/转换底座) |
| add/mul/scale/silu | op-eltwise | P0 |
| get_rows | op-rows | P0 |
| rope_neox | op-rope | P1 |
| softmax | op-softmax | P1 |
| gemm_f16 | op-gemm | P1 (prefill) |
| fa_prefill_f16 | op-flash-attn | P2 (M3) |

## 22. CMakeLists 详细设计

```cmake
# ggml/src/ggml-ocl/CMakeLists.txt (参考 ggml-opencl/CMakeLists.txt 骨架)
find_package(OpenCL REQUIRED)

set(TARGET_NAME ggml-ocl)
ggml_add_backend_library(${TARGET_NAME}
    ggml-ocl.cpp ggml-ocl-caps.cpp
    ggml-ocl-kernels.cpp ggml-ocl-mem.cpp ggml-ocl-exec.cpp
    ggml-ocl-fusion.cpp
    ops/op-gemv.cpp ops/op-gemm.cpp ops/op-flash-attn.cpp
    ops/op-norm.cpp ops/op-rope.cpp ops/op-softmax.cpp
    ops/op-cpy.cpp ops/op-eltwise.cpp ops/op-rows.cpp ops/op-other.cpp
    vendor/gf-arise.cpp
    ../../include/ggml-ocl.h)
target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIBRARIES})

option(GGML_OCL_EMBED_KERNELS "Embed OpenCL kernels into the executable" ON)
option(GGML_OCL_PROFILING     "OpenCL profiling (CPU overhead)" OFF)

# kernel 源嵌入: 沿用 ggml-opencl 的 embed_kernel.py (CMake 调 Python 生成 .cl.h)
# 运行期: 启动时 (backend 创建) 用嵌入源码全量编译一次, 推理期零编译 (见第 15 节)
if (GGML_OCL_EMBED_KERNELS)
    # 对 kernels/ 下每个 .cl 生成 autogenerated/<name>.cl.h
endif()
if (GGML_OCL_PROFILING)
    add_compile_definitions(GGML_OCL_PROFILING)
endif()
add_compile_definitions(GGML_OCL_TARGET_VERSION=${GGML_OPENCL_TARGET_VERSION})
```

接入方式：在 `ggml/src/CMakeLists.txt` 增加 `ggml_add_backend(ggml-ocl)` 分支（与 ggml-opencl 并列，`GGML_OCL` 选项），deps 不含 CUDA/HIP。

## 23. kernel harness 详细设计 (tools/ocl-kernel-test)

### 23.1 目录与模块

```
tools/ocl-kernel-test/
├── main.cpp            # CLI: <kernel 名> [--shape] [--seed] [--iter]
├── case-registry.h     # REGISTER_CASE(name) 宏注册表
├── driver.cpp          # OpenCL 环境 (创建 context/queue, 加载 caps)
├── ref/                # CPU 参考实现 (直接复用 ggml 的 quantize/dequantize/dot)
├── compare.cpp         # 逐元素比较: 绝对/相对容差, 输出首个差异位置与统计
├── metrics.cpp         # 计时 -> 带宽 GB/s 与利用率 % (对照 caps 常量带宽)
└── cases/
    ├── gemv_q4_1.cpp
    ├── gemm_f16.cpp
    ├── fa_decode_f16.cpp
    ├── rms_norm.cpp
    └── ...
```

### 23.2 用例模板

```cpp
REGISTER_CASE(gemv_q4_1) {
    for (int n : {2048, 11008, 151936}) {          // Qwen3 三类行数
        for (int k : {2048}) {
            run_case("gemv_q4_1", n, k, /*seed=*/42);
        }
    }
}
// run_case 内部:
//   1. 生成随机权重 (f32) -> 用 ggml 参考量化到 q4_1
//   2. 生成激活 f32; CPU 参考 dot 计算期望
//   3. 设备执行 kernel; 比较 (相对误差 <= 2e-5)
//   4. 计时 -> 打印: n, k, 耗时, 带宽 GB/s, 利用率 %
```

### 23.3 容差策略

| 场景 | 容差 |
|---|---|
| fp32 累加点积 (gemv/gemm) | 相对误差 <= 2e-5 |
| fa (softmax 类) | 相对误差 <= 1e-4 |
| fp16 中间量 | 按 kernel 定义放宽, 与 CPU 参考同口径 |

### 23.4 定位

- harness = 正确性 + 微基准（开发期主工具）；
- `test-backend-ops` = 全算子回归（集成期）；
- `llama-bench` = 端到端性能验收（与 8.1 节预算对照）。

## 24. 模块级实现顺序

| 阶段 | 开发顺序 | 每步验收 |
|---|---|---|
| M0 | 12 (内部头) -> 13 (caps) -> 22 (CMake + embed) -> ggml-ocl.cpp 空注册 (全 op 回退) | llama-cli 可跑; 日志打印 caps; 启动期 kernel 全量编译日志正常; test-backend-ops 全回退不崩 |
| M1 | 15 (kernels 加载框架) -> 16 (mem) -> 17 (exec 骨架) -> 23 (harness) -> 19 (P0 kernel: gemv/rms_norm_mul/cpy/eltwise/rows) | test-backend-ops 全绿; Qwen3-1.7B 可生成; harness 每个 P0 kernel 数值通过 |
| M2 | 18 (fusion) -> gemv_q4_1_n2048 特化 -> fa_decode_f16 -> 双队列重叠 | 4k ctx f16 KV decode >= 18 t/s; host <= 2ms |
| M3 | gemm_f16 + split-K -> fa_prefill_f16 -> KV q8_0 -> image1d 实测 -> 常量调参收敛 (harness 导出 CSV 人工导入) | prefill 达下界 80%; 32k+q8 KV decode >= 10 t/s |
| M4 | 全量化类型 -> MoE -> 真机 CI 每日回归 | 多模型多量化稳定 |

---

## 附录 A: Qwen3-1.7B 关键形状（待以模型配置确认）

| 项 | 参考值 |
|---|---|
| 层数 | 28 |
| hidden / head / kv_head / head_dim | 2048 / 16 / 8 / 128 (GQA 2:1) |
| FFN intermediate | 约 10816 (SwiGLU, 待确认) |
| vocab | 151936 |
| rope | neox 风格, theta=1e6 (待确认) |
| Q4_1 权重体积 | 约 1.06 GB (5 bits/weight) |
| KV 每 token 体积 (f16) | 28 层 x 4096 B = 112 KB/token; 4k ctx = 0.47 GB, 8k = 0.94 GB, 32k = 3.76 GB; q8_0 减半 |
| 显存支撑 (12 GB) | f16 KV 至约 64k 上下文; q8_0 KV 至约 128k 上下文 |

## 附录 B: Q4_1 块布局 (ggml-common.h:200)

```
typedef struct {
    half2 dm;              // d (scale) + m (min), 4 字节
    uint8_t qs[16];        // 32 x 4-bit nibbles, 16 字节
} block_q4_1;              // 共 20 字节 / 32 权重
// 解量化: x = d * q + m, q in [0,15]
// 注: 20 字节块 -> AoS 行内非 16B 对齐, SoA-DM 布局可恢复对齐 (第 6.4 节)
```

## 附录 C: 参考阅读 (ggml-opencl 对应位置)

| 主题 | 文件:行 |
|---|---|
| 能力查询式初始化 | ggml-opencl.cpp:5809 |
| 编译失败降级 (3 次重试) | ggml-opencl.cpp:1147 |
| FA 变体 (dk,dv) 调参表 (参考) | ggml-opencl.cpp:4527, fa_tune.h |
| SoA 权重拆分 (subbuffer 零拷贝) | ggml-opencl.cpp:7961 |
| image1d KV 包装 (参考, 低带宽硬件需实测) | ggml-opencl.cpp:18622 |
| 算子融合 | ggml-opencl.cpp:6952 |
| 全节点同步反例 (避免) | ggml-opencl.cpp:6939 |
| profiling 基础设施 | ggml-opencl.cpp:350 |
