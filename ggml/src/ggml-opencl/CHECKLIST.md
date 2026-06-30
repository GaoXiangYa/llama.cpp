# OpenCL 算子练习清单 — Qwen3 FP32

按难度递进排列，每个阶段实现一个算子并替换官方实现。

**开关方式**：设置环境变量 `GGML_OPENCL_MY_KERNELS=1`，所有已注册的练习 kernel 都会被加载并使用。

**文件位置**：所有练习 `.cl` 文件放在 `ggml/src/ggml-opencl/kernels/` 下，命名为 `my_<op>.cl`。

**注册步骤**（每个算子都要做）：
1. 在 `ggml-opencl.cpp` 的 `ggml_backend_opencl_context` 结构体中添加 `cl_program` + `cl_kernel` 字段
2. 在 `load_cl_kernels()` 中添加 `#ifdef GGML_OPENCL_MY_KERNELS` 加载块 ── 更推荐复用 `use_my_kernels` 开关
3. 在对应的 host dispatch 函数（如 `ggml_cl_add`）中添加对 `my_kernel_*` 的分发
4. 在 `CMakeLists.txt` 的 `GGML_OPENCL_KERNELS` 列表中添加 `my_<op>`

---

## Stage 1: `ggml_add` — 通用张量加法

**数学**：`dst[i] = src0[i] + src1[i_broadcast]`

**Qwen3 用途**：残差连接（每层 2 次）+ bias 加法

**文件**：`kernels/my_add.cl`（已创建模板）

**Kernel 签名**：`kernel void my_kernel_add(global char * src0, ulong offset0, ...)` — 30 参数

**工作组映射**：
| OpenCL 维度 | 含义 |
|-------------|------|
| gid(0) | dst dim 1 索引 |
| gid(1) | dst dim 2 索引 |
| gid(2) | dst dim 3 索引 |
| lid(0) | dim 0 上的循环索引（步长 lsize(0)） |

**难度**：⭐ | **代码量**：~20 行 | **已就绪**：✅ 模板已创建

**注册要点**：已完成（结构体字段、加载代码、分发逻辑均已添加）

---

## Stage 2: `ggml_mul` — 逐元素乘法

**数学**：`dst[i] = src0[i] * src1[i_broadcast]`

**Qwen3 用途**：gate_s / up_s / down_s 输出缩放（`wo_s` 乘法在 `build_lora_mm` 中）

**文件**：`kernels/my_mul.cl`

**Kernel 签名**：`kernel void my_kernel_mul(global char * src0, ulong offset0, global char * src1, ulong offset1, global char * dst, ulong offsetd, int ne00, int ne01, int ne02, int ne03, ulong nb00, ulong nb01, ulong nb02, ulong nb03, int ne10, int ne11, int ne12, int ne13, ulong nb10, ulong nb11, ulong nb12, ulong nb13, int ne0, int ne1, int ne2, int ne3, ulong nb0, ulong nb1, ulong nb2, ulong nb3)` — 30 参数

**Host 分发函数**：`ggml_cl_mul`（参照 `ggml_cl_add` 的模式改写）

**工作组**：同 add

**难度**：⭐ | **代码量**：~20 行

**练习要点**：与 add 结构完全一致，只改运算符号。巩固 OpenCL 索引模型。

**注册改动**：
- 结构体添加：`cl_program program_my_mul; cl_kernel my_kernel_mul;`
- 加载块：参照 `my_add` 模式，源文件 `my_mul.cl`，kernel 名 `my_kernel_mul`
- 分发：`ggml_cl_mul` 中 F32 非广播路径，`kernel = use_my_kernels ? my_kernel_mul : kernel_mul`

---

## Stage 3: `ggml_rms_norm` — RMS 归一化

**数学**：`dst[i] = src[i] / sqrt(mean(src^2) + eps) * weight`

**Qwen3 用途**：每层 3 次 RMS Norm（attn_norm、q_norm/k_norm、ffn_norm）+ 最终 output_norm

**文件**：`kernels/my_rms_norm.cl`

**Kernel 签名**：
```
kernel void my_kernel_rms_norm(
    global void * src0, ulong offset0,
    global float * dst, ulong offsetd,
    int ne00, int ne01, int ne02, int ne03,
    ulong nb01, ulong nb02, ulong nb03,
    float eps,
    local float * sum
)
```
— 13 参数（含 1 个 local memory buffer）

**工作组映射**：
- `gid(0)` = 行索引（dim 1），`gid(1)` = dim 2，`gid(2)` = dim 3
- `lid(0)` = dim 0 上的局部索引

**核心算法**（两阶段归约）：
1. **Phase 1**：每个 work-item 累加部分 `sum_sq`，存到 local memory
2. **barrier + reduction tree**：local memory 内归约得到行均值
3. **Phase 2**：每个 work-item 用 `1/sqrt(mean + eps)` 归一化该行

**难度**：⭐⭐ | **代码量**：~60 行

**核心学习点**：
- local memory 的使用（`__local float *`）
- work-group barrier（`barrier(CLK_LOCAL_MEM_FENCE)`）
- reduction tree 模式
- subgroup 优化（可选）

**Host 分发函数**：`ggml_cl_rms_norm`
- 注意：arg 12 的 local memory size = `sizeof(float) * nth / sgs`，参数传 `NULL` + size

**注册改动**：
- 结构体添加：`cl_program program_my_rms_norm; cl_kernel my_kernel_rms_norm;`
- 加载块 + 分发均参照 add 模式

---

## Stage 4: `ggml_swiglu_split` — 融合 SiLU+Multiply

**数学**：`dst[i] = silu(gate[i]) * up[i]`，其中 `silu(x) = x / (1 + e^(-x))`

**Qwen3 用途**：FFN 中的 SwiGLU 激活（gate ⊙ up → 送 down 投影）

**文件**：`kernels/my_swiglu.cl`

**Kernel 签名**：
```
__kernel void my_kernel_swiglu(
    __global char * src0, ulong offset0,       // gate tensor
    __global char * src1, ulong offset1,       // up tensor
    __global char * dst,  ulong offsetd,
    ulong nb01,     // gate 的 dim-1 stride（字节）
    ulong nb11,     // up 的 dim-1 stride（字节）
    int ne0,        // 每行元素数（= n_ff / 2）
    ulong nb1,      // dst 的 dim-1 stride
    int ne00_off,   // gate 的 dim-1 偏移（通常为 0）
    int ne10_off    // up 的 dim-1 偏移（通常为 0）
)
```
— 12 参数

**工作组映射**：2D
- `gid(0)` = 行内元素索引（dim 0）
- `gid(1)` = 行索引（dim 1），`gid(2)` = dim 2

**核心算法**：
```
float g = *(float*)(src0_ptr + i0 * sizeof(float));
float u = *(float*)(src1_ptr + i0 * sizeof(float));
dst[i] = (g / (1.f + exp(-g))) * u;
```
（使用 `native_divide` + `native_exp` 可提升性能）

**难度**：⭐⭐ | **代码量**：~30 行

**核心学习点**：
- 算子融合的意义（避免中间张量的显存往返）
- `native_exp` vs `exp` 的性能/精度权衡
- 2D 工作组索引

**Host 分发函数**：`ggml_cl_glu`
- 在 `GGML_GLU_OP_SWIGLU` 分支中，当 `dst->type == GGML_TYPE_F32` 时替换

**注册改动**：
- 结构体添加：`cl_program program_my_swiglu; cl_kernel my_kernel_swiglu;`
- 分发：`ggml_cl_glu` 中 `case GGML_GLU_OP_SWIGLU` 的 F32 路径

---

## Stage 5: `ggml_get_rows` — 按索引查表（Gather）

**数学**：`dst[i, j] = src0[indices[i], j]`

**Qwen3 用途**：Token 嵌入查表（`tok_embd[token_ids]`）

**文件**：`kernels/my_get_rows.cl`

**Kernel 签名**：
```
kernel void my_kernel_get_rows_f32(
    global void * src0, ulong offset0,         // 嵌入表 [vocab_size, n_embd]
    global int  * src1, ulong offset1,         // 索引 [n_tokens]
    global float * dst,  ulong offsetd,
    int ne00,           // n_embd（每行元素数）
    ulong nb01, ulong nb02, ulong nb03,       // src0 strides
    int ne10,           // n_tokens（索引数量）
    ulong nb10, ulong nb11, ulong nb12,       // src1 strides
    ulong nb1, ulong nb2, ulong nb3           // dst strides
)
```
— 17 参数

**工作组映射**：2D
- `gid(0)` = dst 的行内元素索引（dim 0），循环遍历 ne00
- `gid(1)` = dst 的行索引（dim 1，= token 索引）
- `gid(2)` = dst 的 dim 2

**核心算法**：
```
int row = src1[i01];                          // 读取索引
float val = *(float*)(src0 + row*nb01 + i0*nb00);  // 查表
dst[i0 + i01*nb1/sizeof(float)] = val;
```

**边界检查**：`row >= 0 && row < ne01`（防止 OOB）

**难度**：⭐⭐ | **代码量**：~25 行

**核心学习点**：
- 间接内存访问（gather）模式
- 性能受限于全局内存延迟
- 边界检查的重要性

**Host 分发函数**：`ggml_cl_get_rows`
- 当 `src0->type == GGML_TYPE_F32` 时分发

**注册改动**：
- 结构体添加：`cl_program program_my_get_rows; cl_kernel my_kernel_get_rows_f32;`
- 分发：`ggml_cl_get_rows` 中 F32 路径

---

## Stage 6: `ggml_soft_max_ext` — Softmax with Mask

**数学**：
```
m = max(scores - mask)
s = exp(scores - mask - m)
dst = s * scale / sum(s)
```
其中 mask 可选，scale = 1/√d_k

**Qwen3 用途**：Attention 中的 softmax（非 flash-attn 路径）

**文件**：`kernels/my_softmax.cl`

**Kernel 签名**：
```
kernel void my_kernel_soft_max(
    global char * src0, ulong offset0,         // QK^T scores
    global char * src1, ulong offset1,         // mask（可为空）
    global char * src2, ulong offset2,         // 保留（可为空）
    global char * dst,  ulong offsetd,
    int ne00,                                  // 每行元素数
    ulong nb01, ulong nb02, ulong nb03,       // src0 strides
    int ne12, int ne13,                        // mask dims
    ulong nb11, ulong nb12, ulong nb13,       // mask strides
    ulong nb1, ulong nb2, ulong nb3,          // dst strides
    float scale,                               // 1/√d_k
    float max_bias,                            // alibi 偏置
    float m0, float m1,                        // 保留
    int n_head_log2                            // 保留
)
```
— 25 参数

**工作组映射**：
- `gid(0)` = 行索引（dim 1），每行独立（不同 token）
- `gid(1)` = dim 2，`gid(2)` = dim 3
- `lid(0)` = 行内元素的局部索引，循环遍历整行

**核心算法**（Online Softmax / 三阶段归约）：
1. **Phase 1 — find max**：遍历行，找最大值（减去 mask）
2. **barrier + reduction**：local memory 归约得全局 max
3. **Phase 2 — exp + sum**：`exp(x - max)` 并累加 sum
4. **barrier + reduction**：归约得全局 sum
5. **Phase 3 — normalize**：`(exp_val / sum) * scale`

**数值稳定性**：先减 mask 再减 max，避免 exp 溢出。

**难度**：⭐⭐⭐ | **代码量**：~80 行

**核心学习点**：
- Online softmax 算法防止数值溢出
- 多阶段归约模式
- mask 的应用（causal mask 下三角）
- local memory barrier 的正确使用

**Host 分发函数**：`ggml_cl_soft_max`
- 注意：当 `ne00 % 4 == 0` 时官方会用向量化版本 `kernel_soft_max_4`，练习阶段可以只处理 `ne00 % 4 != 0` 的情况

**注册改动**：
- 结构体添加：`cl_program program_my_softmax; cl_kernel my_kernel_soft_max;`
- 分发：`ggml_cl_soft_max` 中 F32 非向量化路径

---

## Stage 7: `ggml_rope_ext` — 旋转位置编码

**数学**（NeoX 风格）：
```
对于 dim i < n_dims/2:
  θ = pos * freq_base^(-2i/n_dims) * freq_scale
  freq = θ，经 ext_factor/attn_factor 调整
  cos = cos(freq), sin = sin(freq)
  x, y 为一对相邻半维的元素
  dst[x] = x*cos - y*sin
  dst[y] = y*cos + x*sin
```

**Qwen3 用途**：Q 和 K 的 RoPE 编码（每层 2 次 `ggml_rope_ext`）
Qwen3 使用 NeoX-style RoPE（半维配对）。

**文件**：`kernels/my_rope.cl`

**Kernel 签名**（NeoX 版本）：
```
kernel void my_kernel_rope_neox_f32(
    global void * src0, ulong offset0,         // Q 或 K tensor
    global int  * src1, ulong offset1,         // pos 数组
    global float * src2, ulong offset2,        // freq_factors（可为空）
    global float * dst,  ulong offsetd,
    int ne00, int ne01, int ne02, int ne03,   // src0 dims
    ulong nb00, ulong nb01, ulong nb02, ulong nb03,
    int ne0, int ne1, int ne2, int ne3,        // dst dims
    ulong nb0, ulong nb1, ulong nb2, ulong nb3,
    int n_past,                                // 历史 token 数
    int n_dims,                                // 参与旋转的维度数
    int n_ctx_orig,                            // 原始上下文长度
    float freq_base,                           // RoPE base（如 1000000）
    float freq_scale,                          // 频率缩放
    float ext_factor,                          // NTK 外推因子
    float attn_factor,                         // 注意力因子
    float beta_fast, float beta_slow           // YaRN 参数
)
```
— 33 参数

> **同时实现 `my_kernel_rope_norm_f32`**（标准 RoPE）——签名完全相同，仅配对方式不同：
> - NeoX: 前半维和后半维的对应位置配对（i 和 i+n_dims/2）
> - Norm: 相邻位置配对（2i 和 2i+1）

**工作组映射**：2D
- `gid(0)` = head × batch 的组合索引
- `lid(0)` = 行内元素索引，循环遍历 n_dims/2

**核心算法**：
```
pos = src1[i_batch];  // 当前 token 的位置
for each (i_half in [0, n_dims/2)):
    theta = pos * pow(freq_base, -2*i_half/n_dims) * freq_scale
    theta = adjust_ntk(theta, ext_factor, attn_factor)
    cosv = cos(theta), sinv = sin(theta)
    // NeoX: i0 = i_half, i1 = i_half + n_dims/2
    // Norm: i0 = 2*i_half, i1 = 2*i_half + 1
    x = src[i0], y = src[i1]
    dst[i0] = x*cosv - y*sinv
    dst[i1] = y*cosv + x*sinv
```

**NTK 调整逻辑**：
```
if ext_factor > 0:
    scale = max(0, min(1, (ext_factor - ratio) / attn_factor))
    theta *= scale
```

**难度**：⭐⭐⭐ | **代码量**：~70 行

**核心学习点**：
- RoPE 的数学原理（旋转矩阵、频率调制）
- NeoX vs Norm 两种配对方式的区别
- NTK-aware 外推的频率调整
- GPU 上三角函数（`cos`/`sin`）的性能

**Host 分发函数**：`ggml_cl_rope`
- 选择 NeoX 或 Norm kernel 取决于 `rope_type` 参数

**注册改动**：
- 结构体添加：`cl_program program_my_rope; cl_kernel my_kernel_rope_norm_f32; cl_kernel my_kernel_rope_neox_f32;`
- 分发：`ggml_cl_rope` 中根据 `is_neox` 选择对应 kernel

---

## Stage 8: `ggml_mul_mat` (F32×F32) — 矩阵乘法

**数学**：`C = A × B`，其中 A 是 weight [n_cols, n_rows]，B 是 input [n_cols, n_tokens]

**Qwen3 用途**：所有线性投影（每层 ~6 次 `ggml_mul_mat`——wqkv、wo、gate、up、down、output）

**文件**：`kernels/my_mul_mat.cl`

**Kernel 签名**：
```
kernel void my_kernel_mul_mat_f32_f32(
    global char * src0, ulong offset0,         // weight [ne00, ne01]
    global char * src1, ulong offset1,         // input [ne10, ne11]
    global float * dst,  ulong offsetd,
    int ne00, int ne01, int ne02,              // src0 dims
    ulong nb00, ulong nb01, ulong nb02, ulong nb03,
    int ne10, int ne11, int ne12,              // src1 dims
    ulong nb10, ulong nb11, ulong nb12, ulong nb13,
    int ne0, int ne1,                          // dst dims
    int r2, int r3                             // 保留
)
```
— 24 参数

**分两阶段实现**：

### 8a: 简单向量版（mat-vec）— `ne11 == 1, ne01 == 1`
```
dst[i0] = sum_j src0[j] * src1[j, i0]   // 点积
```

### 8b: Tile 分块版（GEMM）— 通用
- 将 B 矩阵分块加载到 local memory
- 每个 work-item 累加部分点积
- 使用 register blocking 减少全局内存访问

**工作组映射**（tile 版，2D）：
- `gid(0)` = 输出列索引（dst dim 1）
- `gid(1)` = 输出行索引（dst dim 0）
- `lid(0)`, `lid(1)` = tile 内的局部索引

**难度**：⭐⭐⭐⭐ | **代码量**：~150 行

**核心学习点**：
- GPU 矩阵乘法的 tile 分块策略
- local memory 作为手动 cache
- 循环展开和寄存器分块
- 内存合并访问（coalesced access）
- 这是 GPU 编程中最重要的 kernel pattern

**Host 分发函数**：`ggml_cl_mul_mat`
- 注意：`ggml_cl_mul_mat` 有多条分发路径（GEMM、Adreno、量化等），练习阶段只需替换 `F32×F32` 的 `kernel_mul_mat_f32_f32` 调用

**推荐分步**：
1. 先写 naive 点积（每行一个 work-item）
2. 加入 tile 分块
3. 加入 register blocking（每 work-item 计算多个输出元素）

**注册改动**：
- 结构体添加：`cl_program program_my_mul_mat; cl_kernel my_kernel_mul_mat_f32_f32;`
- 分发：`ggml_cl_mul_mat` 中 `kernel_mul_mat_f32_f32` 的替换

---

## Stage 9: `ggml_flash_attn_ext` — Flash Attention

**数学**：融合 QK^T + softmax + PV 的单个 kernel

```
S = Q × K^T * scale           // scores
P = softmax(S, mask)          // attention weights
O = P × V                      // output
```
整个过程在单次 kernel 调用中完成，避免实例化中间张量 S 和 P。

**Qwen3 用途**：Attention 计算（`cparams.flash_attn == true` 时）

**文件**：`kernels/my_flash_attn.cl`

**Kernel 签名**：
```
__kernel void my_flash_attn_f32(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void, ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, ulong sinks_offset
)
```
— 40 参数

**核心算法**（Online Softmax + Tiling）：

```
对每个 query block B_r（沿 n_q 分块）:
  初始化: O = 0, l = 0, m = -inf
  对每个 key block B_c（沿 n_kv 分块）:
    S = Q_block × K_block^T * scale
    if causal: 对 S[i,j] 当 i < j 时置 -inf
    m_new = max(m, row_max(S))
    P = exp(S - m_new)
    l_new = exp(m - m_new) * l + row_sum(P)
    O = exp(m - m_new) * O + P × V_block
    m = m_new, l = l_new
  O = O / l
```

**难度**：⭐⭐⭐⭐⭐ | **代码量**：~300 行

**核心学习点**：
- Online softmax（防止数值溢出 + 流式处理）
- Tile-based Flash Attention 的核心思想
- 大规模 kernel 的寄存器压力和 occupancy 平衡
- local memory 容量约束下的 tile 大小选择
- Causal mask 的 tile 内处理

**Host 分发函数**：`ggml_cl_flash_attn`
- 官方有多 tile-size 映射表（`std::map<std::pair<int,int>, cl_kernel>`），练习阶段先实现单一 tile size

**推荐分步**：
1. 固定 tile size（如 BR=32, BC=32），先实现非 causal 版本
2. 加入 causal mask 支持
3. 优化 tile size 和循环展开
4. 实现 n_q==1 的快速路径（`flash_attn_f32_q1`）

**注册改动**：
- 结构体添加：`cl_program program_my_flash_attn;`
  - 不用 `std::map`，先只加 `cl_kernel my_kernel_flash_attn_f32;`
- 分发：`ggml_cl_flash_attn` 中 F32 路径替换

---

## 学习里程碑

完成某个 Stage 后，Qwen3 的部分功能即可使用你的 kernel 运行：

| 完成 | 可用功能 |
|------|---------|
| 1-2 | 残差、缩放等辅助操作由 GPU 执行 |
| 1-3 | 基本前向结构完整（Norm + Add + Mul） |
| 1-4 | FFN 的 SwiGLU 激活由 GPU 执行 |
| 1-5 | Token 嵌入查表由 GPU 执行 |
| 1-6 | Attention 中的 softmax（非 flash 模式）由 GPU 执行 |
| 1-7 | RoPE 位置编码由 GPU 执行 |
| **1-8** | **所有线性投影由 GPU 执行 — Qwen3 完整 GPU 推理** |
| 1-9 | Flash Attention 模式由 GPU 执行（更快） |

**Stage 8 是关键里程碑**——完成后 Qwen3 就可以完全在你的 GPU kernel 上运行。

---

## 调试技巧

1. **数值验证**：不加 `GGML_OPENCL_MY_KERNELS` 跑一次（官方 kernel），加环境变量跑一次（你的 kernel），对比 logits 是否一致
2. **中间检查**：在 CL kernel 中用 `printf` 调试（需要 OpenCL 2.0+）
3. **精度容差**：`native_exp`/`native_cos` 等近似函数可能产生 ~1e-4 级别的差异，属正常
4. **CL 编译错误**：`clBuildProgram` 失败时会打印 build log，注意检查
