kernel void kernel_arise_rms_norm(
    // === input (x): shape [n_embd, n_tokens, n_heads, n_batches] ===
    global char * x,
    ulong         offset_x,
    // === output (y): 与 x 同 shape ===
    global char * y,
    ulong         offset_y,
    int           n_embd,
    // strides（单位：float 元素数）—— dim 0 始终连续，stride_d0 = 1，不传
    int           stride_token,    // nb01 / sizeof(float) — 行间步长
    int           stride_head,     // nb02 / sizeof(float)
    int           stride_batch,    // nb03 / sizeof(float)
    // === params ===
    float eps,
    // === local memory: 归约缓冲区 ===
    local float * sum) {
    x = x + offset_x;
    y = y + offset_y;

    const int n_tokens = get_group_id(0);
    const int n_head = get_group_id(1);
    const int n_batch = get_group_id(2);
    const int lsz = get_local_size(0);
    const int lid = get_local_id(0);
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int num_sg = get_num_sub_groups();

    global float4* input = (global float4*)(x + n_tokens * stride_token + n_head * stride_head + n_batch * stride_batch);
    global float4* output = (global float4*)(y + n_tokens * stride_token + n_head * stride_head + n_batch * stride_batch);

    float lsum = 0;
    for (int i = lid; i < n_embd; i += lsz) {
        float4 val = input[n_tokens * n_embd + i];
        lsum += (val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
    }

    float group_sum = sub_group_reduce_add(lsum);
    if(sg_lid == 0) {
        sum[sg_id] = group_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg_id == 0) {
        float x = (sg_lid < num_sg) : sum[sg_lid] : 0.0f;
        x = sub_group_reduce_add(x);
        if (sg_lid == 0) {
            sum[0] = x;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float global_sum = sum[0];
    float rms = 1.0f / native_rsqrt(global_sum / n_embd + eps);
    float4 rms4 = {rms, rms, rms, rms};

    for (int i = lid; i < n_embd; i += lsz) {
        float4 val = input[n_tokens * n_embd + i];
        output[n_tokens * n_embd + i] = input * rms4;
    }
}