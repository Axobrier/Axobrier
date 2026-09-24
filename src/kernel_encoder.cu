/*
 * Copyright 2026 Axobrier Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

#include "axobrier_core.h"

/* ─── Constants ───────────────────────────────────────────────────── */

#define AXO_ENC_WARP_SIZE  32

/* ─── Device Helpers ──────────────────────────────────────────────── */

__device__ __forceinline__ float warp_reduce_sum_enc(float val) {
    #pragma unroll
    for (int offset = AXO_ENC_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_enc(float val) {
    #pragma unroll
    for (int offset = AXO_ENC_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/* Fast GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
__device__ __forceinline__ float gelu_approx(float x) {
    const float c = 0.7978845608f;  /* sqrt(2/pi) */
    const float k = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x3)));
}

/* ─── Kernel 1: Embedding Lookup + LayerNorm ─────────────────────── */

__global__ void kernel_embed_layernorm(
    const int32_t*  __restrict__ token_ids,     /* [B × S]     */
    const __half*   __restrict__ embed_table,   /* [V × D]     */
    const float*    __restrict__ ln_gamma,      /* [D]         */
    const float*    __restrict__ ln_beta,       /* [D]         */
    __half*         __restrict__ output,        /* [B × S × D] */
    uint32_t        embed_dim,
    uint32_t        seq_len,
    uint32_t        vocab_size
) {
    const uint32_t s = blockIdx.x;   /* token position */
    const uint32_t b = blockIdx.y;   /* batch index    */

    const int32_t token_id = token_ids[b * seq_len + s];
    const uint32_t safe_id = (token_id >= 0 && (uint32_t)token_id < vocab_size)
                           ? (uint32_t)token_id : 0u;

    const __half* emb_row = embed_table + (size_t)safe_id * embed_dim;

    /* Shared memory for warp-level reduction */
    extern __shared__ char smem_raw[];
    float* smem_reduce = (float*)smem_raw;

    const uint32_t num_warps = (blockDim.x + AXO_ENC_WARP_SIZE - 1) / AXO_ENC_WARP_SIZE;
    const uint32_t warp_id   = threadIdx.x / AXO_ENC_WARP_SIZE;
    const uint32_t lane_id   = threadIdx.x % AXO_ENC_WARP_SIZE;

    /* Pass 1: Mean */
    float local_sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        local_sum += __half2float(emb_row[d]);
    }
    float warp_sum = warp_reduce_sum_enc(local_sum);
    if (lane_id == 0) smem_reduce[warp_id] = warp_sum;
    __syncthreads();

    float total_sum = 0.0f;
    if (threadIdx.x < num_warps) total_sum = smem_reduce[threadIdx.x];
    total_sum = warp_reduce_sum_enc(total_sum);
    if (threadIdx.x == 0) smem_reduce[0] = total_sum;
    __syncthreads();
    float mean = smem_reduce[0] / (float)embed_dim;

    /* Pass 2: Variance */
    float local_var = 0.0f;
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float val = __half2float(emb_row[d]) - mean;
        local_var += val * val;
    }
    float warp_var = warp_reduce_sum_enc(local_var);
    if (lane_id == 0) smem_reduce[warp_id] = warp_var;
    __syncthreads();

    float total_var = 0.0f;
    if (threadIdx.x < num_warps) total_var = smem_reduce[threadIdx.x];
    total_var = warp_reduce_sum_enc(total_var);
    if (threadIdx.x == 0) smem_reduce[0] = total_var;
    __syncthreads();
    float inv_std = rsqrtf(smem_reduce[0] / (float)embed_dim + 1e-5f);

    /* Pass 3: Normalize + affine transform */
    __half* out_row = output + ((size_t)b * seq_len + s) * embed_dim;
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float val = __half2float(emb_row[d]);
        float normed = (val - mean) * inv_std;
        float gamma = ln_gamma ? ln_gamma[d] : 1.0f;
        float beta  = ln_beta  ? ln_beta[d]  : 0.0f;
        float result = normed * gamma + beta;
        out_row[d] = __float2half(result);
    }
}

/* ─── Kernel 2a: QKV Projection (Coalesced & Shared Memory) ────────── */

__global__ void kernel_qkv_proj(
    const __half*  __restrict__ input,       /* [B × S × D]  */
    const __half*  __restrict__ qkv_proj,    /* [D × 3D]     */
    __half*        __restrict__ qkv_buf,     /* [B × S × 3D] */
    uint32_t       embed_dim,
    uint32_t       seq_len
) {
    const uint32_t s = blockIdx.x;
    const uint32_t b = blockIdx.y;

    extern __shared__ char smem_raw[];
    __half* smem_in = (__half*)smem_raw;

    const __half* in_row = input + ((size_t)b * seq_len + s) * embed_dim;
    __half*       qkv_row = qkv_buf + ((size_t)b * seq_len + s) * 3 * embed_dim;

    /* Load input token into shared memory */
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        smem_in[d] = in_row[d];
    }
    __syncthreads();

    const uint32_t total_cols = 3 * embed_dim;
    for (uint32_t col = threadIdx.x; col < total_cols; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t d = 0; d < embed_dim; ++d) {
            acc += __half2float(smem_in[d]) * __half2float(qkv_proj[d * total_cols + col]);
        }
        qkv_row[col] = __float2half(acc);
    }
}

/* ─── Kernel 2b: Multi-Head Attention (Per-Head Block) ─────────────── */

__global__ void kernel_attention_fused(
    const __half*   __restrict__ qkv_buf,        /* [B × S × 3D]    */
    float*          __restrict__ attn_logits,    /* [B × H × S × S] */
    __half*         __restrict__ attn_out,       /* [B × S × D]     */
    uint32_t        embed_dim,
    uint32_t        seq_len,
    uint32_t        num_heads
) {
    const uint32_t h = blockIdx.x;   /* head index   */
    const uint32_t b = blockIdx.y;   /* batch index  */
    const uint32_t s = threadIdx.x;  /* query position */

    if (s >= seq_len) return;

    const uint32_t head_dim = embed_dim / num_heads;
    const float scale = rsqrtf((float)head_dim);

    /* Q for this query token and head: length = head_dim */
    const __half* q_ptr = qkv_buf + ((size_t)b * seq_len + s) * 3 * embed_dim + h * head_dim;

    /* Pointer to attention logits for this query position */
    float* my_attn = attn_logits + ((size_t)b * num_heads + h) * (size_t)seq_len * seq_len
                     + (size_t)s * seq_len;

    /* Compute attention scores Q[s] · K[j]^T */
    float max_score = -INFINITY;
    for (uint32_t j = 0; j < seq_len; j++) {
        const __half* k_ptr = qkv_buf + ((size_t)b * seq_len + j) * 3 * embed_dim
                              + embed_dim + h * head_dim;
        float dot = 0.0f;
        for (uint32_t hd = 0; hd < head_dim; hd++) {
            dot += __half2float(q_ptr[hd]) * __half2float(k_ptr[hd]);
        }
        dot *= scale;
        my_attn[j] = dot;
        max_score = fmaxf(max_score, dot);
    }

    /* Softmax over sequence length */
    float sum_exp = 0.0f;
    for (uint32_t j = 0; j < seq_len; j++) {
        float e = expf(my_attn[j] - max_score);
        my_attn[j] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    for (uint32_t j = 0; j < seq_len; j++) {
        my_attn[j] *= inv_sum;
    }

    /* Weighted sum of V: result is head_dim values stored in attn_out */
    __half* out_slot = attn_out + ((size_t)b * seq_len + s) * embed_dim + h * head_dim;

    for (uint32_t hd = 0; hd < head_dim; hd++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < seq_len; j++) {
            const __half* v_ptr = qkv_buf + ((size_t)b * seq_len + j) * 3 * embed_dim
                                  + 2 * embed_dim + h * head_dim;
            acc += my_attn[j] * __half2float(v_ptr[hd]);
        }
        out_slot[hd] = __float2half(acc);
    }
}

/* ─── Kernel 2c: Output Projection + Residual ───────────────────────── */

__global__ void kernel_out_proj_residual(
    const __half*   __restrict__ input,       /* [B × S × D]  -  residual input    */
    const __half*   __restrict__ attn_out,    /* [B × S × D]  -  raw attention     */
    const __half*   __restrict__ out_proj,    /* [D × D]                         */
    __half*         __restrict__ output,      /* [B × S × D]  -  output buffer     */
    uint32_t        embed_dim,
    uint32_t        seq_len
) {
    const uint32_t s = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const size_t   row_off = ((size_t)b * seq_len + s) * embed_dim;

    extern __shared__ char smem_raw[];
    __half* smem_attn = (__half*)smem_raw;

    const __half* attn_row  = attn_out + row_off;
    const __half* input_row = input + row_off;
    __half*       out_row   = output + row_off;

    /* Load attention output into shared memory */
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        smem_attn[d] = attn_row[d];
    }
    __syncthreads();

    /* Compute projected result + residual */
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t d2 = 0; d2 < embed_dim; ++d2) {
            acc += __half2float(smem_attn[d2]) * __half2float(out_proj[d2 * embed_dim + d]);
        }
        acc += __half2float(input_row[d]);
        out_row[d] = __float2half(acc);
    }
}

/* ─── Kernel 3: Fast Coalesced GELU FFN + Residual ─────────────────── */

__global__ void kernel_ffn_gelu(
    const __half*   __restrict__ input,       /* [B × S × D]  */
    const __half*   __restrict__ up_proj,     /* [D × 4D]     */
    const __half*   __restrict__ down_proj,   /* [4D × D]     */
    __half*         __restrict__ output,      /* [B × S × D]  (can be == input) */
    uint32_t        embed_dim,
    uint32_t        ffn_dim,
    uint32_t        seq_len
) {
    const uint32_t s = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const size_t   row_off = ((size_t)b * seq_len + s) * embed_dim;

    /* Shared memory layout (Blackwell sm_120 offset alignment):
     * [0 .. smem_in_bytes)            smem_in (half)
     * [smem_align .. smem_align+smem_hidden_bytes) smem_hidden (float)
     */
    extern __shared__ char smem_raw[];
    size_t smem_in_bytes = embed_dim * sizeof(__half);
    size_t smem_align    = (smem_in_bytes + 3u) & ~3u;

    __half* smem_in    = (__half*)smem_raw;
    float*  smem_hidden = (float*)(smem_raw + smem_align);

    const __half* in_row  = input + row_off;
    __half*       out_row = output + row_off;

    /* Step 1: Load input row into shared memory */
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        smem_in[d] = in_row[d];
    }
    __syncthreads();

    /* Step 2: Compute up-projection to hidden activations + GELU */
    for (uint32_t f = threadIdx.x; f < ffn_dim; f += blockDim.x) {
        float up_val = 0.0f;
        for (uint32_t d = 0; d < embed_dim; ++d) {
            up_val += __half2float(smem_in[d]) * __half2float(up_proj[d * ffn_dim + f]);
        }
        smem_hidden[f] = gelu_approx(up_val);
    }
    __syncthreads();

    /* Step 3: Down-projection + residual addition */
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float residual = __half2float(smem_in[d]);
        float acc = 0.0f;
        for (uint32_t f = 0; f < ffn_dim; ++f) {
            acc += smem_hidden[f] * __half2float(down_proj[f * embed_dim + d]);
        }
        out_row[d] = __float2half(residual + acc);
    }
}

/* ─── Kernel 4: Masked Mean Pooling + L2 Normalization ───────────── */

__global__ void kernel_mean_pool_norm(
    const __half*   __restrict__ encoded,      /* [B × S × D]  */
    const int32_t*  __restrict__ token_ids,    /* [B × S]      */
    __half*         __restrict__ pooled,       /* [B × D]      */
    uint32_t        embed_dim,
    uint32_t        seq_len,
    int32_t         pad_token_id
) {
    const uint32_t b = blockIdx.x;

    extern __shared__ char pool_smem_raw[];
    float* smem_reduce = (float*)pool_smem_raw;

    const uint32_t num_warps = (blockDim.x + AXO_ENC_WARP_SIZE - 1) / AXO_ENC_WARP_SIZE;
    const uint32_t warp_id   = threadIdx.x / AXO_ENC_WARP_SIZE;
    const uint32_t lane_id   = threadIdx.x % AXO_ENC_WARP_SIZE;

    /* Count valid (non-pad) tokens */
    uint32_t local_count = 0;
    for (uint32_t s = threadIdx.x; s < seq_len; s += blockDim.x) {
        if (token_ids[b * seq_len + s] != pad_token_id) local_count++;
    }
    float warp_cnt = warp_reduce_sum_enc((float)local_count);
    if (lane_id == 0) smem_reduce[warp_id] = warp_cnt;
    __syncthreads();

    float total_count = 0.0f;
    if (threadIdx.x < num_warps) total_count = smem_reduce[threadIdx.x];
    total_count = warp_reduce_sum_enc(total_count);
    if (threadIdx.x == 0) smem_reduce[0] = total_count;
    __syncthreads();

    float valid_count = fmaxf(smem_reduce[0], 1.0f);
    float inv_count   = 1.0f / valid_count;

    /* Sum valid embeddings per dimension */
    __half* out_row = pooled + (size_t)b * embed_dim;
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t s = 0; s < seq_len; s++) {
            if (token_ids[b * seq_len + s] != pad_token_id) {
                sum += __half2float(encoded[((size_t)b * seq_len + s) * embed_dim + d]);
            }
        }
        out_row[d] = __float2half(sum * inv_count);
    }
    __syncthreads();

    /* L2-normalization */
    float local_sq = 0.0f;
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float v = __half2float(out_row[d]);
        local_sq += v * v;
    }
    float warp_sq = warp_reduce_sum_enc(local_sq);
    if (lane_id == 0) smem_reduce[warp_id] = warp_sq;
    __syncthreads();

    float total_sq = 0.0f;
    if (threadIdx.x < num_warps) total_sq = smem_reduce[threadIdx.x];
    total_sq = warp_reduce_sum_enc(total_sq);
    if (threadIdx.x == 0) smem_reduce[0] = total_sq;
    __syncthreads();

    float inv_norm = rsqrtf(smem_reduce[0] + 1e-8f);
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float v = __half2float(out_row[d]);
        out_row[d] = __float2half(v * inv_norm);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  HOST LAUNCHERS
 * ═══════════════════════════════════════════════════════════════════ */

extern "C" AxoStatus axo_launch_embed_layernorm(
    const int32_t*   token_ids,
    const void*      embed_table,
    const float*     ln_gamma,
    const float*     ln_beta,
    void*            output,
    uint32_t         batch_size,
    uint32_t         seq_len,
    uint32_t         embed_dim,
    uint32_t         vocab_size,
    cudaStream_t     stream
) {
    if (!token_ids || !embed_table || !output) return AXO_ERROR_INVALID_ARG;
    if (batch_size == 0 || seq_len == 0 || embed_dim == 0) return AXO_ERROR_INVALID_ARG;

    uint32_t block_dim = (embed_dim <= 256) ? embed_dim : 256;
    uint32_t num_warps = (block_dim + AXO_ENC_WARP_SIZE - 1) / AXO_ENC_WARP_SIZE;
    size_t   smem_size = num_warps * sizeof(float);

    dim3 grid(seq_len, batch_size);
    cudaGetLastError();

    kernel_embed_layernorm<<<grid, block_dim, smem_size, stream>>>(
        token_ids,
        reinterpret_cast<const __half*>(embed_table),
        ln_gamma,
        ln_beta,
        reinterpret_cast<__half*>(output),
        embed_dim, seq_len, vocab_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] kernel_embed_layernorm failed: %s\n", cudaGetErrorString(err));
        return AXO_ERROR_CUDA;
    }
    return AXO_SUCCESS;
}

extern "C" AxoStatus axo_launch_encoder_layer(
    const void*      input,           /* [B × S × D] fp16  -  read         */
    void*            output,          /* [B × S × D] fp16  -  write        */
    const void*      qkv_proj,
    const void*      out_proj,
    const void*      ffn_up,
    const void*      ffn_down,
    const float*     ln_gamma,
    const float*     ln_beta,
    void*            qkv_buf,
    void*            attn_logits,
    void*            attn_out,
    uint32_t         batch_size,
    uint32_t         seq_len,
    uint32_t         embed_dim,
    uint32_t         num_heads,
    uint32_t         ffn_mult,
    cudaStream_t     stream
) {
    if (!input || !output || !qkv_proj || !out_proj || !ffn_up || !ffn_down ||
        !qkv_buf || !attn_logits || !attn_out) {
        return AXO_ERROR_INVALID_ARG;
    }
    if (batch_size == 0 || seq_len == 0 || embed_dim == 0 || num_heads == 0) {
        return AXO_ERROR_INVALID_ARG;
    }
    if (embed_dim % num_heads != 0) return AXO_ERROR_INVALID_ARG;

    /* ── Step 1: QKV Projection ── */
    dim3 grid_tokens(seq_len, batch_size);
    uint32_t block_qkv = (embed_dim <= 256) ? embed_dim : 256;
    size_t   smem_qkv  = embed_dim * sizeof(__half);

    cudaGetLastError();
    kernel_qkv_proj<<<grid_tokens, block_qkv, smem_qkv, stream>>>(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(qkv_proj),
        reinterpret_cast<__half*>(qkv_buf),
        embed_dim, seq_len
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] kernel_qkv_proj failed: %s\n", cudaGetErrorString(err));
        return AXO_ERROR_CUDA;
    }

    /* ── Step 2: Multi-Head Attention ── */
    dim3 grid_heads(num_heads, batch_size);
    uint32_t block_attn = seq_len;

    kernel_attention_fused<<<grid_heads, block_attn, 0, stream>>>(
        reinterpret_cast<const __half*>(qkv_buf),
        reinterpret_cast<float*>(attn_logits),
        reinterpret_cast<__half*>(attn_out),
        embed_dim, seq_len, num_heads
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] kernel_attention_fused failed: %s\n", cudaGetErrorString(err));
        return AXO_ERROR_CUDA;
    }

    /* ── Step 3: Out-projection + Residual → writes to output ── */
    uint32_t block_proj = (embed_dim <= 256) ? embed_dim : 256;
    size_t   smem_proj  = embed_dim * sizeof(__half);

    kernel_out_proj_residual<<<grid_tokens, block_proj, smem_proj, stream>>>(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(attn_out),
        reinterpret_cast<const __half*>(out_proj),
        reinterpret_cast<__half*>(output),
        embed_dim, seq_len
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] kernel_out_proj_residual failed: %s\n", cudaGetErrorString(err));
        return AXO_ERROR_CUDA;
    }

    /* ── Step 4: GELU FFN + Residual → in-place on output ── */
    uint32_t ffn_dim = embed_dim * ffn_mult;
    uint32_t block_ffn = (embed_dim <= 256) ? embed_dim : 256;

    size_t smem_in_bytes = embed_dim * sizeof(__half);
    size_t smem_align    = (smem_in_bytes + 3u) & ~3u;
    size_t smem_ffn      = smem_align + ffn_dim * sizeof(float);

    kernel_ffn_gelu<<<grid_tokens, block_ffn, smem_ffn, stream>>>(
        reinterpret_cast<const __half*>(output),
        reinterpret_cast<const __half*>(ffn_up),
        reinterpret_cast<const __half*>(ffn_down),
        reinterpret_cast<__half*>(output),
        embed_dim, ffn_dim, seq_len
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] kernel_ffn_gelu failed: %s\n", cudaGetErrorString(err));
        return AXO_ERROR_CUDA;
    }

    return AXO_SUCCESS;
}

extern "C" AxoStatus axo_launch_mean_pool_norm(
    const void*      encoded,
    const int32_t*   token_ids,
    void*            pooled,
    uint32_t         batch_size,
    uint32_t         seq_len,
    uint32_t         embed_dim,
    int32_t          pad_token_id,
    cudaStream_t     stream
) {
    if (!encoded || !token_ids || !pooled) return AXO_ERROR_INVALID_ARG;
    if (batch_size == 0 || seq_len == 0 || embed_dim == 0) return AXO_ERROR_INVALID_ARG;

    uint32_t block_dim = (embed_dim <= 256) ? embed_dim : 256;
    uint32_t num_warps = (block_dim + AXO_ENC_WARP_SIZE - 1) / AXO_ENC_WARP_SIZE;
    size_t   smem_size = num_warps * sizeof(float);

    cudaGetLastError();
    kernel_mean_pool_norm<<<batch_size, block_dim, smem_size, stream>>>(
        reinterpret_cast<const __half*>(encoded),
        token_ids,
        reinterpret_cast<__half*>(pooled),
        embed_dim, seq_len, pad_token_id
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] kernel_mean_pool_norm failed: %s\n", cudaGetErrorString(err));
        return AXO_ERROR_CUDA;
    }
    return AXO_SUCCESS;
}

extern "C" AxoStatus axo_launch_encoder(
    const int32_t*   token_ids,
    const void*      embed_table,
    const void*      qkv_proj,
    const void*      out_proj,
    const void*      ffn_up,
    const void*      ffn_down,
    const float*     ln_gamma,
    const float*     ln_beta,
    void*            state_ping,
    void*            state_pong,
    void*            qkv_buf,
    void*            attn_logits,
    void*            attn_out,
    void*            pooled_embed,
    uint32_t         batch_size,
    uint32_t         seq_len,
    uint32_t         embed_dim,
    uint32_t         num_heads,
    uint32_t         vocab_size,
    uint32_t         ffn_mult,
    int32_t          pad_token_id,
    cudaStream_t     stream
) {
    /* Stage 1: Embed + LN into state_ping */
    AxoStatus s = axo_launch_embed_layernorm(
        token_ids, embed_table, ln_gamma, ln_beta,
        state_ping, batch_size, seq_len, embed_dim, vocab_size, stream
    );
    if (s != AXO_SUCCESS) return s;

    /* Stage 2: 1 Layer from ping to pong */
    s = axo_launch_encoder_layer(
        state_ping, state_pong,
        qkv_proj, out_proj, ffn_up, ffn_down,
        ln_gamma, ln_beta,
        qkv_buf, attn_logits, attn_out,
        batch_size, seq_len, embed_dim, num_heads, ffn_mult, stream
    );
    if (s != AXO_SUCCESS) return s;

    /* Stage 3: Pool from state_pong */
    return axo_launch_mean_pool_norm(
        state_pong, token_ids, pooled_embed,
        batch_size, seq_len, embed_dim, pad_token_id, stream
    );
}
