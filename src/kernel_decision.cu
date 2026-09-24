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

#define AXO_MAX_BLOCK_DIM   512
#define AXO_WARP_SIZE       32

#ifndef AXO_HASH_SEED_PRIME
#define AXO_HASH_SEED_PRIME 0x696c6f63636f7262ULL
#endif
#ifndef AXO_ARENA_GUARD_TAG
#define AXO_ARENA_GUARD_TAG 0x636f7262 /* 'b','r','o','c' */
#endif

/* Warp-level butterfly shuffle reduction primitives */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = AXO_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = AXO_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/*
 * Fused choice kernel: coalesced FP16 dot-product + Platt temperature scaling + block-wide softmax.
 * Stride pattern: choice matrix column-major [D x C] in FP16 gives 128-byte coalesced transactions.
 */
__global__ void kernel_decision_fused(
    const __half*   __restrict__ embeddings,
    const __half*   __restrict__ choice_matrix,
    float           platt_temp,
    float           platt_bias,
    float*          __restrict__ logits_out,
    float*          __restrict__ softmax_out,
    uint32_t        embed_dim,
    uint32_t        num_choices
) {
    const uint32_t bid    = blockIdx.x;
    const uint32_t cid    = threadIdx.x;
    const uint32_t active = (cid < num_choices) ? 1 : 0;

    const uint32_t num_warps = (blockDim.x + AXO_WARP_SIZE - 1) / AXO_WARP_SIZE;
    const uint32_t wid       = threadIdx.x / AXO_WARP_SIZE;
    const uint32_t lane      = threadIdx.x % AXO_WARP_SIZE;

    extern __shared__ char smem_raw[];
    __half* smem_embed = (__half*)smem_raw;

    /* 4-byte memory alignment boundary for float warp accumulator */
    const size_t smem_embed_bytes = embed_dim * sizeof(__half);
    const size_t smem_align = (smem_embed_bytes + 3u) & ~3u;
    float* smem_warp = (float*)(smem_raw + smem_align);

    const __half* emb_row = embeddings + (size_t)bid * embed_dim;
    for (uint32_t i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        smem_embed[i] = emb_row[i];
    }
    __syncthreads();

    /* Dot product */
    float dot = 0.0f;
    if (active) {
        for (uint32_t d = 0; d < embed_dim; ++d) {
            const float e = __half2float(smem_embed[d]);
            const float w = __half2float(choice_matrix[d * num_choices + cid]);
            dot += e * w;
        }
    }

    /* Platt temperature calibration: logit = dot / T + bias */
    const float logit = active ? (dot / platt_temp + platt_bias) : -INFINITY;

    if (active) {
        logits_out[(size_t)bid * num_choices + cid] = logit;
    }

    /* Three-pass cooperative softmax: warp shuffle max reduction */
    const float warp_max = warp_reduce_max(logit);
    if (lane == 0) smem_warp[wid] = warp_max;
    __syncthreads();

    float block_max = -INFINITY;
    if (threadIdx.x < num_warps) block_max = smem_warp[threadIdx.x];
    block_max = warp_reduce_max(block_max);
    if (threadIdx.x == 0) smem_warp[0] = block_max;
    __syncthreads();
    block_max = smem_warp[0];

    /* Exponentiation & warp shuffle sum reduction */
    const float exp_val = active ? expf(logit - block_max) : 0.0f;
    const float warp_sum = warp_reduce_sum(exp_val);
    if (lane == 0) smem_warp[wid] = warp_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (threadIdx.x < num_warps) block_sum = smem_warp[threadIdx.x];
    block_sum = warp_reduce_sum(block_sum);
    if (threadIdx.x == 0) smem_warp[0] = block_sum;
    __syncthreads();
    block_sum = smem_warp[0];

    /* Final calibrated probability normalization */
    if (active) {
        softmax_out[(size_t)bid * num_choices + cid] = exp_val / block_sum;
    }
}

extern "C" AxoStatus axo_launch_decision_kernel(
    const void*    embeddings,
    const void*    choice_matrix,
    float          platt_temp,
    float          platt_bias,
    void*          logits_scratch,
    void*          softmax_scratch,
    uint32_t       batch_size,
    uint32_t       embed_dim,
    uint32_t       num_choices,
    cudaStream_t   stream
) {
    if (!embeddings || !choice_matrix || !logits_scratch || !softmax_scratch) {
        return AXO_ERROR_INVALID_ARG;
    }
    if (batch_size == 0 || embed_dim == 0 || num_choices == 0) {
        return AXO_ERROR_INVALID_ARG;
    }
    if (num_choices > AXO_MAX_BLOCK_DIM) {
        return AXO_ERROR_CARDINALITY_OVERFLOW;
    }

    if (fabsf(platt_temp) < 1e-8f) platt_temp = 1.0f;

    uint32_t block_dim = ((num_choices + AXO_WARP_SIZE - 1) / AXO_WARP_SIZE) * AXO_WARP_SIZE;
    if (block_dim > AXO_MAX_BLOCK_DIM) block_dim = AXO_MAX_BLOCK_DIM;

    const dim3 grid(batch_size, 1, 1);
    const dim3 block(block_dim, 1, 1);

    const uint32_t num_warps  = (block_dim + AXO_WARP_SIZE - 1) / AXO_WARP_SIZE;
    const size_t   smem_embed = embed_dim * sizeof(__half);
    const size_t   smem_align = (smem_embed + 3u) & ~3u;
    const size_t   smem_warp  = num_warps * sizeof(float);
    const size_t   smem_total = smem_align + smem_warp;

    kernel_decision_fused<<<grid, block, smem_total, stream>>>(
        reinterpret_cast<const __half*>(embeddings),
        reinterpret_cast<const __half*>(choice_matrix),
        platt_temp,
        platt_bias,
        reinterpret_cast<float*>(logits_scratch),
        reinterpret_cast<float*>(softmax_scratch),
        embed_dim,
        num_choices
    );

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] kernel launch failed: %s\n", cudaGetErrorString(err));
        return AXO_ERROR_CUDA;
    }

    return AXO_SUCCESS;
}
