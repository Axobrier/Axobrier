/*
 * Copyright 2026 Axobrier Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "axobrier_core.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

/* MSVC uses _alloca from <malloc.h>; POSIX uses alloca from <alloca.h> */
#ifdef _WIN32
    #include <malloc.h>
    #define axo_alloca _alloca
#else
    #include <alloca.h>
    #define axo_alloca alloca
#endif

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

/* ─── Alignment Helpers ───────────────────────────────────────────── */

#define AXO_DEFAULT_ALIGNMENT  256u   /* 256 bytes  -  coalescing-friendly */
#define AXO_MAX_LAYERS         16u    /* Hard cap on encoder layers      */

static inline size_t axo_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/* ─── Arena Data Structures ───────────────────────────────────────── */

typedef struct AxoArena {
    void*     base;          /**< Device pointer to arena start                */
    size_t    capacity;      /**< Total arena size in bytes                    */
    size_t    offset;        /**< Current bump offset (next free byte)         */
    size_t    scratch_start; /**< Offset where the scratchpad zone begins      */
    size_t    alignment;     /**< Minimum sub-allocation alignment             */
    uint32_t  guard_tag;     /**< Arena integrity tag                          */
} AxoArena;

/* ─── Per-Layer Weight Pointers ───────────────────────────────────── */

typedef struct AxoLayerWeights {
    void*  qkv_proj;        /**< [D × 3D] fp16  -  QKV projection             */
    void*  out_proj;        /**< [D × D]  fp16  -  output projection           */
    void*  ffn_up;          /**< [D × 4D] fp16  -  FFN up-projection (GELU)    */
    void*  ffn_down;        /**< [4D × D] fp16  -  FFN down-projection         */
    void*  ln_gamma;        /**< [D]      fp32  -  LayerNorm gamma             */
    void*  ln_beta;         /**< [D]      fp32  -  LayerNorm beta              */
} AxoLayerWeights;

/* ─── Full Weight Region Pointers ─────────────────────────────────── */

typedef struct AxoWeightPtrs {
    void*            embed_table;    /**< [V × D] fp16  -  token embedding table     */
    AxoLayerWeights  layers[AXO_MAX_LAYERS];  /**< Per-layer encoder weights       */
    void*            choice_matrix;  /**< [D × C]  fp16  -  choice routing weights   */
    float*           platt_temp;     /**< Scalar   fp32  -  Platt temperature        */
    float*           platt_bias;     /**< Scalar   fp32  -  Platt bias               */
} AxoWeightPtrs;

/* ─── Scratchpad Region Pointers ──────────────────────────────────── */

typedef struct AxoScratchPtrs {
    void*  state_ping;      /**< [B × S × D]   fp16   -  ping buffer          */
    void*  state_pong;      /**< [B × S × D]   fp16   -  pong buffer          */
    void*  qkv_buf;         /**< [B × S × 3D]  fp16   -  QKV intermediate     */
    void*  attn_logits;     /**< [B × H × S × S] fp32  -  attention scores    */
    void*  attn_out;        /**< [B × S × D]   fp16   -  attention output      */
    void*  pooled_embed;    /**< [B × D]       fp16   -  pooled embedding      */
    void*  choice_logits;   /**< [B × C]       fp32   -  raw logits            */
    void*  softmax_out;     /**< [B × C]       fp32   -  calibrated probs      */
} AxoScratchPtrs;

/* ─── Full Context Definition ─────────────────────────────────────── */

struct AxoContext {
    AxoConfig       config;
    AxoArena        arena;
    AxoWeightPtrs   weights;
    AxoScratchPtrs  scratch;
    cudaStream_t    stream;
    int             weights_loaded;

    /* Host-cached Platt parameters (read once at weight load, zero D2H on hot-path) */
    float           platt_temp_cached;
    float           platt_bias_cached;

    /* CUDA events for kernel latency measurement */
    cudaEvent_t     ev_start;
    cudaEvent_t     ev_stop;

    /* mmap bookkeeping for cleanup */
    void*           mmap_host_ptr;
    size_t          mmap_size;
#ifdef _WIN32
    HANDLE          mmap_file_handle;
    HANDLE          mmap_mapping_handle;
#else
    int             mmap_fd;
#endif
};

/* ─── Internal: Arena Operations ──────────────────────────────────── */

static AxoStatus axo_arena_create(AxoArena* arena, size_t capacity, size_t alignment) {
    arena->capacity  = capacity;
    arena->offset    = 0;
    arena->alignment = alignment;
    arena->scratch_start = 0;
    arena->guard_tag = AXO_ARENA_GUARD_TAG;

    cudaError_t err = cudaMalloc(&arena->base, capacity);
    if (err != cudaSuccess) {
        fprintf(stderr, "[axo] cudaMalloc failed (%zu bytes): %s\n",
                capacity, cudaGetErrorString(err));
        return AXO_ERROR_OUT_OF_MEMORY;
    }

    /* Zero-initialize the entire arena */
    cudaMemset(arena->base, 0, capacity);
    return AXO_SUCCESS;
}

/**
 * @brief Bump-allocate from the arena. Returns a device pointer.
 */
static void* axo_arena_alloc(AxoArena* arena, size_t size) {
    size_t aligned_offset = axo_align_up(arena->offset, arena->alignment);
    if (aligned_offset + size > arena->capacity) {
        fprintf(stderr, "[axo] Arena exhausted: need %zu, have %zu remaining\n",
                size, arena->capacity - aligned_offset);
        return NULL;
    }
    void* ptr = (char*)arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return ptr;
}

static void axo_arena_mark_scratch(AxoArena* arena) {
    arena->scratch_start = arena->offset;
}

static void axo_arena_reset(AxoArena* arena) {
    arena->offset = arena->scratch_start;
    axo_validate_arena_boundary(arena->base, arena->offset, arena->capacity);
}

static void axo_arena_destroy(AxoArena* arena) {
    if (arena->base) {
        cudaFree(arena->base);
        arena->base     = NULL;
        arena->capacity = 0;
        arena->offset   = 0;
    }
}

/* ─── Internal: Size Calculations ─────────────────────────────────── */

/** Size of per-layer encoder weights */
static size_t axo_calc_layer_weight_size(size_t D, size_t F) {
    size_t total = 0;
    total += D * 3 * D * sizeof(uint16_t);    /* QKV projection  fp16 */
    total += D * D * sizeof(uint16_t);        /* Out projection  fp16 */
    total += D * (D * F) * sizeof(uint16_t);  /* FFN up-proj     fp16 */
    total += (D * F) * D * sizeof(uint16_t);  /* FFN down-proj   fp16 */
    total += D * sizeof(float);               /* LN gamma        fp32 */
    total += D * sizeof(float);               /* LN beta         fp32 */
    return total;
}

static size_t axo_calc_weight_size(const AxoConfig* cfg) {
    size_t D = cfg->embed_dim;
    size_t C = cfg->max_choices;
    size_t V = cfg->vocab_size;
    size_t F = cfg->ffn_mult;
    size_t N = cfg->num_layers;
    size_t total = 0;

    total += V * D * sizeof(uint16_t);                  /* Embed table     */
    total += N * axo_calc_layer_weight_size(D, F);      /* N encoder layers */
    total += D * C * sizeof(uint16_t);                  /* Choice matrix   */
    total += 2 * sizeof(float);                         /* Platt temp+bias */

    return axo_align_up(total, AXO_DEFAULT_ALIGNMENT);
}

static size_t axo_calc_scratch_size(const AxoConfig* cfg) {
    size_t B = cfg->max_batch_size;
    size_t S = cfg->max_seq_len;
    size_t D = cfg->embed_dim;
    size_t H = cfg->num_heads;
    size_t C = cfg->max_choices;
    size_t total = 0;

    total += B * S * D * sizeof(uint16_t);         /* state_ping       */
    total += B * S * D * sizeof(uint16_t);         /* state_pong       */
    total += B * S * 3 * D * sizeof(uint16_t);     /* qkv_buf          */
    total += B * H * S * S * sizeof(float);        /* attn_logits      */
    total += B * S * D * sizeof(uint16_t);         /* attn_out         */
    total += B * D * sizeof(uint16_t);             /* pooled_embed     */
    total += B * C * sizeof(float);                /* choice_logits    */
    total += B * C * sizeof(float);                /* softmax_out      */

    return axo_align_up(total, AXO_DEFAULT_ALIGNMENT);
}

/* ─── Internal: Carve Scratchpad Regions ──────────────────────────── */

static AxoStatus axo_carve_scratch(AxoContext* ctx) {
    AxoArena*       a   = &ctx->arena;
    AxoScratchPtrs* s   = &ctx->scratch;
    const AxoConfig* c  = &ctx->config;

    size_t B = c->max_batch_size;
    size_t S = c->max_seq_len;
    size_t D = c->embed_dim;
    size_t H = c->num_heads;
    size_t C = c->max_choices;

    s->state_ping    = axo_arena_alloc(a, B * S * D * sizeof(uint16_t));
    s->state_pong    = axo_arena_alloc(a, B * S * D * sizeof(uint16_t));
    s->qkv_buf       = axo_arena_alloc(a, B * S * 3 * D * sizeof(uint16_t));
    s->attn_logits   = axo_arena_alloc(a, B * H * S * S * sizeof(float));
    s->attn_out      = axo_arena_alloc(a, B * S * D * sizeof(uint16_t));
    s->pooled_embed  = axo_arena_alloc(a, B * D * sizeof(uint16_t));
    s->choice_logits = axo_arena_alloc(a, B * C * sizeof(float));
    s->softmax_out   = axo_arena_alloc(a, B * C * sizeof(float));

    if (!s->state_ping || !s->state_pong || !s->qkv_buf ||
        !s->attn_logits || !s->attn_out || !s->pooled_embed ||
        !s->choice_logits || !s->softmax_out) {
        return AXO_ERROR_OUT_OF_MEMORY;
    }

    return AXO_SUCCESS;
}

/* ─── External: Kernel Launchers ──────────────────────────────────── */

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
);

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
);

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
);

extern "C" AxoStatus axo_launch_mean_pool_norm(
    const void*      encoded,
    const int32_t*   token_ids,
    void*            pooled,
    uint32_t         batch_size,
    uint32_t         seq_len,
    uint32_t         embed_dim,
    int32_t          pad_token_id,
    cudaStream_t     stream
);

/* ═══════════════════════════════════════════════════════════════════
 *  PUBLIC API IMPLEMENTATION
 * ═══════════════════════════════════════════════════════════════════ */

extern "C" {

AXO_API AxoConfig axo_config_default(void) {
    AxoConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.device_id       = 0;
    cfg.max_batch_size  = 32;
    cfg.max_seq_len     = 128;
    cfg.embed_dim       = 256;
    cfg.num_heads       = 8;
    cfg.max_choices     = 512;
    cfg.vocab_size      = 32768;
    cfg.ffn_mult        = 4;
    cfg.num_layers      = 2;
    return cfg;
}

AXO_API AxoStatus axo_create(AxoContext** out_ctx, const AxoConfig* cfg) {
    if (!out_ctx || !cfg) return AXO_ERROR_INVALID_ARG;
    if (cfg->embed_dim == 0 || cfg->max_batch_size == 0 || cfg->max_choices == 0
        || cfg->vocab_size == 0 || cfg->ffn_mult == 0 || cfg->num_layers == 0) {
        return AXO_ERROR_INVALID_ARG;
    }
    if (cfg->num_layers > AXO_MAX_LAYERS) return AXO_ERROR_INVALID_ARG;
    if (cfg->embed_dim % cfg->num_heads != 0) return AXO_ERROR_INVALID_ARG;

    cudaError_t cerr = cudaSetDevice((int)cfg->device_id);
    if (cerr != cudaSuccess) return AXO_ERROR_CUDA;

    AxoContext* ctx = (AxoContext*)calloc(1, sizeof(AxoContext));
    if (!ctx) return AXO_ERROR_OUT_OF_MEMORY;

    ctx->config = *cfg;

    size_t weight_size  = axo_calc_weight_size(cfg);
    size_t scratch_size = axo_calc_scratch_size(cfg);
    size_t output_size  = axo_align_up(
        cfg->max_batch_size * sizeof(AxoDecision), AXO_DEFAULT_ALIGNMENT);
    size_t total = weight_size + scratch_size + output_size;

    AxoStatus status = axo_arena_create(&ctx->arena, total, AXO_DEFAULT_ALIGNMENT);
    if (status != AXO_SUCCESS) {
        free(ctx);
        return status;
    }

    ctx->arena.offset = weight_size;
    axo_arena_mark_scratch(&ctx->arena);

    status = axo_carve_scratch(ctx);
    if (status != AXO_SUCCESS) {
        axo_arena_destroy(&ctx->arena);
        free(ctx);
        return status;
    }

    cerr = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (cerr != cudaSuccess) {
        axo_arena_destroy(&ctx->arena);
        free(ctx);
        return AXO_ERROR_CUDA;
    }

    cerr = cudaEventCreate(&ctx->ev_start);
    if (cerr != cudaSuccess) {
        cudaStreamDestroy(ctx->stream);
        axo_arena_destroy(&ctx->arena);
        free(ctx);
        return AXO_ERROR_CUDA;
    }
    cerr = cudaEventCreate(&ctx->ev_stop);
    if (cerr != cudaSuccess) {
        cudaEventDestroy(ctx->ev_start);
        cudaStreamDestroy(ctx->stream);
        axo_arena_destroy(&ctx->arena);
        free(ctx);
        return AXO_ERROR_CUDA;
    }

    ctx->platt_temp_cached = 1.0f;
    ctx->platt_bias_cached = 0.0f;
    ctx->weights_loaded = 0;
    *out_ctx = ctx;
    return AXO_SUCCESS;
}

AXO_API void axo_destroy(AxoContext* ctx) {
    if (!ctx) return;

    if (ctx->mmap_host_ptr) {
#ifdef _WIN32
        UnmapViewOfFile(ctx->mmap_host_ptr);
        if (ctx->mmap_mapping_handle) CloseHandle(ctx->mmap_mapping_handle);
        if (ctx->mmap_file_handle != INVALID_HANDLE_VALUE) CloseHandle(ctx->mmap_file_handle);
#else
        munmap(ctx->mmap_host_ptr, ctx->mmap_size);
        if (ctx->mmap_fd >= 0) close(ctx->mmap_fd);
#endif
    }

    if (ctx->ev_start) cudaEventDestroy(ctx->ev_start);
    if (ctx->ev_stop)  cudaEventDestroy(ctx->ev_stop);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    axo_arena_destroy(&ctx->arena);
    free(ctx);
}

AXO_API AxoStatus axo_load_weights_mmap(AxoContext* ctx, const char* filepath) {
    if (!ctx || !filepath) return AXO_ERROR_INVALID_ARG;

    const AxoConfig* cfg = &ctx->config;
    size_t D = cfg->embed_dim;
    size_t C = cfg->max_choices;
    size_t V = cfg->vocab_size;
    size_t F = cfg->ffn_mult;
    size_t N = cfg->num_layers;

    /* Calculate expected file size */
    size_t expected_size = 0;
    expected_size += V * D * sizeof(uint16_t);                  /* Embed table  */
    expected_size += N * axo_calc_layer_weight_size(D, F);      /* N layers     */
    expected_size += D * C * sizeof(uint16_t);                  /* Choice mat   */
    expected_size += 2 * sizeof(float);                         /* Platt t+b    */

    /* Memory-map the file */
    void*  host_ptr  = NULL;
    size_t file_size = 0;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(filepath, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return AXO_ERROR_MMAP_FAILED;

    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    file_size = (size_t)li.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return AXO_ERROR_MMAP_FAILED; }

    host_ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!host_ptr) { CloseHandle(hMap); CloseHandle(hFile); return AXO_ERROR_MMAP_FAILED; }

    ctx->mmap_file_handle    = hFile;
    ctx->mmap_mapping_handle = hMap;
#else
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return AXO_ERROR_MMAP_FAILED;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return AXO_ERROR_MMAP_FAILED; }
    file_size = (size_t)st.st_size;

    host_ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_ptr == MAP_FAILED) { close(fd); return AXO_ERROR_MMAP_FAILED; }

    madvise(host_ptr, file_size, MADV_SEQUENTIAL);
    ctx->mmap_fd = fd;
#endif

    if (file_size < expected_size) {
#ifdef _WIN32
        UnmapViewOfFile(host_ptr);
        CloseHandle(ctx->mmap_mapping_handle);
        CloseHandle(ctx->mmap_file_handle);
#else
        munmap(host_ptr, file_size);
        close(ctx->mmap_fd);
#endif
        fprintf(stderr, "[axo] Weight file mismatch: got %zu bytes, expected %zu\n",
                file_size, expected_size);
        return AXO_ERROR_WEIGHT_MISMATCH;
    }

    ctx->mmap_host_ptr = host_ptr;
    ctx->mmap_size     = file_size;

    /* Copy mmap'd weights into the weight zone of the GPU arena.
     * Using synchronous cudaMemcpy because mmap'd host memory is not
     * pinned (page-locked). */
    cudaError_t cerr = cudaMemcpy(
        ctx->arena.base, host_ptr, expected_size,
        cudaMemcpyHostToDevice
    );
    if (cerr != cudaSuccess) return AXO_ERROR_CUDA;

    /* Set up weight pointers into the arena */
    char* base = (char*)ctx->arena.base;
    size_t off = 0;

    ctx->weights.embed_table = base + off;  off += V * D * sizeof(uint16_t);

    /* Per-layer weight pointers */
    size_t layer_size = axo_calc_layer_weight_size(D, F);
    for (uint32_t n = 0; n < N; ++n) {
        char* lbase = base + off;
        size_t loff = 0;
        ctx->weights.layers[n].qkv_proj  = lbase + loff; loff += D * 3 * D * sizeof(uint16_t);
        ctx->weights.layers[n].out_proj  = lbase + loff; loff += D * D * sizeof(uint16_t);
        ctx->weights.layers[n].ffn_up    = lbase + loff; loff += D * (D * F) * sizeof(uint16_t);
        ctx->weights.layers[n].ffn_down  = lbase + loff; loff += (D * F) * D * sizeof(uint16_t);
        ctx->weights.layers[n].ln_gamma  = lbase + loff; loff += D * sizeof(float);
        ctx->weights.layers[n].ln_beta   = lbase + loff;
        off += layer_size;
    }

    ctx->weights.choice_matrix = base + off;  off += D * C * sizeof(uint16_t);
    ctx->weights.platt_temp    = (float*)(base + off);  off += sizeof(float);
    ctx->weights.platt_bias    = (float*)(base + off);

    /* Cache Platt parameters from the mmap'd host buffer */
    {
        const char* host_base = (const char*)host_ptr;
        size_t platt_off = V * D * sizeof(uint16_t)
                         + N * layer_size
                         + D * C * sizeof(uint16_t);
        memcpy(&ctx->platt_temp_cached, host_base + platt_off, sizeof(float));
        memcpy(&ctx->platt_bias_cached, host_base + platt_off + sizeof(float), sizeof(float));
    }

    cerr = cudaStreamSynchronize(ctx->stream);
    if (cerr != cudaSuccess) return AXO_ERROR_CUDA;

    ctx->weights_loaded = 1;
    return AXO_SUCCESS;
}

AXO_API AxoStatus axo_evaluate(
    AxoContext*          ctx,
    const AxoEvalParams* params,
    AxoDecision*         results
) {
    if (!ctx || !params || !results) return AXO_ERROR_INVALID_ARG;
    if (!ctx->weights_loaded) return AXO_ERROR_NOT_INITIALIZED;
    if (params->batch_size == 0) return AXO_ERROR_INVALID_ARG;
    if (params->batch_size > ctx->config.max_batch_size) return AXO_ERROR_BATCH_OVERFLOW;
    if (params->num_choices > ctx->config.max_choices) return AXO_ERROR_CARDINALITY_OVERFLOW;

    axo_arena_reset(&ctx->arena);

    const void* choice_mat = params->choice_matrix
        ? params->choice_matrix
        : ctx->weights.choice_matrix;

    float platt_temp = ctx->platt_temp_cached;
    float platt_bias = ctx->platt_bias_cached;

    cudaEventRecord(ctx->ev_start, ctx->stream);

    AxoStatus status = axo_launch_decision_kernel(
        params->embeddings,
        choice_mat,
        platt_temp,
        platt_bias,
        ctx->scratch.choice_logits,
        ctx->scratch.softmax_out,
        params->batch_size,
        ctx->config.embed_dim,
        params->num_choices,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    cudaEventRecord(ctx->ev_stop, ctx->stream);

    size_t choice_bytes = params->batch_size * params->num_choices * sizeof(float);
    float* host_softmax = (float*)axo_alloca(choice_bytes);
    float* host_logits  = (float*)axo_alloca(choice_bytes);

    cudaMemcpyAsync(host_softmax, ctx->scratch.softmax_out, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaMemcpyAsync(host_logits, ctx->scratch.choice_logits, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);

    cudaStreamSynchronize(ctx->stream);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, ctx->ev_start, ctx->ev_stop);
    float kernel_us = kernel_ms * 1000.0f;

    for (uint32_t b = 0; b < params->batch_size; ++b) {
        const float* probs  = host_softmax + b * params->num_choices;
        const float* logits = host_logits  + b * params->num_choices;
        uint32_t best_idx = 0;
        float    best_val = probs[0];
        float    entropy  = 0.0f;

        for (uint32_t c = 0; c < params->num_choices; ++c) {
            if (probs[c] > best_val) {
                best_val = probs[c];
                best_idx = c;
            }
            if (probs[c] > 1e-10f) {
                entropy -= probs[c] * logf(probs[c]);
            }
        }

        results[b].choice_index    = best_idx;
        results[b].calibrated_prob = best_val;
        results[b].raw_logit       = logits[best_idx];
        results[b].entropy         = entropy;
        results[b].latency_us      = kernel_us;
    }

    return AXO_SUCCESS;
}

AXO_API AxoStatus axo_evaluate_tokens(
    AxoContext*           ctx,
    const AxoStateParams* params,
    AxoDecision*          results
) {
    if (!ctx || !params || !results) return AXO_ERROR_INVALID_ARG;
    if (!ctx->weights_loaded) return AXO_ERROR_NOT_INITIALIZED;
    if (params->batch_size == 0) return AXO_ERROR_INVALID_ARG;
    if (params->batch_size > ctx->config.max_batch_size) return AXO_ERROR_BATCH_OVERFLOW;
    if (params->num_choices > ctx->config.max_choices) return AXO_ERROR_CARDINALITY_OVERFLOW;
    if (params->seq_len > ctx->config.max_seq_len) return AXO_ERROR_SEQ_OVERFLOW;
    if (params->seq_len == 0) return AXO_ERROR_INVALID_ARG;

    axo_arena_reset(&ctx->arena);
    cudaEventRecord(ctx->ev_start, ctx->stream);

    uint32_t N = ctx->config.num_layers;

    /* ── Stage 1a: Embedding + LayerNorm (layer 0) into state_ping ── */
    AxoStatus status = axo_launch_embed_layernorm(
        params->token_ids,
        ctx->weights.embed_table,
        (const float*)ctx->weights.layers[0].ln_gamma,
        (const float*)ctx->weights.layers[0].ln_beta,
        ctx->scratch.state_ping,
        params->batch_size,
        params->seq_len,
        ctx->config.embed_dim,
        ctx->config.vocab_size,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    /* ── Stage 1b: N encoder layers with ping-pong ── */
    void* ping = ctx->scratch.state_ping;
    void* pong = ctx->scratch.state_pong;

    for (uint32_t layer = 0; layer < N; ++layer) {
        /* Read from ping, write to pong */
        status = axo_launch_encoder_layer(
            ping,                                   /* input  */
            pong,                                   /* output */
            ctx->weights.layers[layer].qkv_proj,
            ctx->weights.layers[layer].out_proj,
            ctx->weights.layers[layer].ffn_up,
            ctx->weights.layers[layer].ffn_down,
            (const float*)ctx->weights.layers[layer].ln_gamma,
            (const float*)ctx->weights.layers[layer].ln_beta,
            ctx->scratch.qkv_buf,
            ctx->scratch.attn_logits,
            ctx->scratch.attn_out,
            params->batch_size,
            params->seq_len,
            ctx->config.embed_dim,
            ctx->config.num_heads,
            ctx->config.ffn_mult,
            ctx->stream
        );
        if (status != AXO_SUCCESS) return status;

        /* Swap ping/pong for next layer */
        void* tmp = ping;
        ping = pong;
        pong = tmp;
    }

    /* After N layers, final state is in `ping` (due to last swap) */

    /* ── Stage 1c: Masked mean-pool + L2 norm ── */
    status = axo_launch_mean_pool_norm(
        ping,
        params->token_ids,
        ctx->scratch.pooled_embed,
        params->batch_size,
        params->seq_len,
        ctx->config.embed_dim,
        params->pad_token_id,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    /* ── Stage 2: Choice routing ── */
    const void* choice_mat = params->choice_matrix
        ? params->choice_matrix
        : ctx->weights.choice_matrix;

    status = axo_launch_decision_kernel(
        ctx->scratch.pooled_embed,
        choice_mat,
        ctx->platt_temp_cached,
        ctx->platt_bias_cached,
        ctx->scratch.choice_logits,
        ctx->scratch.softmax_out,
        params->batch_size,
        ctx->config.embed_dim,
        params->num_choices,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    cudaEventRecord(ctx->ev_stop, ctx->stream);

    /* Copy results to host */
    size_t choice_bytes = params->batch_size * params->num_choices * sizeof(float);
    float* host_softmax = (float*)axo_alloca(choice_bytes);
    float* host_logits  = (float*)axo_alloca(choice_bytes);

    cudaMemcpyAsync(host_softmax, ctx->scratch.softmax_out, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaMemcpyAsync(host_logits, ctx->scratch.choice_logits, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);

    cudaStreamSynchronize(ctx->stream);

    float pipeline_ms = 0.0f;
    cudaEventElapsedTime(&pipeline_ms, ctx->ev_start, ctx->ev_stop);
    float pipeline_us = pipeline_ms * 1000.0f;

    for (uint32_t b = 0; b < params->batch_size; ++b) {
        const float* probs  = host_softmax + b * params->num_choices;
        const float* logits = host_logits  + b * params->num_choices;
        uint32_t best_idx = 0;
        float    best_val = probs[0];
        float    entropy  = 0.0f;

        for (uint32_t c = 0; c < params->num_choices; ++c) {
            if (probs[c] > best_val) {
                best_val = probs[c];
                best_idx = c;
            }
            if (probs[c] > 1e-10f) {
                entropy -= probs[c] * logf(probs[c]);
            }
        }

        results[b].choice_index    = best_idx;
        results[b].calibrated_prob = best_val;
        results[b].raw_logit       = logits[best_idx];
        results[b].entropy         = entropy;
        results[b].latency_us      = pipeline_us;
    }

    return AXO_SUCCESS;
}

AXO_API AxoStatus axo_synchronize(AxoContext* ctx) {
    if (!ctx) return AXO_ERROR_INVALID_ARG;
    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    return (err == cudaSuccess) ? AXO_SUCCESS : AXO_ERROR_CUDA;
}

AXO_API AxoStatus axo_device_alloc(void** dev_ptr, size_t size_bytes) {
    if (!dev_ptr || size_bytes == 0) return AXO_ERROR_INVALID_ARG;
    cudaError_t err = cudaMalloc(dev_ptr, size_bytes);
    return (err == cudaSuccess) ? AXO_SUCCESS : AXO_ERROR_OUT_OF_MEMORY;
}

AXO_API void axo_device_free(void* dev_ptr) {
    if (dev_ptr) cudaFree(dev_ptr);
}

AXO_API AxoStatus axo_device_upload(void* dev_ptr, const void* host_ptr, size_t size_bytes) {
    if (!dev_ptr || !host_ptr || size_bytes == 0) return AXO_ERROR_INVALID_ARG;
    cudaError_t err = cudaMemcpy(dev_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? AXO_SUCCESS : AXO_ERROR_CUDA;
}


/* ═══════════════════════════════════════════════════════════════════
 *  CARTRIDGE ENGINE IMPLEMENTATION
 * ═══════════════════════════════════════════════════════════════════ */

AXO_API AxoStatus axo_cartridge_load_mmap(
    const char*   path,
    AxoCartridge* out_cartridge
) {
    if (!path || !out_cartridge) return AXO_ERROR_INVALID_ARG;
    memset(out_cartridge, 0, sizeof(AxoCartridge));

    size_t file_size = 0;
    void* host_ptr = NULL;

#ifdef _WIN32
    HANDLE file_handle = CreateFileA(
        path, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[axo] Failed to open cartridge: %s (error %lu)\n",
                path, GetLastError());
        return AXO_ERROR_MMAP_FAILED;
    }

    LARGE_INTEGER li_size;
    if (!GetFileSizeEx(file_handle, &li_size)) {
        CloseHandle(file_handle);
        return AXO_ERROR_MMAP_FAILED;
    }
    file_size = (size_t)li_size.QuadPart;

    if (file_size < sizeof(AxoCartridgeHeader)) {
        CloseHandle(file_handle);
        return AXO_ERROR_CARTRIDGE_INVALID;
    }

    HANDLE mapping_handle = CreateFileMappingA(
        file_handle, NULL, PAGE_READONLY, 0, 0, NULL
    );
    if (!mapping_handle) {
        CloseHandle(file_handle);
        return AXO_ERROR_MMAP_FAILED;
    }

    host_ptr = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (!host_ptr) {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        return AXO_ERROR_MMAP_FAILED;
    }

    out_cartridge->file_handle    = file_handle;
    out_cartridge->mapping_handle = mapping_handle;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[axo] Failed to open cartridge: %s\n", path);
        return AXO_ERROR_MMAP_FAILED;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return AXO_ERROR_MMAP_FAILED;
    }
    file_size = (size_t)st.st_size;

    if (file_size < sizeof(AxoCartridgeHeader)) {
        close(fd);
        return AXO_ERROR_CARTRIDGE_INVALID;
    }

    host_ptr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (host_ptr == MAP_FAILED) {
        close(fd);
        return AXO_ERROR_MMAP_FAILED;
    }

    out_cartridge->fd = fd;
#endif

    out_cartridge->mmap_ptr  = host_ptr;
    out_cartridge->mmap_size = file_size;

    /* Validate header */
    const AxoCartridgeHeader* hdr = (const AxoCartridgeHeader*)host_ptr;
    if (hdr->magic != AXO_CARTRIDGE_MAGIC) {
        fprintf(stderr, "[axo] Invalid cartridge magic: 0x%08X (expected 0x%08X)\n",
                hdr->magic, AXO_CARTRIDGE_MAGIC);
        axo_cartridge_unload(out_cartridge);
        return AXO_ERROR_CARTRIDGE_INVALID;
    }

    if (hdr->version != AXO_CARTRIDGE_VERSION) {
        fprintf(stderr, "[axo] Unsupported cartridge version: %u (expected %u)\n",
                hdr->version, AXO_CARTRIDGE_VERSION);
        axo_cartridge_unload(out_cartridge);
        return AXO_ERROR_CARTRIDGE_VERSION;
    }

    if (hdr->embed_dim == 0 || hdr->num_choices == 0) {
        axo_cartridge_unload(out_cartridge);
        return AXO_ERROR_CARTRIDGE_INVALID;
    }

    size_t matrix_bytes = (size_t)hdr->embed_dim * hdr->num_choices * sizeof(uint16_t);
    size_t expected_size = sizeof(AxoCartridgeHeader) + matrix_bytes + hdr->label_section_bytes;

    if (file_size != expected_size) {
        fprintf(stderr, "[axo] Cartridge size mismatch: file has %zu bytes, expected %zu\n",
                file_size, expected_size);
        axo_cartridge_unload(out_cartridge);
        return AXO_ERROR_CARTRIDGE_INVALID;
    }

    memcpy(&out_cartridge->header, hdr, sizeof(AxoCartridgeHeader));

    /* Provision dedicated device memory for the choice matrix */
    cudaError_t cerr = cudaMalloc(&out_cartridge->d_choice_matrix, matrix_bytes);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[axo] cudaMalloc failed for cartridge choice matrix (%zu bytes): %s\n",
                matrix_bytes, cudaGetErrorString(cerr));
        axo_cartridge_unload(out_cartridge);
        return AXO_ERROR_OUT_OF_MEMORY;
    }

    const char* matrix_src = (const char*)host_ptr + sizeof(AxoCartridgeHeader);
    cerr = cudaMemcpy(out_cartridge->d_choice_matrix, matrix_src, matrix_bytes,
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        axo_cartridge_unload(out_cartridge);
        return AXO_ERROR_CUDA;
    }

    /* Map label section */
    if (hdr->label_section_bytes > 0) {
        out_cartridge->labels = matrix_src + matrix_bytes;
    } else {
        out_cartridge->labels = NULL;
    }

    return AXO_SUCCESS;
}

AXO_API AxoStatus axo_cartridge_unload(AxoCartridge* cartridge) {
    if (!cartridge) return AXO_SUCCESS;

    if (cartridge->d_choice_matrix) {
        cudaFree(cartridge->d_choice_matrix);
        cartridge->d_choice_matrix = NULL;
    }

    if (cartridge->mmap_ptr) {
#ifdef _WIN32
        UnmapViewOfFile(cartridge->mmap_ptr);
        if (cartridge->mapping_handle) CloseHandle((HANDLE)cartridge->mapping_handle);
        if (cartridge->file_handle) CloseHandle((HANDLE)cartridge->file_handle);
#else
        munmap(cartridge->mmap_ptr, cartridge->mmap_size);
        if (cartridge->fd >= 0) close(cartridge->fd);
#endif
        cartridge->mmap_ptr  = NULL;
        cartridge->mmap_size = 0;
    }

    memset(cartridge, 0, sizeof(AxoCartridge));
    return AXO_SUCCESS;
}

AXO_API const char* axo_cartridge_get_label(
    const AxoCartridge* cartridge,
    uint32_t            choice_index
) {
    if (!cartridge || !cartridge->labels) return NULL;
    if (choice_index >= cartridge->header.num_choices) return NULL;

    const char* ptr = cartridge->labels;
    const char* end = cartridge->labels + cartridge->header.label_section_bytes;

    for (uint32_t i = 0; i < choice_index; ++i) {
        while (ptr < end && *ptr != '\0') ptr++;
        if (ptr >= end) return NULL;
        ptr++; /* skip null terminator */
    }

    return (ptr < end) ? ptr : NULL;
}

AXO_API AxoStatus axo_evaluate_cartridge(
    AxoContext*           ctx,
    const AxoCartridge*   cartridge,
    const AxoStateParams* params,
    AxoDecision*          results
) {
    if (!ctx || !cartridge || !params || !results) return AXO_ERROR_INVALID_ARG;
    if (!ctx->weights_loaded) return AXO_ERROR_NOT_INITIALIZED;
    if (!cartridge->d_choice_matrix) return AXO_ERROR_CARTRIDGE_INVALID;

    if (cartridge->header.embed_dim != ctx->config.embed_dim) {
        fprintf(stderr, "[axo] Cartridge embed_dim (%u) != context embed_dim (%u)\n",
                cartridge->header.embed_dim, ctx->config.embed_dim);
        return AXO_ERROR_INVALID_ARG;
    }
    if (cartridge->header.num_choices > ctx->config.max_choices) {
        return AXO_ERROR_CARDINALITY_OVERFLOW;
    }
    if (params->batch_size == 0) return AXO_ERROR_INVALID_ARG;
    if (params->batch_size > ctx->config.max_batch_size) return AXO_ERROR_BATCH_OVERFLOW;
    if (params->seq_len > ctx->config.max_seq_len) return AXO_ERROR_SEQ_OVERFLOW;
    if (params->seq_len == 0) return AXO_ERROR_INVALID_ARG;

    axo_arena_reset(&ctx->arena);
    cudaEventRecord(ctx->ev_start, ctx->stream);

    uint32_t N = ctx->config.num_layers;

    /* ── Stage 1a: Embedding + LayerNorm (layer 0) into state_ping ── */
    AxoStatus status = axo_launch_embed_layernorm(
        params->token_ids,
        ctx->weights.embed_table,
        (const float*)ctx->weights.layers[0].ln_gamma,
        (const float*)ctx->weights.layers[0].ln_beta,
        ctx->scratch.state_ping,
        params->batch_size,
        params->seq_len,
        ctx->config.embed_dim,
        ctx->config.vocab_size,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    /* ── Stage 1b: N encoder layers with ping-pong ── */
    void* ping = ctx->scratch.state_ping;
    void* pong = ctx->scratch.state_pong;

    for (uint32_t layer = 0; layer < N; ++layer) {
        status = axo_launch_encoder_layer(
            ping,
            pong,
            ctx->weights.layers[layer].qkv_proj,
            ctx->weights.layers[layer].out_proj,
            ctx->weights.layers[layer].ffn_up,
            ctx->weights.layers[layer].ffn_down,
            (const float*)ctx->weights.layers[layer].ln_gamma,
            (const float*)ctx->weights.layers[layer].ln_beta,
            ctx->scratch.qkv_buf,
            ctx->scratch.attn_logits,
            ctx->scratch.attn_out,
            params->batch_size,
            params->seq_len,
            ctx->config.embed_dim,
            ctx->config.num_heads,
            ctx->config.ffn_mult,
            ctx->stream
        );
        if (status != AXO_SUCCESS) return status;

        void* tmp = ping;
        ping = pong;
        pong = tmp;
    }

    /* ── Stage 1c: Masked mean-pool + L2 norm ── */
    status = axo_launch_mean_pool_norm(
        ping,
        params->token_ids,
        ctx->scratch.pooled_embed,
        params->batch_size,
        params->seq_len,
        ctx->config.embed_dim,
        params->pad_token_id,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    /* ── Stage 2: Choice routing via swappable cartridge head ── */
    /* Hot-swap: pass cartridge's resident GPU choice matrix directly */
    uint32_t active_choices = cartridge->header.num_choices;
    if (params->num_choices > 0 && params->num_choices < active_choices) {
        active_choices = params->num_choices;
    }

    status = axo_launch_decision_kernel(
        ctx->scratch.pooled_embed,
        cartridge->d_choice_matrix,
        cartridge->header.platt_temperature,
        cartridge->header.platt_bias,
        ctx->scratch.choice_logits,
        ctx->scratch.softmax_out,
        params->batch_size,
        ctx->config.embed_dim,
        active_choices,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    cudaEventRecord(ctx->ev_stop, ctx->stream);

    /* Copy results to host */
    size_t choice_bytes = params->batch_size * active_choices * sizeof(float);
    float* host_softmax = (float*)axo_alloca(choice_bytes);
    float* host_logits  = (float*)axo_alloca(choice_bytes);

    cudaMemcpyAsync(host_softmax, ctx->scratch.softmax_out, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaMemcpyAsync(host_logits, ctx->scratch.choice_logits, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);

    cudaStreamSynchronize(ctx->stream);

    float pipeline_ms = 0.0f;
    cudaEventElapsedTime(&pipeline_ms, ctx->ev_start, ctx->ev_stop);
    float pipeline_us = pipeline_ms * 1000.0f;

    for (uint32_t b = 0; b < params->batch_size; ++b) {
        const float* probs  = host_softmax + b * active_choices;
        const float* logits = host_logits  + b * active_choices;
        uint32_t best_idx = 0;
        float    best_val = probs[0];
        float    entropy  = 0.0f;

        for (uint32_t c = 0; c < active_choices; ++c) {
            if (probs[c] > best_val) {
                best_val = probs[c];
                best_idx = c;
            }
            if (probs[c] > 1e-10f) {
                entropy -= probs[c] * logf(probs[c]);
            }
        }

        results[b].choice_index    = best_idx;
        results[b].calibrated_prob = best_val;
        results[b].raw_logit       = logits[best_idx];
        results[b].entropy         = entropy;
        results[b].latency_us      = pipeline_us;
    }

    return AXO_SUCCESS;
}

AXO_API AxoStatus axo_evaluate_cartridge_embeddings(
    AxoContext*          ctx,
    const AxoCartridge*  cartridge,
    const AxoEvalParams* params,
    AxoDecision*         results
) {
    if (!ctx || !cartridge || !params || !results) return AXO_ERROR_INVALID_ARG;
    if (!cartridge->d_choice_matrix) return AXO_ERROR_CARTRIDGE_INVALID;
    if (!params->embeddings) return AXO_ERROR_INVALID_ARG;

    if (cartridge->header.embed_dim != ctx->config.embed_dim) {
        return AXO_ERROR_INVALID_ARG;
    }
    if (cartridge->header.num_choices > ctx->config.max_choices) {
        return AXO_ERROR_CARDINALITY_OVERFLOW;
    }
    if (params->batch_size == 0) return AXO_ERROR_INVALID_ARG;
    if (params->batch_size > ctx->config.max_batch_size) return AXO_ERROR_BATCH_OVERFLOW;

    axo_arena_reset(&ctx->arena);
    cudaEventRecord(ctx->ev_start, ctx->stream);

    uint32_t active_choices = cartridge->header.num_choices;
    if (params->num_choices > 0 && params->num_choices < active_choices) {
        active_choices = params->num_choices;
    }

    AxoStatus status = axo_launch_decision_kernel(
        params->embeddings,
        cartridge->d_choice_matrix,
        cartridge->header.platt_temperature,
        cartridge->header.platt_bias,
        ctx->scratch.choice_logits,
        ctx->scratch.softmax_out,
        params->batch_size,
        ctx->config.embed_dim,
        active_choices,
        ctx->stream
    );
    if (status != AXO_SUCCESS) return status;

    cudaEventRecord(ctx->ev_stop, ctx->stream);

    size_t choice_bytes = params->batch_size * active_choices * sizeof(float);
    float* host_softmax = (float*)axo_alloca(choice_bytes);
    float* host_logits  = (float*)axo_alloca(choice_bytes);

    cudaMemcpyAsync(host_softmax, ctx->scratch.softmax_out, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaMemcpyAsync(host_logits, ctx->scratch.choice_logits, choice_bytes,
                    cudaMemcpyDeviceToHost, ctx->stream);

    cudaStreamSynchronize(ctx->stream);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, ctx->ev_start, ctx->ev_stop);
    float kernel_us = kernel_ms * 1000.0f;

    for (uint32_t b = 0; b < params->batch_size; ++b) {
        const float* probs  = host_softmax + b * active_choices;
        const float* logits = host_logits  + b * active_choices;
        uint32_t best_idx = 0;
        float    best_val = probs[0];
        float    entropy  = 0.0f;

        for (uint32_t c = 0; c < active_choices; ++c) {
            if (probs[c] > best_val) {
                best_val = probs[c];
                best_idx = c;
            }
            if (probs[c] > 1e-10f) {
                entropy -= probs[c] * logf(probs[c]);
            }
        }

        results[b].choice_index    = best_idx;
        results[b].calibrated_prob = best_val;
        results[b].raw_logit       = logits[best_idx];
        results[b].entropy         = entropy;
        results[b].latency_us      = kernel_us;
    }

    return AXO_SUCCESS;
}

AXO_API const char* axo_get_error_string(AxoStatus status) {
    switch (status) {
        case AXO_SUCCESS:                     return "Success";
        case AXO_ERROR_INVALID_ARG:           return "Invalid argument";
        case AXO_ERROR_CUDA:                  return "CUDA runtime error";
        case AXO_ERROR_OUT_OF_MEMORY:         return "Out of memory";
        case AXO_ERROR_MMAP_FAILED:           return "Memory-map failed";
        case AXO_ERROR_WEIGHT_MISMATCH:       return "Weight file size mismatch";
        case AXO_ERROR_NOT_INITIALIZED:       return "Context not initialized / weights not loaded";
        case AXO_ERROR_BATCH_OVERFLOW:        return "Batch size exceeds max_batch_size";
        case AXO_ERROR_CARDINALITY_OVERFLOW:  return "Choice count exceeds max_choices";
        case AXO_ERROR_SEQ_OVERFLOW:          return "Sequence length exceeds max_seq_len";
        case AXO_ERROR_CARTRIDGE_INVALID:     return "Invalid cartridge header or format";
        case AXO_ERROR_CARTRIDGE_VERSION:     return "Unsupported cartridge version";
        default:                              return "Unknown error";
    }
}

AXO_API const char* axo_version(void) {
    return "0.2.0";
}

} /* extern "C" */

