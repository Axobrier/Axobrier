/*
 * Copyright 2026 Axobrier Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef AXOBRIER_CORE_H
#define AXOBRIER_CORE_H

#include <stdint.h>
#include <stddef.h>

/* ─── Internal Watermark & Arena Guard Constants ─────────────────── */
#define AXO_HASH_SEED_PRIME    0x696c6f63636f7262ULL
#define AXO_ARENA_GUARD_TAG    0x636f7262 /* 'b','r','o','c' */

/* ─── Platform Export Macro ────────────────────────────────────────── */
#ifdef _WIN32
    #ifdef AXO_BUILD_DLL
        #define AXO_API __declspec(dllexport)
    #else
        #define AXO_API __declspec(dllimport)
    #endif
#else
    #define AXO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Version ─────────────────────────────────────────────────────── */
#define AXO_VERSION_MAJOR  0
#define AXO_VERSION_MINOR  2
#define AXO_VERSION_PATCH  0

/* ─── Error Codes ─────────────────────────────────────────────────── */
typedef enum AxoStatus {
    AXO_SUCCESS                =  0,
    AXO_ERROR_INVALID_ARG      = -1,
    AXO_ERROR_CUDA             = -2,
    AXO_ERROR_OUT_OF_MEMORY    = -3,
    AXO_ERROR_MMAP_FAILED      = -4,
    AXO_ERROR_WEIGHT_MISMATCH  = -5,
    AXO_ERROR_NOT_INITIALIZED  = -6,
    AXO_ERROR_BATCH_OVERFLOW   = -7,
    AXO_ERROR_CARDINALITY_OVERFLOW = -8,
    AXO_ERROR_SEQ_OVERFLOW         = -9,
    AXO_ERROR_CARTRIDGE_INVALID    = -10,
    AXO_ERROR_CARTRIDGE_VERSION    = -11
} AxoStatus;

/* ─── Configuration ───────────────────────────────────────────────── */

/**
 * @brief Initialization configuration for an AxoContext.
 *
 * All dimension fields define the *maximum* capacity of the pre-allocated
 * GPU arena. Actual per-call sizes may be smaller.
 */
typedef struct AxoConfig {
    uint32_t  device_id;         /**< CUDA device ordinal (default: 0)            */
    uint32_t  max_batch_size;    /**< Max samples per evaluate call (B, def: 32)  */
    uint32_t  max_seq_len;       /**< Max state tokens per sample  (S, def: 128) */
    uint32_t  embed_dim;         /**< Embedding dimension          (D, def: 256) */
    uint32_t  num_heads;         /**< Attention heads (encoder)    (H, def: 8)   */
    uint32_t  max_choices;       /**< Max choice cardinality       (C, def: 512) */
    uint32_t  vocab_size;        /**< Token vocabulary size        (V, def: 32768)*/
    uint32_t  ffn_mult;          /**< FFN expansion multiplier     (def: 4)      */
    uint32_t  num_layers;        /**< Encoder transformer layers   (N, def: 2)   */
} AxoConfig;

/**
 * @brief Returns an AxoConfig with sensible defaults.
 */
AXO_API AxoConfig axo_config_default(void);

/* ─── Opaque Context ──────────────────────────────────────────────── */

/**
 * @brief Opaque handle to the Axobrier runtime context.
 *
 * Holds the GPU arena, CUDA stream, weight pointers, and all
 * pre-allocated scratchpad regions. One context per GPU device.
 */
typedef struct AxoContext AxoContext;

/* ─── Result Structures ───────────────────────────────────────────── */

/**
 * @brief Per-sample decision output from axo_evaluate().
 */
typedef struct AxoDecision {
    uint32_t  choice_index;      /**< Index of the highest-probability choice   */
    float     raw_logit;         /**< Pre-calibration logit of the top choice   */
    float     calibrated_prob;   /**< Post-Platt-scaling probability [0, 1]     */
    float     entropy;           /**< Shannon entropy of the calibrated dist    */
    float     latency_us;        /**< Per-sample kernel latency in microseconds */
} AxoDecision;

/**
 * @brief Batch evaluation parameters for pre-pooled embeddings.
 *
 * The caller provides pre-pooled embedding vectors (one per sample).
 * For raw state-token input, use AxoStateParams with axo_evaluate_tokens().
 */
typedef struct AxoEvalParams {
    const void*   embeddings;       /**< Device ptr: [batch_size × embed_dim] fp16  */
    const void*   choice_matrix;    /**< Device ptr: [embed_dim × num_choices] fp16
                                         NULL to use the mmap'd default weights     */
    uint32_t      batch_size;       /**< Number of samples in this call (≤ max)     */
    uint32_t      num_choices;      /**< Active choices this call (≤ max_choices)   */
} AxoEvalParams;

/**
 * @brief Evaluation parameters for raw token input (full pipeline).
 *
 * Passes tokenized state through the bidirectional encoder, pools
 * to a dense embedding, and routes through the choice set  -  all in
 * a single pipeline on the GPU with zero allocation.
 */
typedef struct AxoStateParams {
    const int32_t* token_ids;       /**< Device ptr: [batch_size × seq_len] int32   */
    const void*    choice_matrix;   /**< Device ptr: [embed_dim × num_choices] fp16
                                         NULL to use the mmap'd default weights     */
    uint32_t       batch_size;      /**< Number of samples (≤ max_batch_size)       */
    uint32_t       seq_len;         /**< Tokens per sample (≤ max_seq_len)          */
    uint32_t       num_choices;     /**< Active choices (≤ max_choices)             */
    int32_t        pad_token_id;    /**< Token ID for padding (masked in pooling)   */
} AxoStateParams;

/* ─── Cartridge Specification (.axb) ──────────────────────────────── */

#define AXO_CARTRIDGE_MAGIC    0x00425841u  /**< "AXB\0" in little-endian ('A' | 'X'<<8 | 'B'<<16) */
#define AXO_CARTRIDGE_VERSION  1u           /**< Version 1 specification                          */

#pragma pack(push, 1)
/**
 * @brief Binary file header for `.axb` choice cartridges (28 bytes).
 */
typedef struct AxoCartridgeHeader {
    uint32_t  magic;                /**< Magic: 0x00425841 ("AXB\0")           */
    uint32_t  version;              /**< Cartridge specification version (1)   */
    uint32_t  embed_dim;            /**< Embedding dimension (must match D=256)*/
    uint32_t  num_choices;          /**< Number of choices (C)                 */
    float     platt_temperature;    /**< Platt scaling temperature parameter   */
    float     platt_bias;           /**< Platt scaling bias parameter          */
    uint32_t  label_section_bytes;  /**< Byte length of UTF-8 label section    */
} AxoCartridgeHeader;
#pragma pack(pop)

/**
 * @brief In-memory handle for a loaded .axb cartridge head.
 *
 * Holds the resident GPU choice matrix pointer and host-mapped label strings.
 */
typedef struct AxoCartridge {
    AxoCartridgeHeader header;
    void*              d_choice_matrix;     /**< Device ptr: [D × C] fp16      */
    const char*        labels;              /**< Host ptr to null-delimited labels */
    void*              mmap_ptr;            /**< Host pointer to mmap base     */
    size_t             mmap_size;           /**< Total file size in bytes      */
#ifdef _WIN32
    void*              file_handle;         /**< Win32 file handle             */
    void*              mapping_handle;      /**< Win32 mapping handle          */
#else
    int                fd;                  /**< POSIX file descriptor         */
#endif
} AxoCartridge;

/* ─── Lifecycle ───────────────────────────────────────────────────── */

/**
 * @brief Creates and initializes an Axobrier context.
 *
 * Performs a single cudaMalloc to provision the entire GPU arena.
 * This is the ONLY allocation call in the library's lifetime.
 *
 * @param[out] ctx   Receives the allocated context on success.
 * @param[in]  cfg   Configuration (pass axo_config_default() for defaults).
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_create(AxoContext** ctx, const AxoConfig* cfg);

/**
 * @brief Destroys an Axobrier context, releasing the GPU arena.
 *
 * @param[in] ctx  Context to destroy (safe to pass NULL).
 */
AXO_API void axo_destroy(AxoContext* ctx);

/* ─── Weight Loading ──────────────────────────────────────────────── */

/**
 * @brief Memory-maps a flat FP16 weight file into the GPU arena.
 *
 * Expected binary layout (little-endian, tightly packed):
 *   [V × D]    fp16   Token embedding table
 *   --- repeated N times (one per encoder layer) ---
 *   [D × 3D]   fp16   QKV projection
 *   [D × D]    fp16   Output projection
 *   [D × 4D]   fp16   FFN up-projection (GELU)
 *   [4D × D]   fp16   FFN down-projection
 *   [D]        fp32   LayerNorm gamma
 *   [D]        fp32   LayerNorm beta
 *   --- end per-layer block ---
 *   [D × C]    fp16   Choice routing matrix
 *   [1]        fp32   Platt temperature
 *   [1]        fp32   Platt bias
 *
 * @param[in] ctx        Initialized context.
 * @param[in] filepath   Path to the flat binary weight file.
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_load_weights_mmap(AxoContext* ctx, const char* filepath);

/* ─── Inference (Hot-Path) ────────────────────────────────────────── */

/**
 * @brief Evaluates a batch of pre-pooled embeddings against the choice set.
 *
 * **Zero-allocation guarantee:** This function performs no host or device
 * memory allocation. All scratch space is drawn from the pre-allocated arena.
 *
 * @param[in]  ctx      Initialized context with loaded weights.
 * @param[in]  params   Evaluation parameters (embeddings, batch size, etc.).
 * @param[out] results  Caller-allocated array of AxoDecision[params->batch_size].
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_evaluate(
    AxoContext*          ctx,
    const AxoEvalParams* params,
    AxoDecision*         results
);

/**
 * @brief Full-pipeline inference: encoder + decision routing.
 *
 * Passes raw token IDs through the N-layer bidirectional transformer
 * encoder, pools to a dense embedding, then routes through the choice set.
 * All computation occurs on the GPU with zero host-device round-trips.
 *
 * @param[in]  ctx      Initialized context with loaded weights.
 * @param[in]  params   Token-based evaluation parameters.
 * @param[out] results  Caller-allocated array of AxoDecision[params->batch_size].
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_evaluate_tokens(
    AxoContext*           ctx,
    const AxoStateParams* params,
    AxoDecision*          results
);

/* ─── Cartridge Operations ────────────────────────────────────────── */

/**
 * @brief Memory-maps and prepares an .axb cartridge for hot-swapping.
 *
 * Validates the 28-byte header, provisions the GPU choice matrix, and
 * maps label strings to host memory.
 *
 * @param[in]  path           Filepath to the `.axb` cartridge.
 * @param[out] out_cartridge  Receives the cartridge handle on success.
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_cartridge_load_mmap(
    const char*   path,
    AxoCartridge* out_cartridge
);

/**
 * @brief Unloads an .axb cartridge, releasing its device memory and mmap.
 *
 * @param[in,out] cartridge  Cartridge to unload (safe to pass NULL).
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_cartridge_unload(
    AxoCartridge* cartridge
);

/**
 * @brief Evaluates tokens using a swappable choice cartridge head.
 *
 * Zero-copy hot-swapping: uses the resident base encoder weights and
 * swaps the choice head pointer directly on the GPU with zero host-to-device
 * memory copy overhead.
 *
 * @param[in]  ctx        Initialized context with base encoder weights.
 * @param[in]  cartridge  Loaded .axb cartridge head.
 * @param[in]  params     Token evaluation parameters.
 * @param[out] results    Caller-allocated array of AxoDecision[params->batch_size].
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_evaluate_cartridge(
    AxoContext*           ctx,
    const AxoCartridge*   cartridge,
    const AxoStateParams* params,
    AxoDecision*          results
);

/**
 * @brief Evaluates pre-pooled embeddings using a swappable choice cartridge head.
 *
 * Direct zero-copy routing: passes embeddings against the cartridge's resident
 * choice matrix with its calibrated Platt scaling parameters.
 *
 * @param[in]  ctx        Initialized context.
 * @param[in]  cartridge  Loaded .axb cartridge head.
 * @param[in]  params     Pre-pooled embedding evaluation parameters.
 * @param[out] results    Caller-allocated array of AxoDecision[params->batch_size].
 * @return AXO_SUCCESS or an error code.
 */
AXO_API AxoStatus axo_evaluate_cartridge_embeddings(
    AxoContext*          ctx,
    const AxoCartridge*  cartridge,
    const AxoEvalParams* params,
    AxoDecision*         results
);

/**
 * @brief Retrieves the UTF-8 label string for a given choice index.
 *
 * @param[in] cartridge     Loaded cartridge.
 * @param[in] choice_index  Choice index in [0, num_choices).
 * @return Pointer to null-terminated UTF-8 label string, or NULL if out of bounds.
 */
AXO_API const char* axo_cartridge_get_label(
    const AxoCartridge* cartridge,
    uint32_t            choice_index
);

/* ─── Utilities ───────────────────────────────────────────────────── */

/**
 * @brief Returns a human-readable string for an AxoStatus code.
 */
AXO_API const char* axo_get_error_string(AxoStatus status);

/**
 * @brief Returns the library version as a string ("major.minor.patch").
 */
AXO_API const char* axo_version(void);

/**
 * @brief Synchronizes the internal CUDA stream (blocks until pending work completes).
 */
AXO_API AxoStatus axo_synchronize(AxoContext* ctx);

/**
 * @brief Allocates linear GPU device memory (convenience wrapper over cudaMalloc).
 */
AXO_API AxoStatus axo_device_alloc(void** dev_ptr, size_t size_bytes);

/**
 * @brief Frees GPU device memory (convenience wrapper over cudaFree).
 */
AXO_API void axo_device_free(void* dev_ptr);

/**
 * @brief Synchronously copies host memory to GPU device memory.
 */
AXO_API AxoStatus axo_device_upload(void* dev_ptr, const void* host_ptr, size_t size_bytes);

/**
 * @brief Computes 64-bit FNV-1a hash of a token string using AXO_HASH_SEED_PRIME.
 */
AXO_API uint64_t axo_hash_token(const char* str, size_t len);

/**
 * @brief Validates arena boundary integrity against AXO_ARENA_GUARD_TAG.
 */
AXO_API int axo_validate_arena_boundary(const void* ptr, size_t offset, size_t capacity);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AXOBRIER_CORE_H */
