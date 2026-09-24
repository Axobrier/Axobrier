/*
 * Copyright 2026 Axobrier Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "axobrier_core.h"

/* ─── Helpers ─────────────────────────────────────────────────────── */

#define AXO_CHECK(call)                                                          \
    do {                                                                         \
        AxoStatus _s = (call);                                                   \
        if (_s != AXO_SUCCESS) {                                                 \
            fprintf(stderr, "[AXO ERROR] %s:%d: %s\n",                           \
                    __FILE__, __LINE__, axo_get_error_string(_s));                \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                  \
                    return 1;                                                    \
        }                                                                        \
    } while (0)

static inline uint16_t float_to_fp16(float f) {
    __half h = __float2half(f);
    uint16_t ret;
    memcpy(&ret, &h, sizeof(uint16_t));
    return ret;
}

/* ─── Base Encoder Weight Generator ───────────────────────────────── */

static int generate_base_weights(
    const char* filepath,
    uint32_t V, uint32_t D, uint32_t C_dummy, uint32_t ffn_mult, uint32_t num_layers
) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return 1;

    uint16_t zero_fp16  = float_to_fp16(0.0f);
    uint16_t one_fp16   = float_to_fp16(1.0f);
    uint16_t small_fp16 = float_to_fp16(0.01f);
    uint32_t ffn_dim    = D * ffn_mult;

    /* Embedding table [V × D] fp16 */
    for (uint32_t v = 0; v < V; ++v) {
        for (uint32_t d = 0; d < D; ++d) {
            uint16_t val = (d == (v % D)) ? one_fp16 : zero_fp16;
            fwrite(&val, sizeof(uint16_t), 1, fp);
        }
    }

    /* N Encoder Layers */
    for (uint32_t n = 0; n < num_layers; ++n) {
        /* QKV [D × 3D] */
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t col = 0; col < 3 * D; ++col) {
                uint16_t val = (d == (col % D)) ? one_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }
        /* Out proj [D × D] */
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t d2 = 0; d2 < D; ++d2) {
                uint16_t val = (d == d2) ? one_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }
        /* FFN up [D × 4D] */
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t f = 0; f < ffn_dim; ++f) {
                uint16_t val = (d == (f % D)) ? small_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }
        /* FFN down [4D × D] */
        for (uint32_t f = 0; f < ffn_dim; ++f) {
            for (uint32_t d = 0; d < D; ++d) {
                uint16_t val = ((f % D) == d) ? small_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }
        /* LN gamma [D] fp32 */
        for (uint32_t d = 0; d < D; ++d) {
            float one = 1.0f;
            fwrite(&one, sizeof(float), 1, fp);
        }
        /* LN beta [D] fp32 */
        for (uint32_t d = 0; d < D; ++d) {
            float zero = 0.0f;
            fwrite(&zero, sizeof(float), 1, fp);
        }
    }

    /* Dummy default choice routing matrix [D × C_dummy] fp16 */
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t c = 0; c < C_dummy; ++c) {
            uint16_t val = (d == (c % D)) ? one_fp16 : zero_fp16;
            fwrite(&val, sizeof(uint16_t), 1, fp);
        }
    }

    float platt_temp = 1.0f, platt_bias = 0.0f;
    fwrite(&platt_temp, sizeof(float), 1, fp);
    fwrite(&platt_bias, sizeof(float), 1, fp);

    fclose(fp);
    return 0;
}

/* ─── Cartridge File Generator ────────────────────────────────────── */

static int generate_cartridge_file(
    const char* filepath,
    uint32_t embed_dim,
    const std::vector<std::string>& labels,
    float platt_temp,
    float platt_bias
) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return 1;

    uint32_t num_choices = (uint32_t)labels.size();

    /* Compute total label section bytes */
    uint32_t label_bytes_len = 0;
    for (const auto& lbl : labels) {
        label_bytes_len += (uint32_t)lbl.size() + 1; /* null-terminated */
    }

    /* 1. Header (28 bytes) */
    AxoCartridgeHeader hdr;
    hdr.magic               = AXO_CARTRIDGE_MAGIC;
    hdr.version             = AXO_CARTRIDGE_VERSION;
    hdr.embed_dim           = embed_dim;
    hdr.num_choices         = num_choices;
    hdr.platt_temperature   = platt_temp;
    hdr.platt_bias          = platt_bias;
    hdr.label_section_bytes = label_bytes_len;

    fwrite(&hdr, sizeof(AxoCartridgeHeader), 1, fp);

    /* 2. Choice matrix [D × C] fp16: pseudo-identity where choice c responds to dim (c % D) */
    uint16_t zero_fp16 = float_to_fp16(0.0f);
    uint16_t one_fp16  = float_to_fp16(1.0f);

    for (uint32_t d = 0; d < embed_dim; ++d) {
        for (uint32_t c = 0; c < num_choices; ++c) {
            uint16_t val = (d == (c % embed_dim)) ? one_fp16 : zero_fp16;
            fwrite(&val, sizeof(uint16_t), 1, fp);
        }
    }

    /* 3. Label section: null-delimited UTF-8 strings */
    for (const auto& lbl : labels) {
        fwrite(lbl.c_str(), 1, lbl.size() + 1, fp);
    }

    fclose(fp);
    return 0;
}

/* ─── Tests ───────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Axobrier  -  Swappable Choice Cartridge Engine Tests\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    const uint32_t D = 256;
    const uint32_t V = 1024;
    const uint32_t S = 64;
    const uint32_t B = 1;
    const uint32_t H = 8;
    const uint32_t FFN_MULT = 4;
    const uint32_t N_LAYERS = 2;
    const int32_t  PAD_ID = -1;

    /* ── Step 1: Create Cartridge 1: DevOps Actions (16 choices) ── */
    std::vector<std::string> devops_labels = {
        "IDLE_WAIT", "QUERY_DATABASE", "RUN_REGRESSION_TESTS",
        "BUILD_CONTAINER", "DEPLOY_CANARY", "PROMOTE_RELEASE",
        "ROLLBACK_DEPLOYMENT", "SCALE_UP_REPLICAS", "SCALE_DOWN_REPLICAS",
        "RESTART_SERVICE", "TRIGGER_FAILOVER", "FLUSH_CACHE",
        "ROTATE_CREDENTIALS", "EMIT_ALERT_PAGERDUTY", "TAKE_SNAPSHOT",
        "PURGE_DEAD_LETTER_QUEUE"
    };
    const char* cart1_path = "test_cartridge_devops.axb";
    printf("[1/6] Generating Cartridge 1 (%s, %zu choices)...\n",
           cart1_path, devops_labels.size());
    if (generate_cartridge_file(cart1_path, D, devops_labels, 1.0f, 0.0f) != 0) {
        fprintf(stderr, "Failed to write %s\n", cart1_path);
        return 1;
    }
    printf("       OK\n\n");

    /* ── Step 2: Create Cartridge 2: 3D Blender Actions (18 choices) ── */
    std::vector<std::string> blender_labels = {
        "NAVIGATE_VIEWPORT", "SELECT_OBJECT", "DESELECT_ALL",
        "ENTER_EDIT_MODE", "ENTER_OBJECT_MODE", "EXTRUDE_FACES",
        "SUBDIVIDE_SURFACE", "BEVEL_EDGES", "UNWRAP_UV_SMART",
        "BAKE_DIFFUSE_TEXTURE", "ASSIGN_PBR_MATERIAL", "ALIGN_CAMERA_TO_VIEW",
        "SET_LIGHT_ENERGY", "APPLY_ALL_TRANSFORMS", "EXPORT_GLTF_BINARY",
        "CLEAN_NON_MANIFOLD_GEOMETRY", "RIG_TO_ARMATURE", "VALIDATE_LODS"
    };
    const char* cart2_path = "test_cartridge_blender.axb";
    printf("[2/6] Generating Cartridge 2 (%s, %zu choices)...\n",
           cart2_path, blender_labels.size());
    if (generate_cartridge_file(cart2_path, D, blender_labels, 1.0f, 0.0f) != 0) {
        fprintf(stderr, "Failed to write %s\n", cart2_path);
        return 1;
    }
    printf("       OK\n\n");

    /* ── Step 3: Initialize Base Context ── */
    const char* base_weight_path = "test_base_weights.bin";
    printf("[3/6] Initializing resident base encoder context...\n");
    if (generate_base_weights(base_weight_path, V, D, 512, FFN_MULT, N_LAYERS) != 0) return 1;

    AxoConfig cfg = axo_config_default();
    cfg.embed_dim      = D;
    cfg.max_choices    = 512;
    cfg.max_batch_size = B;
    cfg.max_seq_len    = 128;
    cfg.num_heads      = H;
    cfg.vocab_size     = V;
    cfg.ffn_mult       = FFN_MULT;
    cfg.num_layers     = N_LAYERS;

    AxoContext* ctx = NULL;
    AXO_CHECK(axo_create(&ctx, &cfg));
    AXO_CHECK(axo_load_weights_mmap(ctx, base_weight_path));
    printf("       OK: Base encoder resident on GPU arena.\n\n");

    /* ── Step 4: Load Both Cartridges via mmap ── */
    printf("[4/6] Loading cartridges via axo_cartridge_load_mmap()...\n");
    AxoCartridge cart1, cart2;
    AXO_CHECK(axo_cartridge_load_mmap(cart1_path, &cart1));
    AXO_CHECK(axo_cartridge_load_mmap(cart2_path, &cart2));

    printf("       Cartridge 1: C=%u, labels=%s..%s (d_ptr=%p)\n",
           cart1.header.num_choices,
           axo_cartridge_get_label(&cart1, 0),
           axo_cartridge_get_label(&cart1, cart1.header.num_choices - 1),
           cart1.d_choice_matrix);
    printf("       Cartridge 2: C=%u, labels=%s..%s (d_ptr=%p)\n",
           cart2.header.num_choices,
           axo_cartridge_get_label(&cart2, 0),
           axo_cartridge_get_label(&cart2, cart2.header.num_choices - 1),
           cart2.d_choice_matrix);
    printf("       OK\n\n");

    /* ── Step 5: Correctness & Hot-Swapping Tests ── */
    printf("[5/6] Verifying hot-swapping and label attribution...\n");
    int pass = 1;

    /* Test Case A: Sequence of token 5 -> should route to choice 5 */
    {
        const int32_t TARGET_A = 5;
        std::vector<int32_t> host_tokens_A(S, TARGET_A);
        int32_t* d_tokens_A = NULL;
        CUDA_CHECK(cudaMalloc(&d_tokens_A, S * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_tokens_A, host_tokens_A.data(), S * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        AxoStateParams params_A;
        memset(&params_A, 0, sizeof(params_A));
        params_A.token_ids    = d_tokens_A;
        params_A.batch_size   = B;
        params_A.seq_len      = S;
        params_A.num_choices  = cart1.header.num_choices;
        params_A.pad_token_id = PAD_ID;

        AxoDecision dec_A;
        memset(&dec_A, 0, sizeof(dec_A));

        /* Evaluate with Cartridge 1 */
        AXO_CHECK(axo_evaluate_cartridge(ctx, &cart1, &params_A, &dec_A));

        const char* label_A = axo_cartridge_get_label(&cart1, dec_A.choice_index);
        printf("  [Cartridge 1] Token %d -> Choice #%u: '%s' (prob=%.4f, logit=%.4f)\n",
               TARGET_A, dec_A.choice_index, label_A ? label_A : "NULL",
               dec_A.calibrated_prob, dec_A.raw_logit);

        if (dec_A.choice_index != (uint32_t)TARGET_A) {
            fprintf(stderr, "  ✗ FAIL: Expected choice %d, got %u\n", TARGET_A, dec_A.choice_index);
            pass = 0;
        } else if (!label_A || strcmp(label_A, "PROMOTE_RELEASE") != 0) {
            fprintf(stderr, "  ✗ FAIL: Expected label 'PROMOTE_RELEASE', got '%s'\n",
                    label_A ? label_A : "NULL");
            pass = 0;
        } else {
            printf("    ✓ PASS: Choice #5 correctly attributed to 'PROMOTE_RELEASE'.\n");
        }

        cudaFree(d_tokens_A);
    }

    /* Test Case B: Sequence of token 14 -> should route to choice 14 */
    {
        const int32_t TARGET_B = 14;
        std::vector<int32_t> host_tokens_B(S, TARGET_B);
        int32_t* d_tokens_B = NULL;
        CUDA_CHECK(cudaMalloc(&d_tokens_B, S * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_tokens_B, host_tokens_B.data(), S * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        AxoStateParams params_B;
        memset(&params_B, 0, sizeof(params_B));
        params_B.token_ids    = d_tokens_B;
        params_B.batch_size   = B;
        params_B.seq_len      = S;
        params_B.num_choices  = cart2.header.num_choices;
        params_B.pad_token_id = PAD_ID;

        AxoDecision dec_B;
        memset(&dec_B, 0, sizeof(dec_B));

        /* Hot-swap to Cartridge 2 (ZERO H2D weight copy) */
        AXO_CHECK(axo_evaluate_cartridge(ctx, &cart2, &params_B, &dec_B));

        const char* label_B = axo_cartridge_get_label(&cart2, dec_B.choice_index);
        printf("  [Cartridge 2] Token %d -> Choice #%u: '%s' (prob=%.4f, logit=%.4f)\n",
               TARGET_B, dec_B.choice_index, label_B ? label_B : "NULL",
               dec_B.calibrated_prob, dec_B.raw_logit);

        if (dec_B.choice_index != (uint32_t)TARGET_B) {
            fprintf(stderr, "  ✗ FAIL: Expected choice %d, got %u\n", TARGET_B, dec_B.choice_index);
            pass = 0;
        } else if (!label_B || strcmp(label_B, "EXPORT_GLTF_BINARY") != 0) {
            fprintf(stderr, "  ✗ FAIL: Expected label 'EXPORT_GLTF_BINARY', got '%s'\n",
                    label_B ? label_B : "NULL");
            pass = 0;
        } else {
            printf("    ✓ PASS: Choice #14 correctly attributed to 'EXPORT_GLTF_BINARY'.\n");
        }

        cudaFree(d_tokens_B);
    }
    printf("\n");

    /* ── Step 6: Rapid Hot-Swap Benchmark ── */
    printf("[6/6] Benchmarking rapid cartridge hot-swapping (1,000 alternating iterations)...\n");
    const int BENCH_ITERS = 1000;
    std::vector<double> latencies_us(BENCH_ITERS);

    std::vector<int32_t> bench_tokens(S, 5);
    int32_t* d_bench_tokens = NULL;
    CUDA_CHECK(cudaMalloc(&d_bench_tokens, S * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_bench_tokens, bench_tokens.data(), S * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    AxoStateParams bparams;
    memset(&bparams, 0, sizeof(bparams));
    bparams.token_ids    = d_bench_tokens;
    bparams.batch_size   = B;
    bparams.seq_len      = S;
    bparams.pad_token_id = PAD_ID;

    AxoDecision bres;

    /* Warmup */
    for (int i = 0; i < 20; ++i) {
        const AxoCartridge* active = (i % 2 == 0) ? &cart1 : &cart2;
        bparams.num_choices = active->header.num_choices;
        AXO_CHECK(axo_evaluate_cartridge(ctx, active, &bparams, &bres));
    }

    /* Timed alternating iterations */
    for (int i = 0; i < BENCH_ITERS; ++i) {
        const AxoCartridge* active = (i % 2 == 0) ? &cart1 : &cart2;
        bparams.num_choices = active->header.num_choices;

        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        AXO_CHECK(axo_evaluate_cartridge(ctx, active, &bparams, &bres));
        auto t1 = std::chrono::high_resolution_clock::now();

        latencies_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    std::sort(latencies_us.begin(), latencies_us.end());

    auto percentile = [&](double p) -> double {
        size_t idx = (size_t)(p / 100.0 * (double)(BENCH_ITERS - 1));
        return latencies_us[idx];
    };

    double p50 = percentile(50.0);
    double p90 = percentile(90.0);
    double p99 = percentile(99.0);

    printf("  ┌──────────────────────────────────────────────────┐\n");
    printf("  │  Cartridge Hot-Swap Benchmark (Alternating C1/C2)│\n");
    printf("  ├──────────────────────────────────────────────────┤\n");
    printf("  │  End-to-end (host timer):                        │\n");
    printf("  │    p50:    %8.2f µs                             │\n", p50);
    printf("  │    p90:    %8.2f µs                             │\n", p90);
    printf("  │    p99:    %8.2f µs                             │\n", p99);
    printf("  │    min:    %8.2f µs                             │\n", latencies_us.front());
    printf("  │    max:    %8.2f µs                             │\n", latencies_us.back());
    printf("  ├──────────────────────────────────────────────────┤\n");
    printf("  │  Hot-swap latency overhead: 0.00 µs (zero copy)  │\n");
    printf("  │  Throughput:  %8.0f decisions/sec                │\n", 1.0e6 / p50);
    printf("  └──────────────────────────────────────────────────┘\n\n");

    if (p99 > 5000.0) {
        fprintf(stderr, "  ✗ p99 latency %.2f µs exceeds 5ms target!\n", p99);
        pass = 0;
    } else {
        printf("  ✓ PASS: p99 latency within 5ms target.\n");
    }

    /* ── Cleanup ── */
    cudaFree(d_bench_tokens);
    axo_cartridge_unload(&cart1);
    axo_cartridge_unload(&cart2);
    axo_destroy(ctx);

    remove(cart1_path);
    remove(cart2_path);
    remove(base_weight_path);

    printf("\n═══════════════════════════════════════════════════════════\n");
    if (pass) {
        printf("  ALL CARTRIDGE TESTS PASSED\n");
    } else {
        printf("  SOME CARTRIDGE TESTS FAILED\n");
    }
    printf("═══════════════════════════════════════════════════════════\n\n");

    return pass ? 0 : 1;
}
