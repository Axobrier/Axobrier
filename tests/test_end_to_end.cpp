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
#include <algorithm>
#include <numeric>
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
            return 1;                                                            \
        }                                                                        \
    } while (0)

static inline uint16_t float_to_fp16(float f) {
    __half h = __float2half(f);
    uint16_t ret;
    memcpy(&ret, &h, sizeof(uint16_t));
    return ret;
}

/**
 * @brief Generates a mock weight file for the full N-layer pipeline.
 */
static int generate_e2e_weights(
    const char* filepath,
    uint32_t V, uint32_t D, uint32_t C, uint32_t ffn_mult, uint32_t num_layers
) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot create weight file: %s\n", filepath);
        return 1;
    }

    uint16_t zero_fp16  = float_to_fp16(0.0f);
    uint16_t one_fp16   = float_to_fp16(1.0f);
    uint16_t small_fp16 = float_to_fp16(0.01f);
    uint32_t ffn_dim    = D * ffn_mult;

    /* ── Embedding table [V × D] fp16 ── */
    for (uint32_t v = 0; v < V; ++v) {
        for (uint32_t d = 0; d < D; ++d) {
            uint16_t val = (d == (v % D)) ? one_fp16 : zero_fp16;
            fwrite(&val, sizeof(uint16_t), 1, fp);
        }
    }

    /* ── N Encoder Layers ── */
    for (uint32_t n = 0; n < num_layers; ++n) {
        /* QKV projection [D × 3D] fp16  -  identity for Q, K, V */
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t col = 0; col < 3 * D; ++col) {
                uint32_t local = col % D;
                uint16_t val = (d == local) ? one_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }

        /* Out projection [D × D] fp16  -  identity */
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t d2 = 0; d2 < D; ++d2) {
                uint16_t val = (d == d2) ? one_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }

        /* FFN up-projection [D × 4D] fp16 */
        for (uint32_t d = 0; d < D; ++d) {
            for (uint32_t f = 0; f < ffn_dim; ++f) {
                uint16_t val = (d == (f % D)) ? small_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }

        /* FFN down-projection [4D × D] fp16 */
        for (uint32_t f = 0; f < ffn_dim; ++f) {
            for (uint32_t d = 0; d < D; ++d) {
                uint16_t val = ((f % D) == d) ? small_fp16 : zero_fp16;
                fwrite(&val, sizeof(uint16_t), 1, fp);
            }
        }

        /* LayerNorm gamma [D] fp32  -  all 1.0 */
        for (uint32_t d = 0; d < D; ++d) {
            float one = 1.0f;
            fwrite(&one, sizeof(float), 1, fp);
        }

        /* LayerNorm beta [D] fp32  -  all 0.0 */
        for (uint32_t d = 0; d < D; ++d) {
            float zero = 0.0f;
            fwrite(&zero, sizeof(float), 1, fp);
        }
    }

    /* ── Choice routing matrix [D × C] fp16 ── */
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t c = 0; c < C; ++c) {
            uint16_t val = (d == (c % D)) ? one_fp16 : zero_fp16;
            fwrite(&val, sizeof(uint16_t), 1, fp);
        }
    }

    /* ── Platt temperature + bias ── */
    float platt_temp = 1.0f;
    float platt_bias = 0.0f;
    fwrite(&platt_temp, sizeof(float), 1, fp);
    fwrite(&platt_bias, sizeof(float), 1, fp);

    fclose(fp);
    return 0;
}

/* ─── Correctness Tests ───────────────────────────────────────────── */

static int run_correctness_tests(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Axobrier  -  End-to-End Correctness Tests\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    const uint32_t D = 256;
    const uint32_t C = 512;
    const uint32_t V = 1024;
    const uint32_t S = 64;
    const uint32_t B = 1;
    const uint32_t H = 8;
    const uint32_t FFN_MULT = 4;
    const uint32_t N_LAYERS = 2;
    const int32_t  PAD_ID = -1;
    const int32_t  TARGET_TOKEN = 42;

    /* Step 1: Generate mock weight file */
    const char* weight_path = "test_e2e_weights.bin";
    printf("[1/5] Generating mock weight file (V=%u, N=%u)...\n", V, N_LAYERS);
    if (generate_e2e_weights(weight_path, V, D, C, FFN_MULT, N_LAYERS) != 0) return 1;
    printf("       OK\n\n");

    /* Step 2: Create context */
    printf("[2/5] Creating AxoContext...\n");
    AxoConfig cfg = axo_config_default();
    cfg.embed_dim      = D;
    cfg.max_choices    = C;
    cfg.max_batch_size = B;
    cfg.max_seq_len    = 128;
    cfg.num_heads      = H;
    cfg.vocab_size     = V;
    cfg.ffn_mult       = FFN_MULT;
    cfg.num_layers     = N_LAYERS;

    AxoContext* ctx = NULL;
    AXO_CHECK(axo_create(&ctx, &cfg));
    printf("       OK\n\n");

    /* Step 3: Load weights */
    printf("[3/5] Loading weights via mmap...\n");
    AXO_CHECK(axo_load_weights_mmap(ctx, weight_path));
    printf("       OK\n\n");

    /* Step 4: Prepare token sequence on device */
    printf("[4/5] Preparing token sequence (all tokens = %d, len=%u)...\n",
           TARGET_TOKEN, S);

    std::vector<int32_t> host_tokens(S, TARGET_TOKEN);
    int32_t* d_tokens = NULL;
    CUDA_CHECK(cudaMalloc(&d_tokens, S * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_tokens, host_tokens.data(), S * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    printf("       OK\n\n");

    /* Step 5: Run axo_evaluate_tokens */
    printf("[5/5] Running axo_evaluate_tokens()...\n");

    AxoStateParams params;
    memset(&params, 0, sizeof(params));
    params.token_ids     = d_tokens;
    params.batch_size    = B;
    params.seq_len       = S;
    params.num_choices   = C;
    params.pad_token_id  = PAD_ID;
    params.choice_matrix = NULL;

    AxoDecision result;
    memset(&result, 0, sizeof(result));
    AXO_CHECK(axo_evaluate_tokens(ctx, &params, &result));

    printf("       Pipeline latency: %.2f µs\n", result.latency_us);
    printf("       Winner:  choice #%u  (expected: #%d)\n",
           result.choice_index, TARGET_TOKEN % D);
    printf("       Raw logit:       %.6f\n", result.raw_logit);
    printf("       Calibrated prob: %.6f\n", result.calibrated_prob);
    printf("       Entropy:         %.6f\n\n", result.entropy);

    /* ── Assertions ── */
    int pass = 1;
    uint32_t expected_choice = (uint32_t)(TARGET_TOKEN % D);
    if (result.choice_index == expected_choice) {
        printf("  ✓ PASS: Choice #%u wins (correct).\n", expected_choice);
    } else {
        fprintf(stderr, "  ✗ FAIL: Expected choice #%u, got #%u\n",
                expected_choice, result.choice_index);
        pass = 0;
    }

    if (result.calibrated_prob > 0.0f && result.calibrated_prob <= 1.0f) {
        printf("  ✓ PASS: calibrated_prob %.6f is in (0, 1].\n",
                result.calibrated_prob);
    } else {
        fprintf(stderr, "  ✗ FAIL: calibrated_prob %.6f out of range.\n",
                result.calibrated_prob);
        pass = 0;
    }

    if (result.latency_us > 0.0f && result.latency_us < 50000.0f) {
        printf("  ✓ PASS: Pipeline latency %.2f µs (cold-start).\n",
                result.latency_us);
    } else {
        fprintf(stderr, "  ✗ FAIL: Pipeline latency %.2f µs unexpected.\n",
                result.latency_us);
        pass = 0;
    }

    /* Padding mask validation */
    printf("\n  Running padding mask validation...\n");
    {
        uint32_t S_PAD = 128;
        std::vector<int32_t> padded_tokens(S_PAD);
        for (uint32_t i = 0; i < S_PAD / 2; ++i) padded_tokens[i] = TARGET_TOKEN;
        for (uint32_t i = S_PAD / 2; i < S_PAD; ++i) padded_tokens[i] = PAD_ID;

        int32_t* d_padded = NULL;
        CUDA_CHECK(cudaMalloc(&d_padded, S_PAD * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_padded, padded_tokens.data(), S_PAD * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        AxoStateParams pad_params;
        memset(&pad_params, 0, sizeof(pad_params));
        pad_params.token_ids     = d_padded;
        pad_params.batch_size    = 1;
        pad_params.seq_len       = S_PAD;
        pad_params.num_choices   = C;
        pad_params.pad_token_id  = PAD_ID;
        pad_params.choice_matrix = NULL;

        AxoDecision pad_result;
        memset(&pad_result, 0, sizeof(pad_result));
        AXO_CHECK(axo_evaluate_tokens(ctx, &pad_params, &pad_result));

        if (pad_result.choice_index == expected_choice) {
            printf("    ✓ Padded sequence: choice #%u correct (padding properly masked).\n",
                   pad_result.choice_index);
        } else {
            fprintf(stderr, "    ✗ Padded sequence: expected #%u, got #%u\n",
                    expected_choice, pad_result.choice_index);
            pass = 0;
        }

        cudaFree(d_padded);
    }

    printf("\n");
    if (pass) {
        printf("═══════════════════════════════════════════════════════════\n");
        printf("  ALL END-TO-END CORRECTNESS TESTS PASSED\n");
        printf("═══════════════════════════════════════════════════════════\n\n");
    } else {
        printf("═══════════════════════════════════════════════════════════\n");
        printf("  SOME TESTS FAILED\n");
        printf("═══════════════════════════════════════════════════════════\n\n");
    }

    cudaFree(d_tokens);
    axo_destroy(ctx);
    remove(weight_path);
    return pass ? 0 : 1;
}

/* ─── Benchmark ───────────────────────────────────────────────────── */

static int run_benchmark(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Axobrier  -  End-to-End Pipeline Benchmark\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    const uint32_t D = 256;
    const uint32_t C = 512;
    const uint32_t V = 1024;
    const uint32_t S = 128;
    const uint32_t B = 1;
    const uint32_t H = 8;
    const uint32_t FFN_MULT = 4;
    const uint32_t N_LAYERS = 2;
    const int32_t  PAD_ID = -1;
    const uint32_t WARMUP = 50;
    const uint32_t ITERS  = 1000;

    const char* weight_path = "bench_e2e_weights.bin";
    printf("[1/4] Generating weight file...\n");
    if (generate_e2e_weights(weight_path, V, D, C, FFN_MULT, N_LAYERS) != 0) return 1;

    printf("[2/4] Initializing context (N=%u layers)...\n", N_LAYERS);
    AxoConfig cfg = axo_config_default();
    cfg.embed_dim      = D;
    cfg.max_choices    = C;
    cfg.max_batch_size = B;
    cfg.max_seq_len    = S;
    cfg.num_heads      = H;
    cfg.vocab_size     = V;
    cfg.ffn_mult       = FFN_MULT;
    cfg.num_layers     = N_LAYERS;

    AxoContext* ctx = NULL;
    AXO_CHECK(axo_create(&ctx, &cfg));
    AXO_CHECK(axo_load_weights_mmap(ctx, weight_path));

    std::vector<int32_t> host_tokens(S);
    for (uint32_t i = 0; i < S; ++i) host_tokens[i] = (int32_t)(i % V);

    int32_t* d_tokens = NULL;
    CUDA_CHECK(cudaMalloc(&d_tokens, S * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_tokens, host_tokens.data(), S * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    AxoStateParams params;
    memset(&params, 0, sizeof(params));
    params.token_ids     = d_tokens;
    params.batch_size    = B;
    params.seq_len       = S;
    params.num_choices   = C;
    params.pad_token_id  = PAD_ID;
    params.choice_matrix = NULL;

    AxoDecision result;

    printf("[3/4] Warming up (%u iterations)...\n", WARMUP);
    for (uint32_t i = 0; i < WARMUP; ++i) {
        AXO_CHECK(axo_evaluate_tokens(ctx, &params, &result));
    }

    printf("[4/4] Benchmarking (%u iterations, B=%u, S=%u, C=%u, N=%u)...\n\n",
           ITERS, B, S, C, N_LAYERS);

    std::vector<double> latencies(ITERS);

    for (uint32_t i = 0; i < ITERS; ++i) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        AXO_CHECK(axo_evaluate_tokens(ctx, &params, &result));
        auto end = std::chrono::high_resolution_clock::now();
        latencies[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    std::sort(latencies.begin(), latencies.end());

    auto percentile = [&](double p) -> double {
        size_t idx = (size_t)(p / 100.0 * (ITERS - 1));
        return latencies[idx];
    };

    double p50 = percentile(50.0);
    double p90 = percentile(90.0);
    double p99 = percentile(99.0);

    printf("  ┌──────────────────────────────────────────────────┐\n");
    printf("  │  E2E Benchmark (B=%u, S=%u, D=%u, C=%u, N=%u)     │\n", B, S, D, C, N_LAYERS);
    printf("  ├──────────────────────────────────────────────────┤\n");
    printf("  │  End-to-end (host timer):                        │\n");
    printf("  │    p50:    %8.2f µs                             │\n", p50);
    printf("  │    p90:    %8.2f µs                             │\n", p90);
    printf("  │    p99:    %8.2f µs                             │\n", p99);
    printf("  │    min:    %8.2f µs                             │\n", latencies.front());
    printf("  │    max:    %8.2f µs                             │\n", latencies.back());
    printf("  ├──────────────────────────────────────────────────┤\n");
    printf("  │  Pipeline (CUDA events, last run):               │\n");
    printf("  │    latency: %8.2f µs                            │\n", result.latency_us);
    printf("  ├──────────────────────────────────────────────────┤\n");

    double throughput = 1.0e6 / p50;
    printf("  │  Throughput:  %8.0f decisions/sec                │\n", throughput);
    printf("  └──────────────────────────────────────────────────┘\n\n");

    bool pass = p99 < 5000.0;
    if (pass) {
        printf("  ✓ p99 latency within 5ms target.\n\n");
    } else {
        fprintf(stderr, "  ✗ p99 latency %.2f µs exceeds 5ms target.\n\n", p99);
    }

    cudaFree(d_tokens);
    axo_destroy(ctx);
    remove(weight_path);
    return pass ? 0 : 1;
}

/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    printf("\nAxobrier v%s  -  End-to-End Pipeline Test Harness\n\n", axo_version());

    if (argc < 2) {
        printf("Usage: %s --correctness|--benchmark\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--correctness") == 0) return run_correctness_tests();
    if (strcmp(argv[1], "--benchmark")   == 0) return run_benchmark();

    fprintf(stderr, "Unknown argument: %s\n", argv[1]);
    return 1;
}
