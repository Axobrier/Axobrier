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
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "axobrier_core.h"

/* ─── Helpers ─────────────────────────────────────────────────────── */

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err_));             \
        return 1;                                                          \
    }                                                                      \
} while(0)

#define AXO_CHECK(call) do {                                               \
    AxoStatus s_ = (call);                                                 \
    if (s_ != AXO_SUCCESS) {                                               \
        fprintf(stderr, "[AXO ERROR] %s:%d: %s\n",                         \
                __FILE__, __LINE__, axo_get_error_string(s_));              \
        return 1;                                                          \
    }                                                                      \
} while(0)

static uint16_t float_to_fp16(float val) {
    __half h = __float2half(val);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

/* ─── Mock Weight File Generator ──────────────────────────────────── */

static int generate_mock_weights(
    const char* filepath,
    uint32_t V, uint32_t D, uint32_t C, uint32_t ffn_mult, uint32_t num_layers,
    float platt_temp, float platt_bias
) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot create weight file: %s\n", filepath);
        return 1;
    }

    uint16_t zero_fp16 = float_to_fp16(0.0f);
    uint32_t ffn_dim   = D * ffn_mult;

    /* Embedding table [V × D] fp16  -  stub: all zeros */
    for (size_t i = 0; i < (size_t)V * D; ++i) {
        fwrite(&zero_fp16, sizeof(uint16_t), 1, fp);
    }

    /* Per-layer weights repeated num_layers times */
    for (uint32_t n = 0; n < num_layers; ++n) {
        /* QKV projection [D × 3D] fp16  -  stub: all zeros */
        for (size_t i = 0; i < (size_t)D * 3 * D; ++i) {
            fwrite(&zero_fp16, sizeof(uint16_t), 1, fp);
        }

        /* Output projection [D × D] fp16  -  stub: all zeros */
        for (size_t i = 0; i < (size_t)D * D; ++i) {
            fwrite(&zero_fp16, sizeof(uint16_t), 1, fp);
        }

        /* FFN up-projection [D × 4D] fp16  -  stub: all zeros */
        for (size_t i = 0; i < (size_t)D * ffn_dim; ++i) {
            fwrite(&zero_fp16, sizeof(uint16_t), 1, fp);
        }

        /* FFN down-projection [4D × D] fp16  -  stub: all zeros */
        for (size_t i = 0; i < (size_t)ffn_dim * D; ++i) {
            fwrite(&zero_fp16, sizeof(uint16_t), 1, fp);
        }

        /* LayerNorm gamma [D] fp32  -  stub: all ones */
        for (uint32_t i = 0; i < D; ++i) {
            float one = 1.0f;
            fwrite(&one, sizeof(float), 1, fp);
        }

        /* LayerNorm beta [D] fp32  -  stub: all zeros */
        for (uint32_t i = 0; i < D; ++i) {
            float zero = 0.0f;
            fwrite(&zero, sizeof(float), 1, fp);
        }
    }

    /* Choice routing matrix [D × C] fp16, column-major.
     * Element (d, c) is at index d * C + c.
     * Pseudo-identity: choice c responds to dimension c % D. */
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t c = 0; c < C; ++c) {
            uint16_t val = (d == (c % D)) ? float_to_fp16(1.0f) : zero_fp16;
            fwrite(&val, sizeof(uint16_t), 1, fp);
        }
    }

    /* Platt temperature [1] fp32 */
    fwrite(&platt_temp, sizeof(float), 1, fp);

    /* Platt bias [1] fp32 */
    fwrite(&platt_bias, sizeof(float), 1, fp);

    fclose(fp);
    return 0;
}

/* ─── Correctness Tests ───────────────────────────────────────────── */

static int run_correctness_tests(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Axobrier  -  Routing Kernel Correctness Tests\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    const uint32_t D = 256;
    const uint32_t C = 512;
    const uint32_t V = 256;
    const uint32_t FFN_MULT = 4;
    const uint32_t N_LAYERS = 2;
    const uint32_t B = 1;
    const uint32_t TARGET_CHOICE = 42;

    /* Step 1: Generate mock weight file */
    const char* weight_path = "test_routing_weights.bin";
    printf("[1/5] Generating mock weight file...\n");
    if (generate_mock_weights(weight_path, V, D, C, FFN_MULT, N_LAYERS, 1.0f, 0.0f) != 0) return 1;
    printf("       OK: %s written (D=%u, C=%u, N=%u, temp=1.0, bias=0.0)\n\n",
           weight_path, D, C, N_LAYERS);

    /* Step 2: Create Axo context */
    printf("[2/5] Creating AxoContext...\n");
    AxoConfig cfg = axo_config_default();
    cfg.embed_dim      = D;
    cfg.max_choices    = C;
    cfg.max_batch_size = B;
    cfg.vocab_size     = V;
    cfg.ffn_mult       = FFN_MULT;
    cfg.num_layers     = N_LAYERS;

    AxoContext* ctx = NULL;
    AXO_CHECK(axo_create(&ctx, &cfg));
    printf("       OK: Arena provisioned, stream created.\n\n");

    /* Step 3: Load weights via mmap */
    printf("[3/5] Loading weights via mmap...\n");
    AXO_CHECK(axo_load_weights_mmap(ctx, weight_path));
    printf("       OK: Weights loaded and cached.\n\n");

    /* Step 4: Prepare synthetic embedding on device.
     * One-hot at dimension TARGET_CHOICE % D = 42. */
    printf("[4/5] Preparing synthetic embedding (one-hot at dim %u)...\n",
           TARGET_CHOICE % D);

    std::vector<uint16_t> host_embed(D, float_to_fp16(0.0f));
    host_embed[TARGET_CHOICE % D] = float_to_fp16(1.0f);

    void* d_embed = NULL;
    CUDA_CHECK(cudaMalloc(&d_embed, D * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_embed, host_embed.data(), D * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    printf("       OK: Embedding uploaded to device.\n\n");

    /* Step 5: Evaluate */
    printf("[5/5] Running axo_evaluate()...\n");

    AxoEvalParams params;
    memset(&params, 0, sizeof(params));
    params.embeddings    = d_embed;
    params.choice_matrix = NULL;  /* use mmap'd default */
    params.batch_size    = B;
    params.num_choices   = C;

    AxoDecision result;
    memset(&result, 0, sizeof(result));
    AXO_CHECK(axo_evaluate(ctx, &params, &result));

    printf("       Kernel latency: %.2f µs\n", result.latency_us);
    printf("       Winner:  choice #%u  (expected: #%u)\n",
           result.choice_index, TARGET_CHOICE);
    printf("       Raw logit:       %.6f\n", result.raw_logit);
    printf("       Calibrated prob: %.6f\n", result.calibrated_prob);
    printf("       Entropy:         %.6f\n\n", result.entropy);

    /* ── Assertions ── */
    int pass = 1;

    if (result.choice_index != TARGET_CHOICE) {
        fprintf(stderr, "  ✗ FAIL: Expected choice #%u, got #%u\n",
                TARGET_CHOICE, result.choice_index);
        pass = 0;
    } else {
        printf("  ✓ PASS: Choice #%u wins.\n", TARGET_CHOICE);
    }

    if (result.calibrated_prob < 0.0f || result.calibrated_prob > 1.0f) {
        fprintf(stderr, "  ✗ FAIL: calibrated_prob %.6f not in [0, 1]\n",
                result.calibrated_prob);
        pass = 0;
    } else {
        printf("  ✓ PASS: calibrated_prob %.6f is in [0, 1].\n",
                result.calibrated_prob);
    }

    if (result.raw_logit < 0.5f) {
        fprintf(stderr, "  ✗ FAIL: raw_logit %.6f is unexpectedly low\n",
                result.raw_logit);
        pass = 0;
    } else {
        printf("  ✓ PASS: raw_logit %.6f is positive and reasonable.\n",
                result.raw_logit);
    }

    float uniform_prob = 1.0f / (float)C;
    if (result.calibrated_prob <= uniform_prob) {
        fprintf(stderr, "  ✗ FAIL: calibrated_prob %.6f <= uniform 1/C=%.6f\n",
                result.calibrated_prob, uniform_prob);
        pass = 0;
    } else {
        printf("  ✓ PASS: calibrated_prob %.6f > uniform %.6f (choice dominates).\n",
                result.calibrated_prob, uniform_prob);
    }

    /* Multi-target validation */
    printf("\n  Running multi-target validation (choices 0, 42, 100, 255)...\n");
    uint32_t test_targets[] = {0, 42, 100, 255};
    int multi_pass = 1;

    for (int t = 0; t < 4; ++t) {
        uint32_t target = test_targets[t];

        std::vector<uint16_t> emb(D, float_to_fp16(0.0f));
        emb[target % D] = float_to_fp16(1.0f);
        CUDA_CHECK(cudaMemcpy(d_embed, emb.data(), D * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));

        AxoDecision r;
        memset(&r, 0, sizeof(r));
        AXO_CHECK(axo_evaluate(ctx, &params, &r));

        if (r.choice_index != target) {
            fprintf(stderr, "    ✗ Target #%u: got #%u\n", target, r.choice_index);
            multi_pass = 0;
        } else {
            printf("    ✓ Target #%u: correct (prob=%.4f, logit=%.4f)\n",
                   target, r.calibrated_prob, r.raw_logit);
        }
    }
    if (!multi_pass) pass = 0;

    /* Verify softmax sum */
    printf("\n  Verifying softmax sum (CPU reference)...\n");
    {
        std::vector<uint16_t> emb(D, float_to_fp16(0.0f));
        emb[42 % D] = float_to_fp16(1.0f);
        CUDA_CHECK(cudaMemcpy(d_embed, emb.data(), D * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));

        AxoDecision r;
        memset(&r, 0, sizeof(r));
        AXO_CHECK(axo_evaluate(ctx, &params, &r));

        float expected_prob = expf(1.0f) / (expf(1.0f) + (float)(C - 1) * expf(0.0f));
        float diff = fabsf(r.calibrated_prob - expected_prob);

        if (diff > 1e-3f) {
            fprintf(stderr, "  ✗ FAIL: calibrated_prob %.6f != expected %.6f (diff=%.6f)\n",
                    r.calibrated_prob, expected_prob, diff);
            pass = 0;
        } else {
            printf("  ✓ PASS: calibrated_prob %.6f ≈ expected %.6f (diff=%.2e)\n",
                   r.calibrated_prob, expected_prob, diff);
        }
    }

    /* Cleanup */
    cudaFree(d_embed);
    axo_destroy(ctx);
    remove(weight_path);

    printf("\n═══════════════════════════════════════════════════════════\n");
    if (pass) {
        printf("  ALL CORRECTNESS TESTS PASSED\n");
    } else {
        printf("  SOME TESTS FAILED\n");
    }
    printf("═══════════════════════════════════════════════════════════\n\n");

    return pass ? 0 : 1;
}

/* ─── Throughput Benchmark ────────────────────────────────────────── */

static int run_benchmark(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Axobrier  -  Routing Kernel Throughput Benchmark\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    const uint32_t D = 256;
    const uint32_t C = 512;
    const uint32_t V = 256;
    const uint32_t FFN_MULT = 4;
    const uint32_t N_LAYERS = 2;
    const uint32_t B = 1;
    const int WARMUP_ITERS = 100;
    const int BENCH_ITERS  = 5000;

    const char* weight_path = "bench_routing_weights.bin";
    printf("[1/4] Generating weight file...\n");
    if (generate_mock_weights(weight_path, V, D, C, FFN_MULT, N_LAYERS, 1.0f, 0.0f) != 0) return 1;

    printf("[2/4] Initializing context...\n");
    AxoConfig cfg = axo_config_default();
    cfg.embed_dim      = D;
    cfg.max_choices    = C;
    cfg.max_batch_size = B;
    cfg.vocab_size     = V;
    cfg.ffn_mult       = FFN_MULT;
    cfg.num_layers     = N_LAYERS;

    AxoContext* ctx = NULL;
    AXO_CHECK(axo_create(&ctx, &cfg));
    AXO_CHECK(axo_load_weights_mmap(ctx, weight_path));

    std::vector<uint16_t> host_embed(D, float_to_fp16(0.0f));
    host_embed[42] = float_to_fp16(1.0f);

    void* d_embed = NULL;
    CUDA_CHECK(cudaMalloc(&d_embed, D * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_embed, host_embed.data(), D * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    AxoEvalParams params;
    memset(&params, 0, sizeof(params));
    params.embeddings    = d_embed;
    params.choice_matrix = NULL;
    params.batch_size    = B;
    params.num_choices   = C;

    AxoDecision result;

    printf("[3/4] Warming up (%d iterations)...\n", WARMUP_ITERS);
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        AXO_CHECK(axo_evaluate(ctx, &params, &result));
    }

    printf("[4/4] Benchmarking (%d iterations, B=%u, C=%u)...\n",
           BENCH_ITERS, B, C);

    std::vector<double> latencies_us(BENCH_ITERS);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        AXO_CHECK(axo_evaluate(ctx, &params, &result));
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    std::sort(latencies_us.begin(), latencies_us.end());

    auto percentile = [&](double p) -> double {
        size_t idx = (size_t)(p / 100.0 * (double)(BENCH_ITERS - 1));
        if (idx >= (size_t)BENCH_ITERS) idx = BENCH_ITERS - 1;
        return latencies_us[idx];
    };

    double p50 = percentile(50.0);
    double p90 = percentile(90.0);
    double p99 = percentile(99.0);
    double min_us = latencies_us.front();
    double max_us = latencies_us.back();
    float kernel_us = result.latency_us;

    printf("\n  ┌──────────────────────────────────────────────────┐\n");
    printf("  │  Benchmark Results (B=%u, C=%u, D=%u)          │\n", B, C, D);
    printf("  ├──────────────────────────────────────────────────┤\n");
    printf("  │  End-to-end (host timer):                        │\n");
    printf("  │    p50:  %10.2f µs                             │\n", p50);
    printf("  │    p90:  %10.2f µs                             │\n", p90);
    printf("  │    p99:  %10.2f µs                             │\n", p99);
    printf("  │    min:  %10.2f µs                             │\n", min_us);
    printf("  │    max:  %10.2f µs                             │\n", max_us);
    printf("  ├──────────────────────────────────────────────────┤\n");
    printf("  │  Kernel-only (CUDA events, last run):            │\n");
    printf("  │    latency: %8.2f µs                           │\n", kernel_us);
    printf("  ├──────────────────────────────────────────────────┤\n");
    printf("  │  Throughput: %10.0f decisions/sec              │\n",
           1e6 / p50 * (double)B);
    printf("  └──────────────────────────────────────────────────┘\n\n");

    cudaFree(d_embed);
    axo_destroy(ctx);
    remove(weight_path);

    if (p99 > 5000.0) {
        fprintf(stderr, "  ⚠ WARNING: p99 latency %.2f µs exceeds 5ms target!\n", p99);
        return 1;
    }

    printf("  ✓ p99 latency within 5ms target.\n\n");
    return 0;
}

/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    printf("\nAxobrier v%s  -  Routing Kernel Test Harness\n\n", axo_version());

    bool do_correctness = true;
    bool do_benchmark   = true;

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--correctness") {
            do_benchmark = false;
        } else if (arg == "--benchmark") {
            do_correctness = false;
        }
    }

    int ret = 0;
    if (do_correctness) {
        ret = run_correctness_tests();
        if (ret != 0) return ret;
    }
    if (do_benchmark) {
        ret = run_benchmark();
        if (ret != 0) return ret;
    }
    return 0;
}
