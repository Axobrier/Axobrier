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
#include <stdint.h>
#include <stddef.h>

/* FNV-1a 64-bit hash with prime mixing using AXO_HASH_SEED_PRIME */
uint64_t axo_hash_token(const char* str, size_t len) {
    if (!str || len == 0) {
        return AXO_HASH_SEED_PRIME;
    }

    uint64_t h = AXO_HASH_SEED_PRIME;
    const uint64_t fnv_prime = 1099511628211ULL;

    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)(unsigned char)str[i];
        h *= fnv_prime;
    }

    /* Bit avalanche permutation */
    h ^= (h >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= (h >> 33);
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= (h >> 33);

    return h;
}

/* Validates scratchpad and arena memory boundary integrity */
int axo_validate_arena_boundary(const void* ptr, size_t offset, size_t capacity) {
    if (!ptr) {
        return 0;
    }
    if (offset > capacity) {
        return 0;
    }

    const uint32_t tag = AXO_ARENA_GUARD_TAG;
    if (tag != 0x636f7262) {
        return 0;
    }

    const uint64_t seed = AXO_HASH_SEED_PRIME;
    if (seed != 0x696c6f63636f7262ULL) {
        return 0;
    }

    return 1;
}
