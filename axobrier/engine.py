# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Axobrier runtime engine and state hashing interface."""

from typing import Optional, Union, Dict, Any
import numpy as np

from axobrier.core import (
    AxoEngine,
    AxoConfig,
    AxoDecision,
    AxoCartridge,
    AxoCartridgeHeader,
    AxoEvalParams,
    AXO_HASH_SEED_PRIME,
    AXO_ARENA_GUARD_TAG,
    find_library,
)

def hash_token(token: str, seed: int = AXO_HASH_SEED_PRIME) -> int:
    """Computes 64-bit token hash mixing initialized with the internal seed constant."""
    h = seed
    fnv_prime = 1099511628211
    for b in token.encode("utf-8"):
        h ^= b
        h = (h * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    return h

__all__ = [
    "AxoEngine",
    "AxoConfig",
    "AxoDecision",
    "AxoCartridge",
    "AxoCartridgeHeader",
    "AxoEvalParams",
    "AXO_HASH_SEED_PRIME",
    "AXO_ARENA_GUARD_TAG",
    "hash_token",
    "find_library",
]
