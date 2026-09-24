#!/usr/bin/env python3
# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import ctypes
import numpy as np

# Add repo root to import text_to_embedding and tools
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from training.train_cartridge import text_to_embedding

# ═══════════════════════════════════════════════════════════════════
#  C ABI STRUCTURES & BINDINGS
# ═══════════════════════════════════════════════════════════════════

class AxoConfig(ctypes.Structure):
    _fields_ = [
        ("device_id", ctypes.c_uint32),
        ("max_batch_size", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("embed_dim", ctypes.c_uint32),
        ("num_heads", ctypes.c_uint32),
        ("max_choices", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("ffn_mult", ctypes.c_uint32),
        ("num_layers", ctypes.c_uint32),
    ]

class AxoDecision(ctypes.Structure):
    _fields_ = [
        ("choice_index", ctypes.c_uint32),
        ("raw_logit", ctypes.c_float),
        ("calibrated_prob", ctypes.c_float),
        ("entropy", ctypes.c_float),
        ("latency_us", ctypes.c_float),
    ]

class AxoEvalParams(ctypes.Structure):
    _fields_ = [
        ("embeddings", ctypes.c_void_p),
        ("choice_matrix", ctypes.c_void_p),
        ("batch_size", ctypes.c_uint32),
        ("num_choices", ctypes.c_uint32),
    ]

class AxoCartridgeHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("embed_dim", ctypes.c_uint32),
        ("num_choices", ctypes.c_uint32),
        ("platt_temperature", ctypes.c_float),
        ("platt_bias", ctypes.c_float),
        ("label_section_bytes", ctypes.c_uint32),
    ]

class AxoCartridge(ctypes.Structure):
    _fields_ = [
        ("header", AxoCartridgeHeader),
        ("d_choice_matrix", ctypes.c_void_p),
        ("labels", ctypes.c_char_p),
        ("mmap_ptr", ctypes.c_void_p),
        ("mmap_size", ctypes.c_size_t),
        ("file_handle", ctypes.c_void_p),
        ("mapping_handle", ctypes.c_void_p),
    ]


def load_axobrier_dll() -> ctypes.CDLL:
    dll_candidates = [
        os.path.join(REPO_ROOT, "build", "Release", "axobrier_core.dll"),
        os.path.join(REPO_ROOT, "build", "axobrier_core.dll"),
        os.path.join(REPO_ROOT, "build", "libaxobrier_core.so"),
    ]
    for path in dll_candidates:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise FileNotFoundError(f"Cannot find axobrier_core library in {dll_candidates}")


# ═══════════════════════════════════════════════════════════════════
#  UNSEEN EVALUATION BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

MESH_PBR_TESTS = [
    {
        "query": "Shoulder joint deformation artifact: vertex is weighted to 5 bones simultaneously in avatar rig.",
        "expected_label": "MAX_4_BONE_INFLUENCES"
    },
    {
        "query": "Baking PBR channels: pack ambient occlusion into red, roughness into green, and metalness into blue.",
        "expected_label": "ORM_TEXTURE_PACKING"
    },
    {
        "query": "Transparent character hair cards flickering and sorting behind other geometry in the scene.",
        "expected_label": "ALPHA_MODE_MASK_FOR_HAIR"
    },
    {
        "query": "Avatar mesh export setup: export rigged model as OpenCOLLADA .dae with SL avatar matrix compatibility.",
        "expected_label": "DAE_ARMATURE_RIGID_EXPORT"
    },
    {
        "query": "glTF 2.0 material requires tangent space normal map authored in a separate RGB texture.",
        "expected_label": "GLTF_PBR_SEPARATE_NORMAL"
    },
    {
        "query": "Collision shape rejected by Havok: concave physics model exceeds 256 convex hulls limit.",
        "expected_label": "CONVEX_HULL_PHYSICS_DECOMP"
    },
    {
        "query": "High download land impact: author manual 4-level LOD reduction with lowest LOD under 30 triangles.",
        "expected_label": "LOD_4_REDUCTION_STRATEGY"
    },
    {
        "query": "Mesh upload failed: draw call has 74,000 vertices, exceeding the 65,536 hardware index buffer limit.",
        "expected_label": "LIMIT_65K_VERTICES_PER_SURFACE"
    }
]

RUNTIME_PERF_TESTS = [
    {
        "query": "Frame rate micro-stutter: allocating new List<int>() every tick inside MonoBehaviour.Update().",
        "expected_label": "GC_ALLOC_IN_UPDATE_LOOP"
    },
    {
        "query": "Database execution plan diagnostic: sequential full table scan on transactions table without B-tree index.",
        "expected_label": "UNINDEXED_DATABASE_QUERY"
    },
    {
        "query": "Worker thread freeze: synchronous File.ReadAllBytes() called directly on the main UI event loop.",
        "expected_label": "BLOCKING_IO_ON_EVENT_LOOP"
    },
    {
        "query": "Operating system resource exhaustion: creating naked Thread().start() per connection leads to 5,000 threads.",
        "expected_label": "UNBOUNDED_THREAD_SPAWNING"
    },
    {
        "query": "Hot path memory churn: string += operator inside 100,000-iteration loop allocates 300MB of temporary strings.",
        "expected_label": "HOT_PATH_STRING_CONCAT"
    },
    {
        "query": "ORM performance alert: iterating over orders triggers 100 separate child SQL queries due to lazy loading.",
        "expected_label": "N_PLUS_ONE_ORM_FETCH"
    },
    {
        "query": "Retained memory leak: event listener attached to global message bus is never unregistered when component unmounts.",
        "expected_label": "LEAKED_EVENT_SUBSCRIPTION"
    },
    {
        "query": "Production server OOM crash: in-memory dictionary cache has no eviction policy and grows indefinitely with user sessions.",
        "expected_label": "UNBOUNDED_MEMORY_CACHE"
    }
]


def run_tests():
    print("===========================================================")
    print("  Axobrier - Production Cartridge End-to-End Validation")
    print("===========================================================\n")

    axo = load_axobrier_dll()

    # Configure signatures
    axo.axo_config_default.restype = AxoConfig
    axo.axo_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(AxoConfig)]
    axo.axo_create.restype = ctypes.c_int
    axo.axo_destroy.argtypes = [ctypes.c_void_p]
    axo.axo_cartridge_load_mmap.argtypes = [ctypes.c_char_p, ctypes.POINTER(AxoCartridge)]
    axo.axo_cartridge_load_mmap.restype = ctypes.c_int
    axo.axo_cartridge_unload.argtypes = [ctypes.POINTER(AxoCartridge)]
    axo.axo_cartridge_unload.restype = ctypes.c_int
    axo.axo_cartridge_get_label.argtypes = [ctypes.POINTER(AxoCartridge), ctypes.c_uint32]
    axo.axo_cartridge_get_label.restype = ctypes.c_char_p
    axo.axo_evaluate_cartridge_embeddings.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AxoCartridge),
        ctypes.POINTER(AxoEvalParams),
        ctypes.POINTER(AxoDecision)
    ]
    axo.axo_evaluate_cartridge_embeddings.restype = ctypes.c_int

    axo.axo_device_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    axo.axo_device_alloc.restype = ctypes.c_int
    axo.axo_device_free.argtypes = [ctypes.c_void_p]
    axo.axo_device_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    axo.axo_device_upload.restype = ctypes.c_int

    # 1. Initialize AxoContext
    print("[1/4] Initializing AxoContext...")
    cfg = axo.axo_config_default()
    ctx_ptr = ctypes.c_void_p()
    err = axo.axo_create(ctypes.byref(ctx_ptr), ctypes.byref(cfg))
    if err != 0:
        raise RuntimeError(f"axo_create failed with code {err}")
    print("       OK: GPU Arena created.\n")

    # 2. Load Cartridges
    print("[2/4] Loading production cartridges via axo_cartridge_load_mmap()...")
    cart_mesh = AxoCartridge()
    cart_perf = AxoCartridge()

    mesh_path = os.path.join(REPO_ROOT, "cartridge_mesh_pbr.axb")
    perf_path = os.path.join(REPO_ROOT, "cartridge_runtime_perf.axb")

    err = axo.axo_cartridge_load_mmap(mesh_path.encode("utf-8"), ctypes.byref(cart_mesh))
    if err != 0:
        raise RuntimeError(f"Failed to load {mesh_path}: code {err}")

    err = axo.axo_cartridge_load_mmap(perf_path.encode("utf-8"), ctypes.byref(cart_perf))
    if err != 0:
        raise RuntimeError(f"Failed to load {perf_path}: code {err}")

    print(f"       [OK] Loaded {mesh_path} (C={cart_mesh.header.num_choices}, Platt temp={cart_mesh.header.platt_temperature:.3f})")
    print(f"       [OK] Loaded {perf_path} (C={cart_perf.header.num_choices}, Platt temp={cart_perf.header.platt_temperature:.3f})\n")

    # Helper to evaluate test queries on a cartridge
    def evaluate_suite(cartridge: AxoCartridge, test_cases: list, domain_name: str):
        print(f"Evaluating unseen queries on {domain_name} ({len(test_cases)} tests):")
        all_passed = True

        for i, tc in enumerate(test_cases):
            query = tc["query"]
            expected = tc["expected_label"]

            # Compute embedding on host -> convert to fp16
            emb_tensor = text_to_embedding(query, embed_dim=256)
            emb_fp16 = emb_tensor.numpy().astype(np.float16)

            # Upload to device using axo_device_alloc / axo_device_upload
            d_emb = ctypes.c_void_p()
            nbytes = emb_fp16.nbytes
            axo.axo_device_alloc(ctypes.byref(d_emb), nbytes)
            axo.axo_device_upload(d_emb, emb_fp16.ctypes.data_as(ctypes.c_void_p), nbytes)

            # Prepare params
            params = AxoEvalParams()
            params.embeddings = d_emb
            params.choice_matrix = None
            params.batch_size = 1
            params.num_choices = cartridge.header.num_choices

            decision = AxoDecision()
            status = axo.axo_evaluate_cartridge_embeddings(
                ctx_ptr, ctypes.byref(cartridge), ctypes.byref(params), ctypes.byref(decision)
            )
            axo.axo_device_free(d_emb)

            if status != 0:
                print(f"  [!] Evaluation failed: code {status}")
                all_passed = False
                continue

            # Retrieve predicted label
            pred_label_raw = axo.axo_cartridge_get_label(ctypes.byref(cartridge), decision.choice_index)
            pred_label = pred_label_raw.decode("utf-8") if pred_label_raw else "UNKNOWN"

            prob = decision.calibrated_prob
            latency = decision.latency_us

            is_correct = (pred_label == expected)
            is_calibrated = (prob >= 0.85)
            status_symbol = "[PASS]" if (is_correct and is_calibrated) else "[FAIL]"

            print(f"  {status_symbol} [{i+1}/{len(test_cases)}] {expected}")
            print(f"         Prediction:  '{pred_label}' | Index: #{decision.choice_index}")
            print(f"         Confidence:  {prob:.4f} (>= 0.85: {'YES' if is_calibrated else 'NO'}) | Logit: {decision.raw_logit:.2f}")
            print(f"         Latency:     {latency:.2f} us")

            if not (is_correct and is_calibrated):
                all_passed = False

        return all_passed

    # 3. Test Domain 1: Mesh & PBR Rules
    print("[3/4] Testing cartridge_mesh_pbr.axb...")
    mesh_ok = evaluate_suite(cart_mesh, MESH_PBR_TESTS, "3D Mesh & PBR Rules")
    print()

    # 4. Test Domain 2: Runtime Performance Triage
    print("[4/4] Testing cartridge_runtime_perf.axb...")
    perf_ok = evaluate_suite(cart_perf, RUNTIME_PERF_TESTS, "Runtime Performance Triage")
    print()

    # Cleanup
    axo.axo_cartridge_unload(ctypes.byref(cart_mesh))
    axo.axo_cartridge_unload(ctypes.byref(cart_perf))
    axo.axo_destroy(ctx_ptr)

    print("===========================================================")
    if mesh_ok and perf_ok:
        print("  ALL PRODUCTION CARTRIDGE TESTS PASSED (16/16, Conf > 0.85)")
    else:
        print("  SOME PRODUCTION CARTRIDGE TESTS FAILED")
    print("===========================================================\n")

    return 0 if (mesh_ok and perf_ok) else 1


if __name__ == "__main__":
    sys.exit(run_tests())
