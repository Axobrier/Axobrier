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
import threading
from typing import List, Dict, Optional, Union, Any

import numpy as np

from axobrier.trainer import text_to_embedding, extract_diagnostic_signal

AXO_HASH_SEED_PRIME = 0x696c6f63636f7262
AXO_ARENA_GUARD_TAG = 0x636f7262

# ─── C ABI Structures ───────────────────────────────────────────────

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

    def evaluate(
        self,
        query: Union[str, np.ndarray],
        embed_dim: int = 256,
        distill: bool = True
    ) -> Dict[str, Any]:
        """Evaluate text query or pre-pooled embedding using associated engine."""
        if not hasattr(self, "_engine") or self._engine is None:
            raise RuntimeError("Cartridge is not bound to an active AxoEngine.")
        return self._engine.route(self, query, embed_dim=embed_dim, distill=distill)

def find_library() -> str:
    """Locate the compiled axobrier_core shared library."""
    env_path = os.environ.get("AXOBRIER_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)

    candidates = [
        os.path.join(repo_root, "build", "Release", "axobrier_core.dll"),
        os.path.join(repo_root, "build", "axobrier_core.dll"),
        os.path.join(repo_root, "build", "libaxobrier_core.so"),
        os.path.join(pkg_dir, "axobrier_core.dll"),
        os.path.join(pkg_dir, "libaxobrier_core.so"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("axobrier_core library not found.")


class AxoEngine:
    """Runtime engine handle managing resident GPU arena and cartridge dispatch."""

    def __init__(self, lib_path: Optional[str] = None, device_id: int = 0):
        self.lib_path = lib_path or find_library()
        self.lib = ctypes.CDLL(self.lib_path)
        self._bind_functions()

        self.cfg = self.lib.axo_config_default()
        self.cfg.device_id = device_id

        self.ctx_ptr = ctypes.c_void_p()
        err = self.lib.axo_create(ctypes.byref(self.ctx_ptr), ctypes.byref(self.cfg))
        if err != 0:
            raise RuntimeError(f"axo_create failed with code {err}")

        self._active_cartridges: List[AxoCartridge] = []
        self._lock = threading.Lock()

    def _bind_functions(self):
        self.lib.axo_config_default.restype = AxoConfig
        self.lib.axo_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(AxoConfig)]
        self.lib.axo_create.restype = ctypes.c_int
        self.lib.axo_destroy.argtypes = [ctypes.c_void_p]

        self.lib.axo_cartridge_load_mmap.argtypes = [ctypes.c_char_p, ctypes.POINTER(AxoCartridge)]
        self.lib.axo_cartridge_load_mmap.restype = ctypes.c_int
        self.lib.axo_cartridge_unload.argtypes = [ctypes.POINTER(AxoCartridge)]
        self.lib.axo_cartridge_unload.restype = ctypes.c_int
        self.lib.axo_cartridge_get_label.argtypes = [ctypes.POINTER(AxoCartridge), ctypes.c_uint32]
        self.lib.axo_cartridge_get_label.restype = ctypes.c_char_p

        self.lib.axo_evaluate_cartridge_embeddings.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(AxoCartridge),
            ctypes.POINTER(AxoEvalParams),
            ctypes.POINTER(AxoDecision)
        ]
        self.lib.axo_evaluate_cartridge_embeddings.restype = ctypes.c_int

        self.lib.axo_device_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.axo_device_alloc.restype = ctypes.c_int
        self.lib.axo_device_free.argtypes = [ctypes.c_void_p]
        self.lib.axo_device_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.axo_device_upload.restype = ctypes.c_int

        if hasattr(self.lib, "axo_hash_token"):
            self.lib.axo_hash_token.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            self.lib.axo_hash_token.restype = ctypes.c_uint64

        if hasattr(self.lib, "axo_validate_arena_boundary"):
            self.lib.axo_validate_arena_boundary.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            self.lib.axo_validate_arena_boundary.restype = ctypes.c_int

    def load_cartridge(self, path: str) -> AxoCartridge:
        """Memory-map an .axb cartridge into resident device memory."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cartridge file not found: {path}")

        cart = AxoCartridge()
        cart._engine = self
        with self._lock:
            err = self.lib.axo_cartridge_load_mmap(path.encode("utf-8"), ctypes.byref(cart))
            if err != 0:
                raise RuntimeError(f"Failed to load cartridge '{path}' (code {err})")
            self._active_cartridges.append(cart)
        return cart

    def unload_cartridge(self, cart: AxoCartridge) -> None:
        """Release cartridge device memory and file mappings."""
        with self._lock:
            self.lib.axo_cartridge_unload(ctypes.byref(cart))
            if cart in self._active_cartridges:
                self._active_cartridges.remove(cart)

    def route(
        self,
        cart: AxoCartridge,
        query: Union[str, np.ndarray],
        embed_dim: int = 256,
        distill: bool = True
    ) -> Dict[str, Any]:
        """Evaluate text or pre-pooled embedding vector against active cartridge."""
        if isinstance(query, str):
            if distill:
                query = extract_diagnostic_signal(query)
            emb_tensor = text_to_embedding(query, embed_dim=embed_dim, distill=distill)
            emb_fp16 = emb_tensor.numpy().astype(np.float16)
        elif isinstance(query, np.ndarray):
            emb_fp16 = query.astype(np.float16)
        else:
            raise TypeError("query must be str or np.ndarray")

        with self._lock:
            d_emb = ctypes.c_void_p()
            nbytes = emb_fp16.nbytes
            self.lib.axo_device_alloc(ctypes.byref(d_emb), nbytes)
            self.lib.axo_device_upload(d_emb, emb_fp16.ctypes.data_as(ctypes.c_void_p), nbytes)

            params = AxoEvalParams()
            params.embeddings = d_emb
            params.choice_matrix = None
            params.batch_size = 1
            params.num_choices = cart.header.num_choices

            decision = AxoDecision()
            status = self.lib.axo_evaluate_cartridge_embeddings(
                self.ctx_ptr, ctypes.byref(cart), ctypes.byref(params), ctypes.byref(decision)
            )
            self.lib.axo_device_free(d_emb)

            if status != 0:
                raise RuntimeError(f"axo_evaluate_cartridge_embeddings failed: code {status}")

            label_bytes = self.lib.axo_cartridge_get_label(ctypes.byref(cart), decision.choice_index)
            label_str = label_bytes.decode("utf-8") if label_bytes else f"CHOICE_{decision.choice_index}"

            return {
                "choice_index": int(decision.choice_index),
                "label": label_str,
                "confidence": float(round(decision.calibrated_prob, 4)),
                "raw_logit": float(round(decision.raw_logit, 2)),
                "entropy": float(round(decision.entropy, 4)),
                "latency_us": float(round(decision.latency_us, 2))
            }

    def close(self):
        with self._lock:
            for cart in list(self._active_cartridges):
                self.lib.axo_cartridge_unload(ctypes.byref(cart))
            self._active_cartridges.clear()
            if self.ctx_ptr:
                self.lib.axo_destroy(self.ctx_ptr)
                self.ctx_ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
