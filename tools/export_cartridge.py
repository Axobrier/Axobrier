#!/usr/bin/env python3
# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import os
import struct
import json
import argparse
from typing import List, Sequence, Union, Optional

AXO_CARTRIDGE_MAGIC = 0x00425841  # "AXB\0" in little-endian
AXO_CARTRIDGE_VERSION = 1
HEADER_FORMAT = "<IIIIffI"        # 28 bytes total
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def pack_choice_matrix_fp16(
    weights: Sequence,
    embed_dim: int,
    num_choices: int
) -> bytes:
    """
    Packs a 2D matrix of shape [embed_dim, num_choices] into column-major
    FP16 bytes: index (d, c) = d * num_choices + c.
    Works with NumPy arrays, PyTorch tensors, or nested Python lists.
    """
    # Check if numpy array
    if hasattr(weights, "shape") and hasattr(weights, "astype"):
        import numpy as np
        arr = np.asarray(weights, dtype=np.float16)
        if arr.shape != (embed_dim, num_choices):
            raise ValueError(
                f"Weight matrix shape mismatch: got {arr.shape}, expected ({embed_dim}, {num_choices})"
            )
        # Axobrier expects column-major: element (d, c) is at d * num_choices + c
        # In C array indexing: choice_matrix[d * num_choices + c]
        # In contiguous C-order of shape [D, C], index is d * C + c
        return arr.tobytes()

    # Pure Python fallback using struct.pack with '<e' (IEEE 754 half-precision)
    packed = bytearray()
    if isinstance(weights[0], (list, tuple)):
        if len(weights) != embed_dim:
            raise ValueError(f"Outer dimension {len(weights)} != embed_dim {embed_dim}")
        for d in range(embed_dim):
            row = weights[d]
            if len(row) != num_choices:
                raise ValueError(f"Row {d} length {len(row)} != num_choices {num_choices}")
            for c in range(num_choices):
                packed.extend(struct.pack("<e", float(row[c])))
    else:
        # Flat 1D list of size D * C
        if len(weights) != embed_dim * num_choices:
            raise ValueError(
                f"Flat weights length {len(weights)} != {embed_dim * num_choices}"
            )
        for val in weights:
            packed.extend(struct.pack("<e", float(val)))

    return bytes(packed)


def export_cartridge(
    filepath: str,
    weights: Sequence,
    labels: Sequence[str],
    platt_temperature: float = 1.0,
    platt_bias: float = 0.0,
    embed_dim: int = 256,
    num_choices: Optional[int] = None
) -> int:
    """
    Exports a choice head to a binary .axb cartridge file.

    Parameters:
        filepath: Destination path for the .axb file.
        weights: Choice matrix of shape [embed_dim, num_choices].
        labels: Sequence of string labels (length must equal num_choices).
        platt_temperature: Calibration temperature (default 1.0).
        platt_bias: Calibration bias (default 0.0).
        embed_dim: Embedding dimension (default 256).
        num_choices: Number of choices (inferred from labels if None).

    Returns:
        Total bytes written.
    """
    if num_choices is None:
        num_choices = len(labels)

    if len(labels) != num_choices:
        raise ValueError(
            f"Labels count ({len(labels)}) does not match num_choices ({num_choices})"
        )

    # 1. Encode labels as null-delimited UTF-8 strings
    label_bytes = bytearray()
    for lbl in labels:
        label_bytes.extend(lbl.encode("utf-8"))
        label_bytes.append(0)  # Null terminator

    label_section_bytes = len(label_bytes)

    # 2. Pack choice matrix
    matrix_bytes = pack_choice_matrix_fp16(weights, embed_dim, num_choices)

    # 3. Pack header (28 bytes)
    header_bytes = struct.pack(
        HEADER_FORMAT,
        AXO_CARTRIDGE_MAGIC,
        AXO_CARTRIDGE_VERSION,
        embed_dim,
        num_choices,
        float(platt_temperature),
        float(platt_bias),
        label_section_bytes
    )

    # 4. Write binary file
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header_bytes)
        f.write(matrix_bytes)
        f.write(label_bytes)

    total_bytes = len(header_bytes) + len(matrix_bytes) + len(label_bytes)
    return total_bytes


def inspect_cartridge(filepath: str) -> dict:
    """
    Parses and prints the metadata and labels of an .axb cartridge.
    """
    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small for .axb header: {len(data)} bytes")

    magic, version, embed_dim, num_choices, temp, bias, label_bytes_len = struct.unpack(
        HEADER_FORMAT, data[:HEADER_SIZE]
    )

    if magic != AXO_CARTRIDGE_MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:08X} (expected 0x{AXO_CARTRIDGE_MAGIC:08X})")

    matrix_bytes_len = embed_dim * num_choices * 2
    expected_total = HEADER_SIZE + matrix_bytes_len + label_bytes_len
    if len(data) != expected_total:
        raise ValueError(f"Size mismatch: {len(data)} != expected {expected_total}")

    labels_data = data[HEADER_SIZE + matrix_bytes_len :]
    raw_labels = labels_data.split(b"\0")
    if raw_labels and raw_labels[-1] == b"":
        raw_labels.pop()  # remove trailing empty split

    labels = [l.decode("utf-8", errors="replace") for l in raw_labels]

    return {
        "magic": hex(magic),
        "version": version,
        "embed_dim": embed_dim,
        "num_choices": num_choices,
        "platt_temperature": temp,
        "platt_bias": bias,
        "label_section_bytes": label_bytes_len,
        "labels": labels,
        "total_size_bytes": len(data)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Axobrier .axb Choice Head Cartridge Exporter"
    )
    parser.add_argument("--output", "-o", type=str, help="Output .axb cartridge file")
    parser.add_argument("--inspect", "-i", type=str, help="Inspect an existing .axb cartridge")
    parser.add_argument("--demo", action="store_true", help="Generate sample mock cartridges for testing")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension (default: 256)")
    parser.add_argument("--temp", type=float, default=1.0, help="Platt temperature parameter (default: 1.0)")
    parser.add_argument("--bias", type=float, default=0.0, help="Platt bias parameter (default: 0.0)")
    args = parser.parse_args()

    if args.inspect:
        info = inspect_cartridge(args.inspect)
        print(f"Cartridge: {args.inspect}")
        print(f"  Version:     {info['version']}")
        print(f"  Embed Dim:   {info['embed_dim']}")
        print(f"  Num Choices: {info['num_choices']}")
        print(f"  Platt Temp:  {info['platt_temperature']:.4f}")
        print(f"  Platt Bias:  {info['platt_bias']:.4f}")
        print(f"  File Size:   {info['total_size_bytes']} bytes")
        print(f"  Labels ({len(info['labels'])}):")
        for idx, lbl in enumerate(info["labels"][:10]):
            print(f"    [{idx}] {lbl}")
        if len(info["labels"]) > 10:
            print(f"    ... and {len(info['labels']) - 10} more")
        return

    if args.demo:
        # Generate two sample cartridges
        dim = args.embed_dim

        # Cartridge 1: DevOps / Autonomous Agent
        devops_labels = [
            "IDLE_WAIT", "QUERY_DATABASE", "RUN_REGRESSION_TESTS",
            "BUILD_CONTAINER", "DEPLOY_CANARY", "PROMOTE_RELEASE",
            "ROLLBACK_DEPLOYMENT", "SCALE_UP_REPLICAS", "SCALE_DOWN_REPLICAS",
            "RESTART_SERVICE", "TRIGGER_FAILOVER", "FLUSH_CACHE",
            "ROTATE_CREDENTIALS", "EMIT_ALERT_PAGERDUTY", "TAKE_SNAPSHOT",
            "PURGE_DEAD_LETTER_QUEUE"
        ]
        C1 = len(devops_labels)
        # Create pseudo-identity weights: choice c responds to dim (c % dim)
        weights1 = [[1.0 if d == (c % dim) else 0.0 for c in range(C1)] for d in range(dim)]
        out1 = args.output or "cartridge_devops.axb"
        b1 = export_cartridge(out1, weights1, devops_labels, 1.0, 0.0, dim)
        print(f"[OK] Created demo cartridge: {out1} ({b1} bytes, {C1} choices)")

        # Cartridge 2: 3D Scene Modeling & Blender Commands
        blender_labels = [
            "NAVIGATE_VIEWPORT", "SELECT_OBJECT", "DESELECT_ALL",
            "ENTER_EDIT_MODE", "ENTER_OBJECT_MODE", "EXTRUDE_FACES",
            "SUBDIVIDE_SURFACE", "BEVEL_EDGES", "UNWRAP_UV_SMART",
            "BAKE_DIFFUSE_TEXTURE", "ASSIGN_PBR_MATERIAL", "ALIGN_CAMERA_TO_VIEW",
            "SET_LIGHT_ENERGY", "APPLY_ALL_TRANSFORMS", "EXPORT_GLTF_BINARY",
            "CLEAN_NON_MANIFOLD_GEOMETRY", "RIG_TO_ARMATURE", "VALIDATE_LODS"
        ]
        C2 = len(blender_labels)
        weights2 = [[1.0 if d == (c % dim) else 0.0 for c in range(C2)] for d in range(dim)]
        out2 = "cartridge_blender.axb"
        b2 = export_cartridge(out2, weights2, blender_labels, 1.0, 0.0, dim)
        print(f"[OK] Created demo cartridge: {out2} ({b2} bytes, {C2} choices)")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
