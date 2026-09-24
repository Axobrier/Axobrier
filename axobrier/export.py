# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import struct
from typing import List, Sequence, Union, Optional, Dict, Any

AXO_CARTRIDGE_MAGIC = 0x00425841
AXO_CARTRIDGE_VERSION = 1
HEADER_FORMAT = "<IIIIffI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def pack_choice_matrix_fp16(
    weights: Sequence,
    embed_dim: int,
    num_choices: int
) -> bytes:
    """Pack 2D matrix of shape [embed_dim, num_choices] into column-major FP16 bytes."""
    if hasattr(weights, "detach") and hasattr(weights, "cpu"):
        weights = weights.detach().cpu().numpy()

    if hasattr(weights, "shape") and hasattr(weights, "astype"):
        import numpy as np
        arr = np.asarray(weights, dtype=np.float16)
        if arr.shape != (embed_dim, num_choices):
            raise ValueError(
                f"Weight matrix shape mismatch: got {arr.shape}, expected ({embed_dim}, {num_choices})"
            )
        return arr.tobytes()

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
    """Serialize choice head weights, Platt parameters, and labels into an .axb binary."""
    if num_choices is None:
        num_choices = len(labels)

    if len(labels) != num_choices:
        raise ValueError(
            f"Labels count ({len(labels)}) does not match num_choices ({num_choices})"
        )

    label_bytes = bytearray()
    for lbl in labels:
        label_bytes.extend(lbl.encode("utf-8"))
        label_bytes.append(0)

    label_section_bytes = len(label_bytes)
    matrix_bytes = pack_choice_matrix_fp16(weights, embed_dim, num_choices)

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

    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(header_bytes)
        f.write(matrix_bytes)
        f.write(label_bytes)

    return len(header_bytes) + len(matrix_bytes) + len(label_bytes)


def inspect_cartridge(filepath: str) -> Dict[str, Any]:
    """Parse and return metadata, calibration constants, and labels from an .axb file."""
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
        raw_labels.pop()

    labels = [l.decode("utf-8", errors="replace") for l in raw_labels]

    return {
        "magic": f"0x{magic:08X}",
        "version": version,
        "embed_dim": embed_dim,
        "num_choices": num_choices,
        "platt_temperature": round(temp, 4),
        "platt_bias": round(bias, 4),
        "label_section_bytes": label_bytes_len,
        "labels": labels,
        "total_size_bytes": len(data)
    }
