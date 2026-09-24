# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import os
import json
import argparse
from typing import Optional

from axobrier.export import inspect_cartridge
from axobrier.trainer import train_cartridge
from axobrier.core import AxoEngine


def handle_train(args: argparse.Namespace) -> int:
    try:
        res = train_cartridge(
            train_path=args.data,
            val_path=args.val,
            output_axb=args.output,
            embed_dim=args.embed_dim,
            epochs=args.epochs,
            lr=args.lr,
            verbose=not args.quiet
        )
        print(json.dumps(res, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "command": "train", "error": str(e)}, indent=2), file=sys.stderr)
        return 1


def handle_route(args: argparse.Namespace) -> int:
    try:
        with AxoEngine(device_id=args.device) as engine:
            cart = engine.load_cartridge(args.cartridge)
            decision = engine.route(cart, args.query, embed_dim=args.embed_dim, distill=getattr(args, "distill", False))
            output = {
                "status": "success",
                "cartridge": os.path.basename(args.cartridge),
                "query": args.query,
                **decision
            }
            print(json.dumps(output, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "command": "route", "cartridge": getattr(args, "cartridge", None), "error": str(e)}, indent=2), file=sys.stderr)
        return 1


def handle_inspect(args: argparse.Namespace) -> int:
    try:
        filepath = args.cartridge
        info = inspect_cartridge(filepath)
        output = {
            "status": "success",
            "cartridge": os.path.basename(filepath),
            **info
        }
        print(json.dumps(output, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "command": "inspect", "cartridge": getattr(args, "cartridge", None), "error": str(e)}, indent=2), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="axobrier",
        description="Axobrier: System 1 Decision Engine CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train choice cartridge from JSONL")
    p_train.add_argument("--data", required=True, help="Input training JSONL")
    p_train.add_argument("--val", default=None, help="Validation JSONL")
    p_train.add_argument("--output", "-o", default="cartridge.axb", help="Output .axb path")
    p_train.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    p_train.add_argument("--epochs", type=int, default=250, help="Training epochs")
    p_train.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    p_train.add_argument("--quiet", "-q", action="store_true", help="Suppress progress")

    p_route = subparsers.add_parser("route", help="Evaluate query on GPU")
    p_route.add_argument("--cartridge", "-c", required=True, help="Path to .axb cartridge")
    p_route.add_argument("--query", "-q", required=True, help="Text query string")
    p_route.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    p_route.add_argument("--device", type=int, default=0, help="CUDA device index")
    p_route.add_argument("--distill", action="store_true", help="Distill multi-line diagnostic signal")

    p_inspect = subparsers.add_parser("inspect", help="Inspect .axb metadata")
    p_inspect.add_argument("cartridge", help="Path to .axb cartridge file")

    args = parser.parse_args()

    if args.command == "train":
        return handle_train(args)
    elif args.command == "route":
        return handle_route(args)
    elif args.command == "inspect":
        return handle_inspect(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
