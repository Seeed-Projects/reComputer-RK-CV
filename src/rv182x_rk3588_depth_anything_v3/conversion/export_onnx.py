#!/usr/bin/env python3
"""Export the DA3-BASE Local, Global, and Head ONNX partitions."""

import argparse
import tempfile
from pathlib import Path

import torch

from da3_export import export_global, export_head, export_local
from da3_export.common import (
    PATCH_SIZE,
    load_da3_model,
    select_device,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "depth-anything/DA3-BASE"
DEFAULT_IMAGE_SIZE = 280
DEFAULT_VIEWS = 10
STAGES = ("local", "global", "head")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fixed-shape DA3-BASE ONNX deployment partitions."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="Whether download model from www.modelscope.cn",
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "model")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    return parser.parse_args()


def output_names():
    return {
        "local": "da3_base_local.onnx",
        "global": "da3_base_global.onnx",
        "head": "da3_base_head.onnx",
    }


def validate_args(args):
    if args.views < 1:
        raise SystemExit("--views must be positive")
    if args.image_size < PATCH_SIZE:
        raise SystemExit(f"--image_size must be at least {PATCH_SIZE}")


def main():
    args = parse_args()
    validate_args(args)
    torch.manual_seed(0)

    if args.modelscope:
        from modelscope import snapshot_download

        args.model_path = snapshot_download(args.model_path)

    stages = STAGES if args.stage == "all" else (args.stage,)
    device = select_device()
    print(f"loading model: {args.model_path}")
    print(f"export device: {device}")
    model = load_da3_model(args.model_path, device)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    names = output_names()
    with tempfile.TemporaryDirectory(
        prefix=".da3_export_", dir=args.output_dir.parent
    ) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        exported = {}
        for stage in stages:
            temporary_output = temporary_dir / names[stage]
            print(f"exporting stage: {stage}")
            if stage == "local":
                export_local(model, args.image_size, temporary_output, device)
            elif stage == "global":
                export_global(
                    model, args.views, args.image_size, temporary_output, device
                )
            else:
                export_head(model, args.views, args.image_size, temporary_output, device)
            exported[stage] = temporary_output
            if device.type == "cuda":
                torch.cuda.empty_cache()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        final_paths = []
        for stage in stages:
            final_path = args.output_dir / names[stage]
            exported[stage].replace(final_path)
            final_paths.append(final_path)

    print("export complete:")
    for path in final_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
