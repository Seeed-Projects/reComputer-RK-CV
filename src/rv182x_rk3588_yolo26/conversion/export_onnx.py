#!/usr/bin/env python3
"""
Export YOLO26 model to ONNX format.

Usage:
    python export_onnx.py --model_path ../model/yolo26n.pt --output ../model/yolo26n.onnx --img_size 640

Environment:
    See ../docs/env_setup.md for required packages and versions.
"""
import argparse
import os
import sys
import onnx
from onnxsim import simplify

def parse_args():
    parser = argparse.ArgumentParser(description='Export YOLO26 to ONNX')
    parser.add_argument('--model_path', type=str, default='../model/yolo26n.pt',
                        help='Path to pretrained weights (default: ../model/yolo26n.pt)')
    parser.add_argument('--output', type=str, default='../model/yolo26n.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Run onnxsim (default: True)')
    return parser.parse_args()

def export(args):
    # Import ultralytics after argparse to allow --help without ultralytics installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Please run: pip install ultralytics")
        sys.exit(1)

    print(f"[INFO] Loading Ultralytics model: {args.model_path}")
    model = YOLO(args.model_path)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f"[INFO] Exporting ONNX to: {args.output}")
    print(f"[INFO] Input size: {args.img_size}x{args.img_size}")

    # Export to ONNX. YOLO26 is exported as an end-to-end ONNX graph here;
    # convert.py later selects the detection head outputs used by RKNN.

    # Export the model
    export_path = model.export(
        format  =   'onnx',
        imgsz   =   args.img_size,
        device  =   'cpu',
        opset   =   19,
        dynamic =   False,
        end2end =   True,
    )



    # The exported file will be named based on the model.
    # For yolo26n.pt, it will be yolo26n.onnx.
    default_onnx = args.model_path.replace('.pt', '.onnx')
    if os.path.exists(default_onnx):
        import shutil
        shutil.move(default_onnx, args.output)
        print(f"[INFO] Moved {default_onnx} to {args.output}")

    # Check the ONNX model
    print("[INFO] Checking ONNX model...")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model is valid")

    # Print model inputs/outputs
    print("\n[INFO] Model inputs:")
    for inp in onnx_model.graph.input:
        print(f"  - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

    print("\n[INFO] Model outputs:")
    for out in onnx_model.graph.output:
        print(f"  - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

    # Simplify
    if args.simplify:
        print("\n[INFO] Simplifying ONNX model...")
        onnx_model, check = simplify(onnx_model)
        assert check, "Simplified ONNX model failed validation"
        onnx.save(onnx_model, args.output)
        print("[OK] ONNX model simplified")

    print(f"\n[DONE] Exported ONNX model to: {args.output}")



if __name__ == '__main__':
    args = parse_args()
    export(args)
