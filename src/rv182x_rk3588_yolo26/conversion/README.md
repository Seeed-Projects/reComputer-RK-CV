# Reproduce the bundled model

Run conversion on an x86_64 development host using Python 3.10 and the official RKNN3 Toolkit 1.1.0. The ARM64 application image contains only Lite inference and does not perform conversion.

Install the SDK toolkit wheel in a separate environment with NumPy 2.2.6, ONNX 1.18.0, onnxsim 0.6.5, Ultralytics 8.4.69 and a compatible PyTorch. See NOTICE.md for source provenance. The scripts explicitly use CPU export and ONNX opset 19 to match Toolkit 1.1.0.

From this conversion/ directory:

```bash
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt -o ../model/yolo26n.pt
python export_onnx.py --model_path ../model/yolo26n.pt --output ../model/yolo26n.onnx --img_size 640 --simplify
python convert.py --onnx ../model/yolo26n.onnx --platform rk1820 --dtype w8a8 --output ../model/yolo26n.rknn --inference
```

The bundled model/images.txt lists the 20 upstream COCO subset images relative to this directory. Model and weight must be deployed together. Recompute SHA256SUMS after intentionally replacing binary assets.
