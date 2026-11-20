# YOLO26 Pose · W8A8 转换 / conversion

本目录供 x86 Ubuntu 开发者复现转换。ARM64 Docker 运行镜像只需要模型、权重与 RKNN3 Lite，不需要 PyTorch 或 x86 Toolkit。

1. 安装用户提供的 RK1820/RK1828 AI SDK 1.1.0 中的 RKNN3 Toolkit x86 wheel，使用 Python 3.10 环境。
2. 本次导出使用 CPU；YOLO 导出环境为 Ultralytics 8.4.69、ONNX 1.18.0、NumPy 2.2.6。导出可在独立虚拟环境中进行。
3. 以下命令从本项目根目录执行。固定 target=rk1820、core_num=1、quantized_dtype=w8a8、do_quantization=True、normal/channel，无手工 W16A16 子图。

```bash
python conversion/make_calibration.py
```

这会生成指向当前目录的绝对校准路径；移动目录后重新生成。普通视觉模型使用官方 COCO 20 张子集；RAFT 使用官方 20 对图像。演示校准和功能检查不等于完整数据集精度评估。

```bash
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt -o model/yolo26n-pose.pt
python conversion/export_yolo_onnx.py --weights model/yolo26n-pose.pt
python conversion/convert_int8.py --family yolo26-pose --onnx model/yolo26n-pose.onnx --dataset conversion/calibration.txt --output model/yolo26n-pose.rknn
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt -o model/yolo26s-pose.pt
python conversion/export_yolo_onnx.py --weights model/yolo26s-pose.pt
python conversion/convert_int8.py --family yolo26-pose --onnx model/yolo26s-pose.onnx --dataset conversion/calibration.txt --output model/yolo26s-pose.rknn
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt -o model/yolo26m-pose.pt
python conversion/export_yolo_onnx.py --weights model/yolo26m-pose.pt
python conversion/convert_int8.py --family yolo26-pose --onnx model/yolo26m-pose.onnx --dataset conversion/calibration.txt --output model/yolo26m-pose.rknn
```

YOLO26 使用 raw heads；检测、分割与姿态解码在本项目运行端执行。导出固定 CPU / opset 19 / 640×640，避免默认 opset 20 超出 Toolkit 1.1.0 的支持范围。


转换将生成同名 `.rknn` 与 `.weight`，必须成对使用。`conversion/*.json` 记录本次已验证产物的参数、原始 ONNX SHA256。运行库、工具和固件均匹配 1.1.0，未混用上游仓库的 1.2 runtime。

[官方示例源码](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/yolo26_pose)
