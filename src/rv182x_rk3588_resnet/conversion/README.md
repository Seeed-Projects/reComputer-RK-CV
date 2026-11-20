# ResNet50-v2 · W8A8 转换 / conversion

本目录供 x86 Ubuntu 开发者复现转换。ARM64 Docker 运行镜像只需要模型、权重与 RKNN3 Lite，不需要 PyTorch 或 x86 Toolkit。

1. 安装用户提供的 RK1820/RK1828 AI SDK 1.1.0 中的 RKNN3 Toolkit x86 wheel，使用 Python 3.10 环境。
2. 本次导出使用 CPU；YOLO 导出环境为 Ultralytics 8.4.69、ONNX 1.18.0、NumPy 2.2.6。导出可在独立虚拟环境中进行。
3. 以下命令从本项目根目录执行。固定 target=rk1820、core_num=1、quantized_dtype=w8a8、do_quantization=True、normal/channel，无手工 W16A16 子图。

```bash
python conversion/make_calibration.py
```

这会生成指向当前目录的绝对校准路径；移动目录后重新生成。普通视觉模型使用官方 COCO 20 张子集；RAFT 使用官方 20 对图像。演示校准和功能检查不等于完整数据集精度评估。

```bash
curl -fL "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/ResNet/resnet50-v2-7.onnx" -o model/resnet50-v2-7.onnx
python conversion/convert_int8.py --family resnet --onnx model/resnet50-v2-7.onnx --dataset conversion/calibration.txt --output model/resnet50-v2-7.rknn
```


转换将生成同名 `.rknn` 与 `.weight`，必须成对使用。`conversion/*.json` 记录本次已验证产物的参数、原始 ONNX SHA256。运行库、工具和固件均匹配 1.1.0，未混用上游仓库的 1.2 runtime。

[官方示例源码](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/resnet)
