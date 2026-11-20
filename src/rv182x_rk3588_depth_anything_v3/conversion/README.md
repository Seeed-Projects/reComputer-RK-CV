# Depth Anything V3 Base · W8A8 转换 / conversion

本目录供 x86 Ubuntu 开发者复现转换。ARM64 Docker 运行镜像只需要模型、权重与 RKNN3 Lite，不需要 PyTorch 或 x86 Toolkit。

1. 安装用户提供的 RK1820/RK1828 AI SDK 1.1.0 中的 RKNN3 Toolkit x86 wheel，使用 Python 3.10 环境。
2. 本次导出使用 CPU；YOLO 导出环境为 Ultralytics 8.4.69、ONNX 1.18.0、NumPy 2.2.6。导出可在独立虚拟环境中进行。
3. 以下命令从本项目根目录执行。固定 target=rk1820、core_num=1、quantized_dtype=w8a8、do_quantization=True、normal/channel，无手工 W16A16 子图。

```bash
python conversion/make_calibration.py
```

这会生成指向当前目录的绝对校准路径；移动目录后重新生成。普通视觉模型使用官方 COCO 20 张子集；RAFT 使用官方 20 对图像。演示校准和功能检查不等于完整数据集精度评估。

从 depth-anything/DA3-BASE 下载 config.json 与 model.safetensors 到 source/DA3-BASE。导出环境使用 depth-anything-3==0.1.1、numpy==1.26.4、onnx==1.18.0，以及 einops、omegaconf、e3nn、safetensors。导出与 Toolkit 转换可使用不同虚拟环境，避免 numpy 版本冲突。

```bash
CUDA_VISIBLE_DEVICES="" python conversion/export_onnx.py --model_path source/DA3-BASE --output_dir conversion/exports --views 1 --image_size 280
python conversion/prepare_depth_calibration.py --models conversion/exports --images conversion/calibration.txt --output conversion/intermediates
python conversion/compile_partition.py --stage da3-local --onnx conversion/exports/da3_base_local.onnx --dataset conversion/calibration.txt --output model/da3_base_local.rknn
python conversion/compile_partition.py --stage da3-global --onnx conversion/exports/da3_base_global.onnx --dataset conversion/intermediates/global.txt --output model/da3_base_global.rknn
python conversion/compile_partition.py --stage da3-head --onnx conversion/exports/da3_base_head.onnx --dataset conversion/intermediates/head.txt --output model/da3_base_head.rknn
```

Global/Head 的校准输入由真实图像的浮点 ONNX 中间结果生成。模型固定单 view、280×280，三段 W8A8。导出器修复单 view 的零长度 camera-token 扩展分支，并直接按 checkpoint config 加载网络，避免引入不使用的 3D 导出依赖。官方 checkpoint 未提供的辅助 ray-head 参数不进入 depth/confidence 输出；其余权重按名称校验。只导出 depth/confidence，不包含相机参数分支。


转换将生成同名 `.rknn` 与 `.weight`，必须成对使用。`conversion/*.json` 记录本次已验证产物的参数、原始 ONNX SHA256。运行库、工具和固件均匹配 1.1.0，未混用上游仓库的 1.2 runtime。

[官方示例源码](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/depth_anything_v3)
