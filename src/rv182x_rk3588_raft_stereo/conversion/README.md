# RAFT-Stereo · W8A8 转换 / conversion

本目录供 x86 Ubuntu 开发者复现转换。ARM64 Docker 运行镜像只需要模型、权重与 RKNN3 Lite，不需要 PyTorch 或 x86 Toolkit。

1. 安装用户提供的 RK1820/RK1828 AI SDK 1.1.0 中的 RKNN3 Toolkit x86 wheel，使用 Python 3.10 环境。
2. 本次导出使用 CPU；YOLO 导出环境为 Ultralytics 8.4.69、ONNX 1.18.0、NumPy 2.2.6。导出可在独立虚拟环境中进行。
3. 以下命令从本项目根目录执行。固定 target=rk1820、core_num=1、quantized_dtype=w8a8、do_quantization=True、normal/channel，无手工 W16A16 子图。

```bash
python conversion/make_calibration.py
```

这会生成指向当前目录的绝对校准路径；移动目录后重新生成。普通视觉模型使用官方 COCO 20 张子集；RAFT 使用官方 20 对图像。演示校准和功能检查不等于完整数据集精度评估。

从官方 RAFT-Stereo 仓库获取 checkpoint，使用本目录 raft/convert2onnx_fp.py（CPU / legacy ONNX export）和 raft-performance.patch。固定输入 160×240、iters=1。eth3d/middlebury/sceneflow 使用 3 层 GRU、n_downsample=2；realtime 使用 2 层 GRU、n_downsample=3、shared_backbone。

```bash
python conversion/raft/convert2onnx_fp.py --raft_source source/RAFT-Stereo --ckpt source/RAFT-Stereo/models/raftstereo-eth3d.pth --out_dir conversion/exports --save_name raftstereo-eth3d-160x240 --height 160 --width 240 --iters 1 --corr_levels 4 --corr_radius 4 --n_gru_layers 3 --n_downsample 2 --context_norm batch --slow_fast_gru
python conversion/convert_int8.py --family raft-stereo --onnx conversion/exports/raftstereo-eth3d-160x240.onnx --dataset conversion/calibration.txt --output model/raftstereo-eth3d-160x240.rknn
```

对其余 checkpoint 使用对应架构参数和输出名称。量化以成对图像校准，W8A8；不使用官方 i8 选项默认的 W8A16。导出脚本移除了无效的 kwargs={iters:32}，实际迭代次数由 --iters 控制。


转换将生成同名 `.rknn` 与 `.weight`，必须成对使用。`conversion/*.json` 记录本次已验证产物的参数、原始 ONNX SHA256。运行库、工具和固件均匹配 1.1.0，未混用上游仓库的 1.2 runtime。

[官方示例源码](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/RAFT-Stereo)
