# RK3588 YOLO 部署指南

本目录包含针对 RK3588 优化的 YOLOv11 推理代码。

## 核心特性
- **硬件加速**：充分利用 RK3588 的 6 TOPS NPU 算力。
- **多核支持**：默认开启 `NPU_CORE_0_1_2` 全核模式。
- **高性能推理**：基于推理时间计算的实时 FPS 显示。

## 目录结构
- `lib/`：包含 RK3588 版 `librknnrt.so`。
- `model/`：存放针对 RK3588 转换的 `.rknn` 模型。
- `realtime_detection.py`：主程序。

## 运行命令
```bash
sudo docker run --rm --privileged --net=host --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/bus/usb:/dev/bus/usb \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/litxaohu/rk3588_yolo/rk3588-yolo:latest
    python realtime_detection.py --model_path model/yolo11n.rknn --camera_id 0
```
