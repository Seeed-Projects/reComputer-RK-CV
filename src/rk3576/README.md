# RK3576 YOLO 部署指南

本目录包含针对 RK3576 优化的 YOLOv11 推理代码。

## 核心特性
- **硬件加速**：针对 RK3576 的 2 TOPS NPU 架构进行了优化。
- **最新驱动**：集成支持 RK3576 的第 5 代 NPU 运行时库。
- **灵活输入**：支持摄像头和本地 MP4 视频输入。

## 目录结构
- `lib/`：包含 RK3576 版 `librknnrt.so`。
- `model/`：存放针对 RK3576 转换的 `.rknn` 模型。
- `realtime_detection.py`：主程序。

## 运行命令
```bash
sudo docker run --rm --privileged --net=host --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/bus/usb:/dev/bus/usb \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/litxaohu/rk3588_yolo/rk3576-yolo:latest
    python realtime_detection.py --model_path model/yolo11n.rknn --camera_id 0
```
