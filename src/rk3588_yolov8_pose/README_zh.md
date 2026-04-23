# RK3588 YOLOv8-Pose 实时检测与 Web 预览项目

该项目基于 RKNN-Toolkit-Lite2，在瑞芯微 RK3588 平台上实现 YOLOv8-Pose 人体姿态估计模型的高性能部署。项目采用纯 Python 架构，集成了 FastAPI 提供 Web API 和 MJPEG 视频流预览，同时支持本地 GUI 实时显示。

## 目录结构

- `model/`: 存放 YOLOv8-Pose 的 RKNN 模型文件 (`yolov8n-pose.rknn`)
- `video/`: 存放用于测试的视频文件 (`test.mp4`)
- `lib/`: 存放 NPU 运行时的依赖库 (`librknnrt.so`)
- `rknn-toolkit-lite2-packages/`: 存放 RKNN-Toolkit-Lite2 的 Python 安装包
- `py_utils/`: 包含推理引擎封装和 Pose 模型相关的后处理/画图工具
- `realtime_detection.py`: 本地 GUI 实时检测与 FastAPI Web API 服务
- `web_detection.py`: 支持多线程推理的 Web API 服务与视频流推流
- `requirements.txt`: Python 依赖列表

## 环境准备

### 1. 硬件要求
- 瑞芯微 RK3588/RK3576 开发板 (例如: reComputer RK-CV 系列)
- 支持 USB 摄像头 (用于实时检测)

### 2. 系统要求
- 使用 armbian

### 3. 安装依赖

```bash
# 1. 更新系统并安装系统依赖
sudo apt update
sudo apt install -y python3-pip python3-dev libgl1-mesa-glx libglib2.0-0

# 2. 安装 Python 基础依赖
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 安装 RKNN-Toolkit-Lite2 (根据 Python 版本选择)
# 以 Python 3.9 为例：
pip3 install rknn-toolkit-lite2-packages/rknn_toolkit_lite2-2.0.0b0-cp39-cp39-linux_aarch64.whl
```

## 运行指南

本项目提供两种运行模式：本地实时检测模式 (附带后台 Web API) 和 高性能 Web 服务模式。

> **注意：** 在运行之前，请确保已将量化好的 `yolov8n-pose.rknn` 放入 `model/` 目录下，并将测试视频 `test.mp4` 放入 `video/` 目录下（用户自行补充）。


### Web 服务 (web_detection.py)

该模式采用多线程推理，专为 Web 端视频流和 API 设计，支持性能分析，不显示本地窗口。

```bash
# 启动服务，默认处理 video/test.mp4
python3 web_detection.py --model_path model/yolov8n-pose.rknn --video video/test.mp4

# 处理摄像头画面
python3 web_detection.py --model_path model/yolov8n-pose.rknn --video 0

# 开启性能分析 (Profile)
python3 web_detection.py --model_path model/yolov8n-pose.rknn --video video/test.mp4 --profile
```

## Web 访问与 API 说明

服务启动后，默认运行在 `0.0.0.0:8000`。

### 1. Web 实时预览
在浏览器中访问：`http://<开发板IP>:8000`
页面提供实时视频流查看，并支持动态调整置信度 (Confidence) 和 NMS 阈值。

### 2. RESTful API 接口

- **获取当前配置**: `GET /api/config`
- **更新配置**: `POST /api/config`
- **获取视频流**: `GET /api/video_feed`
- **推理预测**: `POST /api/models/yolo11/predict` (兼容原有 YOLO11 的路径设计)
  - 支持上传图片 (`file`)
  - 支持上传视频并指定时间戳 (`video`, `timestamp`)
  - 支持使用当前摄像头画面 (`realtime=true`)
  - 返回结果包含类别、置信度、边界框 (`box`) 和 17 个关键点坐标 (`keypoints`)

```json
    {
      "class": "person",
      "confidence": 0.92,
      "box": { "x1": 100, "y1": 200, "x2": 300, "y2": 500 },
      "keypoints": [
        { "x": 150.5, "y": 210.2, "conf": 0.88 },
        { "x": 160.1, "y": 205.5, "conf": 0.91 },
        { "x": 140.2, "y": 206.1, "conf": 0.85 },
        { "x": 170.8, "y": 215.3, "conf": 0.89 },
        { "x": 130.4, "y": 218.7, "conf": 0.82 },
        { "x": 180.2, "y": 250.4, "conf": 0.95 },
        { "x": 120.5, "y": 255.1, "conf": 0.93 },
        { "x": 190.6, "y": 300.2, "conf": 0.87 },
        { "x": 110.3, "y": 305.8, "conf": 0.86 },
        { "x": 200.1, "y": 350.5, "conf": 0.81 },
        { "x": 100.9, "y": 355.2, "conf": 0.80 },
        { "x": 175.4, "y": 400.1, "conf": 0.90 },
        { "x": 125.6, "y": 405.3, "conf": 0.88 },
        { "x": 185.2, "y": 450.6, "conf": 0.85 },
        { "x": 115.8, "y": 455.4, "conf": 0.84 },
        { "x": 195.7, "y": 500.2, "conf": 0.79 },
        { "x": 105.3, "y": 505.7, "conf": 0.78 }
      ]
    }
```

## 性能说明

- 项目已默认配置使用 `RKNNLite.NPU_CORE_0_1_2`，充分利用 RK3588 的 3 个 NPU 核心 (6 TOPS 算力)。
- YOLOv8-Pose 的后处理计算量较大（包含关键点解码），项目中采用 Numpy 进行向量化加速。
