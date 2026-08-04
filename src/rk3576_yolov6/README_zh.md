# RK3576 YOLOv6 部署指南

[English](./README.md) | [中文]

本目录包含针对 RK3576 优化的 YOLOv6 目标检测服务，提供 RKNN 推理、浏览器预览、REST API、离线视频处理，以及可直接运行的 Docker 镜像。

## 核心特性

- **硬件加速**：通过 RKNN-Toolkit-Lite2 在 RK3576 NPU 上运行 YOLOv6 RKNN 模型。
- **COCO 目标检测**：使用 `640 x 640` letterbox 预处理，返回类别、置信度和检测框坐标。
- **多模型支持**：内置 YOLOv6n、YOLOv6s 和 YOLOv6m，默认使用 YOLOv6n。
- **灵活输入**：支持上传图片、提取 MP4 指定时间帧、本地 MP4 循环播放和摄像头输入。
- **Web 与 API**：提供浏览器预览、MJPEG 视频流、健康检查和 REST 推理接口。

## 目录结构

- `lib/`：RK3576 版 `librknnrt.so` 运行时库。
- `model/`：YOLOv6 RKNN 模型、COCO 标签和示例图片。
- `py_utils/`：letterbox、坐标还原、结果绘制和 RKNN 辅助工具。
- `video/`：容器默认命令使用的示例 MP4 视频。
- `web_detection.py`：提供目标检测、Web 预览和 FastAPI 接口的主程序。
- `requirements.txt`：Python 运行依赖。

可用模型包括 `model/yolov6n.rknn`、`model/yolov6s.rknn` 和 `model/yolov6m.rknn`，默认模型为 `model/yolov6n.rknn`。

## 快速开始

### 1. 使用 Docker 运行

在 RK3576 开发板上直接运行已发布镜像：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolov6:latest
```

默认命令会循环分析镜像内置的 `video/test.mp4`。通过 `http://<开发板IP>:8000` 访问 Web 预览，通过 `http://<开发板IP>:8000/docs` 查看交互式 API 文档。

切换为 YOLOv6s 模型：

```bash
sudo docker run --rm --privileged --net=host \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolov6:latest \
    python3 web_detection.py --platform rk3576 \
    --model_path model/yolov6s.rknn --video_path video/test.mp4
```

如果只使用文件上传 Web 模式，可将 `--video_path video/test.mp4` 替换为 `--camera_id -1`。

### 2. 本地构建镜像

在仓库根目录执行：

```bash
docker build -f docker/rk3576/yolov6.dockerfile \
    -t rk3576-yolov6:local src/rk3576_yolov6
```

## API 接口文档

### 1. 模型推理接口（Predict）

**Endpoint：** `POST /api/models/yolov6/predict`

#### 请求参数（`multipart/form-data`）

- `file`：可选，待检测的图片文件。
- `video`：可选，待提取画面的 MP4 文件。
- `timestamp`：可选，提取视频帧的时间戳，单位为秒；默认读取第一帧。
- `realtime`：可选，布尔值。未上传文件时使用摄像头或本地视频的当前帧。
- `conf`：可选，本次请求的置信度阈值，默认值为 `0.25`。
- `iou`：可选，本次请求的 NMS IoU 阈值，默认值为 `0.45`。

#### 调用示例

```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolov6/predict" \
    -F "file=@model/bus.jpg" -F "conf=0.25" -F "iou=0.45"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolov6/predict" \
    -F "video=@video/test.mp4" -F "timestamp=5.5"
```

#### 响应格式

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "bus ",
      "confidence": 0.91,
      "box": {"x1": 100, "y1": 120, "x2": 520, "y2": 430}
    }
  ],
  "image": {"width": 640, "height": 480}
}
```

### 2. 系统配置接口（Config）

- `GET /api/config`：返回 `{"obj_thresh": 0.25, "nms_thresh": 0.45}`。
- `POST /api/config`：接受 `{"obj_thresh": 0.3, "nms_thresh": 0.5}` 格式的 JSON。

全局阈值用于预览、离线分析，以及未单独提供 `conf` 或 `iou` 的推理请求。

### 3. 视频与服务接口

- `GET /api/video_feed`：返回叠加检测结果的 MJPEG 预览流。
- `POST /api/video/upload`：通过 `file` 表单字段上传 MP4 文件。
- `POST /api/video/analyze`：通过已上传的 `filename` 表单字段启动异步处理。
- `GET /api/video/status`：返回处理进度和错误状态。
- `GET /api/video/list`：列出已上传视频和生成结果。
- `GET /api/video/download/{filename}`：下载生成的 MP4 结果。
- `GET /api/health`：返回平台、模型文件和模型就绪状态。

## 开发者指南

### 代码说明

- `web_detection.py` 负责初始化 RK3576 NPU 运行时、循环处理摄像头或视频输入、提供 FastAPI 接口，并实现 YOLOv6 专用解码和 NMS。
- 输入画面经过 `640 x 640` letterbox 和 BGR 到 RGB 转换，检测框随后映射回原始画面尺寸。
- 运行时上传文件和输出结果默认写入 `workspace/`，可通过 `RK_CV_WORKSPACE` 修改目录。

### 替换模型

1. 将兼容 RK3576 的 YOLOv6 `.rknn` 模型放入 `model/`。
2. 确认模型输入尺寸和输出张量与 `web_detection.py` 中的 YOLOv6 预处理及后处理一致。
3. 如果模型不使用默认 COCO 80 类，请同步更新类别配置。
4. 使用 `--model_path model/<模型文件名>.rknn` 指定新模型。
