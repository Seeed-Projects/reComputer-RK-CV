# RK3588 PP-YOLOE 部署指南

[English](./README.md) | [中文]

本目录提供面向 RK3588 的标准化 PP-YOLOE 部署：RKNN NPU 推理、浏览器预览、REST API 和 Docker 一键运行。

## 核心特性

- **NPU 加速**：通过 RKNN Toolkit Lite2 在 RK3588 NPU 上运行 PP-YOLOE RKNN 模型。
- **多种输入**：支持图片、MP4 指定时间帧、内置示例视频和摄像头。
- **Web 服务**：提供浏览器预览、MJPEG 视频流、健康检查、阈值配置和异步视频分析。
- **标准输出**：以 JSON 返回类别、置信度、检测框和图像尺寸。
- **两种模型规模**：默认使用 `ppyoloe_s.rknn`，同时提供 `ppyoloe_m.rknn`。

## 目录结构

- `lib/`：RK3588 RKNN 运行时库。
- `model/ppyoloe_s.rknn`：默认 PP-YOLOE RKNN 模型。
- `model/ppyoloe_m.rknn`：可选的较大模型。
- `model/coco_80_labels_list.txt`：COCO 类别名称。
- `py_utils/`：RKNN 执行器和检测工具。
- `video/test.mp4`：默认示例输入。
- `web_detection.py`：推理、后处理、Web 预览和 REST API 服务。
- `requirements.txt` 和 `rknn-toolkit-lite2-packages/`：运行依赖。

## 快速开始

### 运行已发布镜像

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-ppyoloe:latest
```

访问 `http://<开发板IP>:8000`，OpenAPI 文档地址为 `http://<开发板IP>:8000/docs`。

使用摄像头时，增加 `--device /dev/video0:/dev/video0` 并覆盖容器命令：

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-ppyoloe:latest \
  python web_detection.py --platform rk3588 --model_path model/ppyoloe_s.rknn \
  --class_path model/coco_80_labels_list.txt --camera_id 0
```

### 本地构建

```bash
docker build -f docker/rk3588/ppyoloe.dockerfile \
  -t rk3588-ppyoloe:local src/rk3588_ppyoloe
```

## API 接口文档

### 模型推理

**接口：** `POST /api/models/ppyoloe/predict`
**类型：** `multipart/form-data`

- `file`：可选，待检测图片。
- `video`：可选，MP4 视频；接口读取其中一帧。
- `timestamp`：可选，视频时间戳，单位为秒。
- `realtime`：可选，使用摄像头当前帧。
- `conf`：可选，置信度阈值，默认 `0.25`。
- `iou`：可选，分类 NMS IoU 阈值，默认 `0.45`。

输入优先级为图片、视频、摄像头或示例视频当前帧。

```bash
curl -X POST http://127.0.0.1:8000/api/models/ppyoloe/predict \
  -F "file=@model/bus.jpg" -F "conf=0.30" -F "iou=0.45"
```

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {"class": "bus", "confidence": 0.91,
     "box": {"x1": 120, "y1": 80, "x2": 520, "y2": 430}}
  ],
  "image": {"width": 640, "height": 480}
}
```

### 配置与视频接口

- `GET /api/health`：运行平台、模型及就绪状态。
- `GET /api/config`：返回 `obj_thresh` 和 `nms_thresh`。
- `POST /api/config`：使用 JSON（如 `{"obj_thresh": 0.3, "nms_thresh": 0.5}`）更新阈值。
- `GET /api/video_feed`：MJPEG 预览流。
- `POST /api/video/upload` 和 `POST /api/video/analyze`：上传 MP4，并通过 multipart `filename` 字段启动分析。
- `GET /api/video/status`、`GET /api/video/list`、`GET /api/video/download/{filename}`：查询并获取结果。

## 开发者指南

`web_detection.py` 完成 640×640 letterbox 预处理、RKNN 推理、PP-YOLOE DFL 输出解码、分类 NMS、坐标还原、结果绘制和 API 序列化。

如需使用其他已转换的 PP-YOLOE 模型，请将模型放入 `model/` 并通过 `--model_path` 指定；输出布局需与内置 DFL 解码器兼容。
