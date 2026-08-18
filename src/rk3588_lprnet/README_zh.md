# RK3588 LPRNet 部署指南

[English](./README.md) | [中文]

本目录将 RKNN Model Zoo 的 LPRNet 案例封装为适用于 RK3588 的标准 reComputer RK-CV 服务，支持 RKNN NPU 识别、场景内车牌候选定位、浏览器实时预览、摄像头和本地视频输入、Web 上传视频分析、REST API 与 ARM64 Docker 镜像。

## 核心功能

- 使用 RKNN LPRNet 模型识别已裁剪的中国车牌。
- 在场景图像中通过 OpenCV 级联分类器、边缘和颜色特征定位多个候选车牌，再逐个识别。
- 在 MJPEG 画面中框选每个候选车牌并显示编号/可渲染字符，Web 结果面板显示全部 Unicode 车牌内容。
- 支持 `/dev/videoN` 摄像头、通过 `--video` 循环播放本地 MP4、图片上传和 Web 上传 MP4 分析。
- API 返回车牌文本、坐标框、候选分数和识别分数。

## 目录结构

- `model/lprnet.rknn`：为 RK3588 转换的 LPRNet 模型。
- `model/test.jpg`：启动预热使用的车牌裁剪图。
- `video/test.mp4`：场景测试视频。
- `task_runtime.py`：候选定位、RKNN 字符识别和画面标注。
- `web_service.py`：FastAPI、英文 Web 页面、MJPEG、摄像头和视频处理。
- `rknn_runtime.py`：线程安全的 RKNNLite 封装。
- `py_utils/`：官方字符表和 CTC 风格解码逻辑。

## 快速开始

### 1. Web 上传模式

该模式仅启动 Web/API，不打开摄像头：

```bash
sudo docker run --rm --name rk3588-lprnet \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-lprnet:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --camera_id -1 --host 0.0.0.0 --port 8000
```

访问 `http://<开发板IP>:8000` 使用英文 Web 页面上传图片或 MP4；访问 `http://<开发板IP>:8000/docs` 查看 OpenAPI。

### 2. 调用本地摄像头

摄像头 ID `N` 对应 `/dev/videoN`，以下以 `/dev/video0` 为例：

```bash
sudo docker run --rm --name rk3588-lprnet-camera \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-lprnet:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

首页和 `GET /api/video_feed` 会持续显示带车牌框的实时结果。如果设备是 `/dev/video1`，请映射该节点并使用 `--camera_id 1`。

### 3. 分析本地 MP4

```bash
sudo docker run --rm --name rk3588-lprnet-video \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-lprnet:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --video /app/video/test.mp4 --host 0.0.0.0 --port 8000
```

`--video` 与 `--video_path` 等价。本地视频优先于 `--camera_id`，并在结束后循环播放，以保持 Web 实时预览。分析宿主机视频时可添加 `-v /path/input.mp4:/data/input.mp4:ro`，然后传入 `--video /data/input.mp4`。

### 4. 本地构建

在仓库根目录执行：

```bash
docker build -f docker/rk3588/lprnet.dockerfile \
  -t rk3588-lprnet:local src/rk3588_lprnet
```

## 命令行参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--platform` | 是 | — | RKNN 目标平台；本目录使用 `rk3588`。 |
| `--model_dir` | 否 | `model` | 包含 `lprnet.rknn` 和 `test.jpg` 的目录。 |
| `--camera_id` | 否 | `-1` | 摄像头索引 `N` 对应 `/dev/videoN`；`-1` 表示仅启用 Web 上传。 |
| `--video`、`--video_path` | 否 | — | 本地 MP4 路径，优先于摄像头并循环播放。 |
| `--host` | 否 | `0.0.0.0` | FastAPI 监听地址。 |
| `--port` | 否 | `8000` | FastAPI 监听端口。 |

`PYTHONUNBUFFERED=1` 让日志立即输出；`RKNN_LOG_LEVEL=0` 隐藏已确认无害的 RKNN 静态模型初始化提示。`renderD128` 是 RK3588 测试设备上已验证的节点；若设备不同，请检查本机 `/dev/dri/`。

## API

### 图片或场景识别

**接口：** `POST /api/models/lprnet/predict`

**类型：** `multipart/form-data`

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | OpenCV 可解码的图片。curl 本地文件路径前必须添加 `@`。 |
| `whole_image` | 否 | 输入已经是紧密裁剪的车牌时设为 `true`；场景图片默认自动定位候选车牌。 |
| `min_score` | 否 | 单次请求的识别分数过滤值，范围 `0` 至 `1`。该分数适合相对过滤，不是经过校准的概率。 |
| `max_plates` | 否 | 最大候选车牌数，范围 `1` 至 `32`。 |
| `min_text_length` | 否 | 最少识别字符数，范围 `1` 至 `8`；默认 `6`，用于过滤过短的误识别。 |

```bash
curl -X POST http://127.0.0.1:8000/api/models/lprnet/predict \
  -F "file=@model/test.jpg" \
  -F "whole_image=true"
```

响应中的 `result.plates` 包含：

- `text`：完整车牌识别结果。
- `box`：原图坐标 `[x1,y1,x2,y2]`。
- `candidate_score`：OpenCV 候选定位分数。
- `recognition_score`：LPRNet 时间步置信度均值。

### 运行配置

```bash
curl http://127.0.0.1:8000/api/config
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"min_score":0.0,"max_plates":8,"min_text_length":6}'
```

该配置同时应用于摄像头、循环本地视频和 Web 上传视频。

### 视频与服务接口

- `GET /api/health`：模型、平台和当前输入源状态。
- `GET /api/results/latest`：最新画面的全部车牌文本与坐标框。
- `GET /api/video_feed`：带标注的 MJPEG 实时流。
- `POST /api/video/upload`：通过 multipart 字段 `file` 上传 `.mp4`。
- `POST /api/video/analyze`：通过 multipart 字段 `filename` 启动分析。
- `GET /api/video/status`：处理进度、输出文件名和错误。
- `GET /api/video/list`：已上传和已生成的 MP4。
- `GET /api/video/download/{filename}`：下载分析后的 MP4。

```bash
curl -X POST http://127.0.0.1:8000/api/video/upload \
  -F "file=@video/test.mp4"
curl -X POST http://127.0.0.1:8000/api/video/analyze \
  -F "filename=test.mp4"
curl http://127.0.0.1:8000/api/video/status
```

## 模型范围与限制

随项目提供的 RKNN 模型是官方 LPRNet 字符识别模型，训练目标是已裁剪的中国车牌，它本身不是端到端车牌检测器。场景模式的框由轻量 OpenCV 方法生成，再送入 LPRNet；距离过远、模糊、大角度、遮挡或非中国车牌可能漏检或误识别。生产环境建议接入专用车牌检测模型，并将检测到的裁剪区域交给本项目的 LPRNet 运行时。

模型转换输入和脚本保留在 `rknn_model_zoo/examples/LPRNet`。
