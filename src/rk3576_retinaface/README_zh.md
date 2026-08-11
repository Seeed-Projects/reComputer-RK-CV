# RK3576 RetinaFace 部署指南

[English](./README.md) | [中文]

本目录将 RetinaFace 人脸检测封装为适用于 RK3576 的 reComputer RK-CV 标准服务，提供 RKNN NPU 推理、五点人脸关键点、浏览器预览、REST API、异步 MP4 处理和 ARM64 Docker 镜像。

## 核心特性

- 在 RK3576 NPU 上运行 320 × 320、MobileNet 骨干网络的 RetinaFace RKNN 模型。
- 为每个保留的人脸返回置信度、边界框和五个关键点。
- 支持可配置置信度阈值、图片推理、MJPEG 预览和上传 MP4 分析。
- 提供 FastAPI/OpenAPI 接口以及带健康检查的 Docker 运行环境。

## 目录结构

- `model/retinaface_mobile.rknn`：当前运行时实际加载的模型。
- `model/retinaface_resnet50.rknn`：随项目提供的备选模型，不会自动选用。
- `model/test.jpg`：启动预热和初始预览图片。
- `py_utils/retinaface_official.py`：先验框生成、框/关键点解码、等比例缩放和 NMS。
- `task_runtime.py`：RKNN 推理、筛选、关键点渲染和结果序列化。
- `web_service.py`：Web 页面、REST API、预览流和异步 MP4 处理。
- `video/`：示例视频资源。

## 快速开始

### 1. 运行已发布 Docker 镜像

```bash
sudo docker run --rm --name rk3576-retinaface \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-retinaface:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id -1 --host 0.0.0.0 --port 8000
```

访问 `http://<开发板IP>:8000` 上传图片并查看标注预览，或访问 `http://<开发板IP>:8000/docs` 查看 OpenAPI。命令使用 `--net=host`；若 8000 端口被占用，请修改 `--port`。

### 2. 调用本地摄像头

映射采集节点并传入对应数字 ID。下面以 `/dev/video0` 为例：

```bash
sudo docker run --rm --name rk3576-retinaface-camera \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-retinaface:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

如果采集节点为 `/dev/video1`，请映射该节点并使用 `--camera_id 1`。服务会持续处理画面，并在首页和 `GET /api/video_feed` 中显示结果。

### 3. 分析本地视频

```bash
sudo docker run --rm --name rk3576-retinaface-video \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-retinaface:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --video video/test.mp4 --host 0.0.0.0 --port 8000
```

`--video` 与 `--video_path` 等价，均优先于 `--camera_id`，并在视频结束后自动循环，以保持 Web 实时预览。上面的命令直接使用镜像内置的 `video/test.mp4`。

### 4. 仅 Web 上传模式

镜像默认使用 `--camera_id -1`，只启动 Web/API 而不打开摄像头。图片可通过首页或推理 API 提交，MP4 使用上传和分析 API。

### 5. 本地构建

```bash
docker build -f docker/rk3576/retinaface.dockerfile \
  -t rk3576-retinaface:local src/rk3576_retinaface
```

### 启动参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--platform` | 是 | — | 目标 NPU；本目录使用 `rk3576`。 |
| `--model_dir` | 否 | `model` | 包含 RetinaFace RKNN 文件和 `test.jpg` 的目录。 |
| `--camera_id` | 否 | `-1` | 摄像头索引 `N` 对应 `/dev/videoN`；`-1` 表示禁用摄像头采集。 |
| `--video_path`、`--video` | 否 | — | 本地视频路径，优先于 `--camera_id` 并循环播放。 |
| `--host` | 否 | `0.0.0.0` | FastAPI 监听地址。 |
| `--port` | 否 | `8000` | 容器内部服务端口。 |

`PYTHONUNBUFFERED=1` 让服务日志即时输出；`RKNN_LOG_LEVEL=0` 会隐藏已确认无害的静态模型初始化提示。排查 RKNN 初始化问题时可去掉后一个环境变量。`renderD129` 是已在 RK3576 开发板上验证的 NPU 节点；如设备节点不同，请以本机 `/dev/dri/` 为准。

## API 接口文档

### 1. 人脸检测

**接口：** `POST /api/models/retinaface/predict`

**类型：** `multipart/form-data`

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | OpenCV 可解码的图片。 |
| `threshold` | 否 | 单次请求的人脸置信度阈值；默认使用全局值 `0.25`。 |

```bash
curl -X POST http://127.0.0.1:8000/api/models/retinaface/predict \
  -F "file=@model/test.jpg" \
  -F "threshold=0.5"
```

`result.faces` 中每项包含 `confidence`、格式为 `[x1,y1,x2,y2]` 的 `box`，以及五个 `landmarks`；`result.count` 表示保留的人脸数量。标注后的图片会成为 `GET /api/video_feed` 的最新画面。

### 2. 置信度配置

`GET /api/config` 返回全局配置。可通过 JSON 修改预览和视频处理使用的阈值：

```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"threshold":0.5}'
```

`threshold` 范围为 0 到 1。通用字段 `topk` 仅用于标准接口兼容，不参与 RetinaFace 后处理；NMS 固定使用 0.5 的 IoU 阈值。

### 3. 视频分析与服务接口

- `GET /api/health`：平台、就绪状态和项目内模型名称。
- `GET /api/video_feed`：最新人脸框和关键点标注的 MJPEG 流。
- `POST /api/video/upload`：通过 multipart 字段 `file` 上传一个 `.mp4`。
- `POST /api/video/analyze`：通过 multipart 字段 `filename` 处理已上传文件。
- `GET /api/video/status`：处理进度和错误信息。
- `GET /api/video/list`：已上传和已生成的 MP4 文件。
- `GET /api/video/download/{filename}`：下载生成结果。

```bash
curl -X POST http://127.0.0.1:8000/api/video/upload -F "file=@video/test.mp4"
curl -X POST http://127.0.0.1:8000/api/video/analyze -F "filename=test.mp4"
curl http://127.0.0.1:8000/api/video/status
```

## 开发者指南

每帧会等比例填充到 320 × 320，并从 BGR 转为 RGB 后送入 RKNN。运行时解码基于先验框的人脸框和五个关键点，将坐标恢复到原图，执行 0.5 NMS，最后按请求的置信度阈值筛选。

转换脚本和原始 ONNX 模型位于 `rknn_model_zoo/examples/RetinaFace`。当前构造函数明确加载 `retinaface_mobile.rknn`；如需选择 `retinaface_resnet50.rknn`，必须修改 `task_runtime.py` 中的模型路径，并确认其三个输出的布局一致。
