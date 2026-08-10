# RK3588 DeepLabV3 部署指南

[English](./README.md) | [中文]

本项目将 DeepLabV3 语义分割封装为适用于 RK3588 的标准案例，包含 RKNN NPU 推理、摄像头/本地视频实时预览、浏览器上传 MP4 分析、REST API 和 Docker 部署。

## 重要设备映射说明

在支持的 reComputer 开发板上，`/dev/dri/renderD129` 是 RKNPU 设备，`renderD128` 是 GPU。RKNN Toolkit Lite2 还会检查开发板兼容信息，因此两个路径都应使用 `-v` 挂载，并建议使用 `--privileged` 通过运行时容器检查。

## 快速开始

### Web 上传模式

镜像默认使用 `--camera_id -1`：只启动 Web/API，不打开摄像头。在 **Video upload** 页签上传 MP4，并在 **Live preview** 页签实时查看处理结果。

```bash
sudo docker run --rm --name rk3588-deeplabv3 \
  --privileged \
  -p 8000:8000 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-deeplabv3:latest \
  python3 web_service.py \
    --platform rk3588 \
    --model_path /app/model/deeplabv3.rknn \
    --sample_path /app/model/test.jpg \
    --overlay_alpha 0.5 \
    --camera_id -1 \
    --host 0.0.0.0 \
    --port 8000
```

访问 `http://<开发板IP>:8000`，OpenAPI 地址为 `http://<开发板IP>:8000/docs`。

如果主机 8000 端口已被占用，只修改 `-p` 左侧端口：

```bash
-p 8001:8000
```

服务仍应监听容器内的 8000 端口，以保证健康检查正确。

### USB 摄像头实时分析

映射摄像头节点并传入对应数字 ID。下面以 `/dev/video0` 为例：

```bash
sudo docker run --rm --name rk3588-deeplabv3-camera \
  --privileged \
  -p 8000:8000 \
  --device /dev/video0:/dev/video0 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-deeplabv3:latest \
  python3 web_service.py --platform rk3588 \
    --model_path /app/model/deeplabv3.rknn --camera_id 0
```

如果采集节点是 `/dev/video1`，则映射该节点并使用 `--camera_id 1`。通过 `http://<开发板IP>:8000` 或 `/api/video_feed` 查看实时分割画面。

### 本地视频实时分析

将宿主机视频挂载到容器并通过 `--video` 传入。该模式优先于 `--camera_id`，视频结束后会自动循环，从而保持 Web 实时预览。

```bash
sudo docker run --rm --name rk3588-deeplabv3-video \
  --privileged \
  -p 8000:8000 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  -v /本机绝对路径/input.mp4:/data/input.mp4:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-deeplabv3:latest \
  python3 web_service.py --platform rk3588 \
    --model_path /app/model/deeplabv3.rknn --video /data/input.mp4
```

### 启动参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--platform` | 是 | — | 目标 NPU；本镜像应使用 `rk3588`。 |
| `--model_path` | 否 | `model/deeplabv3.rknn` | DeepLabV3 RKNN 模型路径。 |
| `--sample_path` | 否 | `model/test.jpg` | 启动预热和 Web 初始预览使用的图片。 |
| `--overlay_alpha` | 否 | `0.5` | 彩色掩码初始透明度，范围 `0`–`1`。 |
| `--camera_id` | 否 | `-1` | 设置为 `N >= 0` 时打开 `/dev/videoN`；`-1` 表示仅 Web 上传模式。 |
| `--video` | 否 | — | 本地视频路径，优先于 `--camera_id` 并循环播放；兼容旧参数名 `--video_path`。 |
| `--host` | 否 | `0.0.0.0` | FastAPI 监听地址；外部访问时保持 `0.0.0.0`。 |
| `--port` | 否 | `8000` | 容器内部监听端口。 |

在仓库根目录本地构建：

```bash
docker build -f docker/rk3588/deeplabv3.dockerfile \
  -t rk3588-deeplabv3:local src/rk3588_deeplabv3
```

## API 接口文档

### 模型推理

**接口：** `POST /api/models/deeplabv3/predict`
**类型：** `multipart/form-data`

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | JPEG/PNG 或其他 OpenCV 可解码图片。 |
| `overlay_alpha` | 否 | 单次请求的掩码透明度，范围 `0`–`1`；仅覆盖当前请求。 |

```bash
curl -X POST http://127.0.0.1:8000/api/models/deeplabv3/predict \
  -F "file=@model/test.jpg" \
  -F "overlay_alpha=0.65"
```

响应包含推理耗时、图像尺寸、透明度，以及画面中所有 PASCAL VOC 类别及对应像素数。渲染后的叠加图可通过 `GET /api/video_feed` 查看。

### 全局配置

```bash
curl http://127.0.0.1:8000/api/config

curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"overlay_alpha":0.7}'
```

### 其他接口

- `GET /api/health`：模型/平台就绪状态，以及实时源模式、运行状态、帧数、延迟和错误。
- `GET /api/video_feed`：最新分割叠加图的 MJPEG 流。
- `POST /api/video/upload`：通过 multipart `file` 上传 MP4。
- `POST /api/video/analyze`：通过 multipart `filename` 启动异步处理。
- `GET /api/video/status`、`GET /api/video/list`、`GET /api/video/download/{filename}`：进度和结果管理。

## 处理流程与模型替换

输入被缩放至 513×513 并从 BGR 转换为 RGB。服务兼容 NHWC/NCHW RKNN 输出，验证输出为 21 类，将 logits 恢复到原图尺寸，执行 `argmax`，再叠加 PASCAL VOC 调色板。

更换模型时可挂载兼容模型并用 `--model_path` 指定。若类别数量不同，需要同步修改 `task_runtime.py` 中的 `LABELS` 和输出处理。
