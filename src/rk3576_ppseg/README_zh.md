# RK3576 PPSeg 部署指南

[English](./README.md) | [中文]

本目录将官方 PP-LiteSeg Cityscapes 案例封装为适用于 RK3576 的 reComputer RK-CV 标准服务，提供 RKNN NPU 推理、浏览器预览、REST API、异步 MP4 分析和 ARM64 Docker 镜像。

## 核心特性

- 通过 RKNN-Toolkit-Lite2 在 RK3576 NPU 上运行 `ppseg.rknn`。
- 输出 Cityscapes 19 类语义分割掩码，并返回各类别的像素数量。
- 支持图片推理、MJPEG 结果预览和上传 MP4 分析。
- 提供 FastAPI/OpenAPI 接口以及带健康检查的 Docker 运行环境。

## 目录结构

- `model/`：RK3576 RKNN 模型和预热图片 `test.png`。
- `video/`：示例视频资源。
- `task_runtime.py`：512 × 512 预处理、RKNN 推理、Cityscapes 后处理和叠加渲染。
- `web_service.py`：Web 页面、REST API、预览流和异步视频处理。
- `lib/`、`rknn-toolkit-lite2-packages/`：RKNN 运行时依赖。

## 快速开始

### 1. 运行已发布 Docker 镜像

```bash
sudo docker run --rm --name rk3576-ppseg \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-ppseg:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id -1 --host 0.0.0.0 --port 8000
```

访问 `http://<开发板IP>:8000` 查看预览页面，访问 `http://<开发板IP>:8000/docs` 查看 OpenAPI。命令使用 `--net=host`；若 8000 端口被占用，请修改 `--port`。

### 2. 调用本地摄像头

映射采集节点并传入对应数字 ID。下面以 `/dev/video0` 为例：

```bash
sudo docker run --rm --name rk3576-ppseg-camera \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-ppseg:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

如果采集节点为 `/dev/video1`，请映射该节点并使用 `--camera_id 1`。服务会持续处理画面，并在首页和 `GET /api/video_feed` 中显示结果。

### 3. 分析本地视频

```bash
sudo docker run --rm --name rk3576-ppseg-video \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-ppseg:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --video video/test.mp4 --host 0.0.0.0 --port 8000
```

`--video` 与 `--video_path` 等价，均优先于 `--camera_id`，并在视频结束后自动循环，以保持 Web 实时预览。上面的命令直接使用镜像内置的 `video/test.mp4`。

### 4. 仅 Web 上传模式

镜像默认使用 `--camera_id -1`，只启动 Web/API 而不打开摄像头。图片可通过首页或推理 API 提交，MP4 使用上传和分析 API。

### 5. 本地构建

在仓库根目录执行：

```bash
docker build -f docker/rk3576/ppseg.dockerfile \
  -t rk3576-ppseg:local src/rk3576_ppseg
```

### 启动参数

Docker 镜像已提供以下默认参数：

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--platform` | 是 | — | 目标 NPU；本目录使用 `rk3576`。 |
| `--model_dir` | 否 | `model` | 包含 `ppseg.rknn` 和 `test.png` 的目录。 |
| `--camera_id` | 否 | `-1` | 摄像头索引 `N` 对应 `/dev/videoN`；`-1` 表示禁用摄像头采集。 |
| `--video_path`、`--video` | 否 | — | 本地视频路径，优先于 `--camera_id` 并循环播放。 |
| `--host` | 否 | `0.0.0.0` | FastAPI 监听地址。 |
| `--port` | 否 | `8000` | 容器内部服务端口。 |

`PYTHONUNBUFFERED=1` 让服务日志即时输出；`RKNN_LOG_LEVEL=0` 会隐藏已确认无害的静态模型初始化提示。排查 RKNN 初始化问题时可去掉后一个环境变量。`renderD129` 是已在 RK3576 开发板上验证的 NPU 节点；如设备节点不同，请以本机 `/dev/dri/` 为准。

## API 接口文档

### 1. 图片语义分割

**接口：** `POST /api/models/ppseg/predict`

**类型：** `multipart/form-data`

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | OpenCV 可解码的图片，例如 JPEG 或 PNG。 |

```bash
curl -X POST http://127.0.0.1:8000/api/models/ppseg/predict \
  -F "file=@model/test.png"
```

响应包含推理耗时、原图尺寸，以及画面中各 Cityscapes 类别和对应像素数。渲染后的彩色掩码会成为 `GET /api/video_feed` 的最新画面。

### 2. 配置接口

`GET /api/config` 和 `POST /api/config` 用于兼容标准服务接口。当前 PP-LiteSeg 后处理使用固定的 argmax 掩码，因此通用的 `threshold` 和 `topk` 参数不会改变分割结果。

### 3. 视频分析与服务接口

- `GET /api/health`：模型名称、平台、已加载 RKNN 文件和就绪状态。
- `GET /api/video_feed`：最新彩色掩码叠加结果的 MJPEG 流。
- `POST /api/video/upload`：通过 multipart 字段 `file` 上传一个 `.mp4`。
- `POST /api/video/analyze`：通过 multipart 字段 `filename` 启动后台处理。
- `GET /api/video/status`：处理状态、进度、当前文件和错误信息。
- `GET /api/video/list`：列出已上传和已生成的 MP4。
- `GET /api/video/download/{filename}`：下载生成结果。

```bash
curl -X POST http://127.0.0.1:8000/api/video/upload -F "file=@video/test.mp4"
curl -X POST http://127.0.0.1:8000/api/video/analyze -F "filename=test.mp4"
curl http://127.0.0.1:8000/api/video/status
```

## 开发者指南

服务将输入缩放到 512 × 512 并从 BGR 转为 RGB，兼容 NCHW/NHWC 的 19 类输出，通过最近邻插值恢复到原图大小，最后叠加 Cityscapes 调色板。

模型转换源文件位于 `rknn_model_zoo/examples/ppseg`。替换模型时请保持文件名为 `ppseg.rknn`，或同步修改 `task_runtime.py`，并确认输出仍为相同的 19 类。
