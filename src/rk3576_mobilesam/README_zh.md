# RK3576 MobileSAM 部署指南

[English](./README.md) | [中文]

本目录将 Mobile Segment Anything（MobileSAM）封装为适用于 RK3576 的 reComputer RK-CV 标准服务。图像编码器和提示解码器均使用 RKNN 模型运行，并通过浏览器预览和 REST API 提供提示式图片分割及异步 MP4 处理。

## 核心特性

- 通过 RKNN-Toolkit-Lite2 在 RK3576 NPU 上运行 MobileSAM 编码器和解码器。
- 通过 `point_coords` 和 `point_labels` 接收点提示或框提示。
- 返回掩码质量分数、选中掩码序号和掩码像素数。
- 支持图片推理、MJPEG 结果预览、上传 MP4 分析、OpenAPI 和 Docker 部署。

## 目录结构

- `model/mobilesam_encoder.rknn`：图像编码器。
- `model/mobilesam_decoder.rknn`：固定双提示输入的解码器。
- `model/picture.jpg`：启动预热和初始预览图片。
- `task_runtime.py`：448 × 448 预处理、提示坐标缩放、解码器推理和掩码渲染。
- `web_service.py`：Web 页面、REST API、预览流和异步 MP4 处理。
- `video/`：示例视频资源。
- `lib/`、`rknn-toolkit-lite2-packages/`：RKNN 运行时依赖。

## 快速开始

### 1. 运行已发布 Docker 镜像

```bash
sudo docker run --rm --name rk3576-mobilesam \
  --privileged \
  -p 8000:8000 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilesam:latest
```

访问 `http://<开发板IP>:8000` 上传图片并查看结果；访问 `http://<开发板IP>:8000/docs` 可通过 OpenAPI 传入自定义提示。若宿主机 8000 端口被占用，可改用 `-p 8001:8000`。

### 2. 调用本地摄像头

映射采集节点并传入对应数字 ID。下面以 `/dev/video0` 为例：

```bash
sudo docker run --rm --name rk3576-mobilesam-camera \
  --privileged \
  -p 8000:8000 \
  --device /dev/video0:/dev/video0 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilesam:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

如果采集节点为 `/dev/video1`，请映射该节点并使用 `--camera_id 1`。服务会持续处理画面，并在首页和 `GET /api/video_feed` 中显示结果。

### 3. 分析本地视频

```bash
sudo docker run --rm --name rk3576-mobilesam-video \
  --privileged \
  -p 8000:8000 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  -v /本机绝对路径/input.mp4:/data/input.mp4:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilesam:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --video_path /data/input.mp4 --host 0.0.0.0 --port 8000
```

`--video_path` 优先于 `--camera_id`，并在视频结束后自动循环，以保持 Web 实时预览；`--video` 是等价别名。

### 4. 仅 Web 上传模式

镜像默认使用 `--camera_id -1`，只启动 Web/API 而不打开摄像头。图片可通过首页或推理 API 提交，MP4 使用上传和分析 API。

### 5. 本地构建

```bash
docker build -f docker/rk3576/mobilesam.dockerfile \
  -t rk3576-mobilesam:local src/rk3576_mobilesam
```

### 启动参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--platform` | 是 | — | 目标 NPU；本目录使用 `rk3576`。 |
| `--model_dir` | 否 | `model` | 同时包含两个 RKNN 文件和 `picture.jpg` 的目录。 |
| `--camera_id` | 否 | `-1` | 摄像头索引 `N` 对应 `/dev/videoN`；`-1` 表示禁用摄像头采集。 |
| `--video_path`、`--video` | 否 | — | 本地视频路径，优先于 `--camera_id` 并循环播放。 |
| `--host` | 否 | `0.0.0.0` | FastAPI 监听地址。 |
| `--port` | 否 | `8000` | 容器内部服务端口。 |

## API 接口文档

### 1. 提示式图片分割

**接口：** `POST /api/models/mobilesam/predict`

**类型：** `multipart/form-data`

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | OpenCV 可解码的图片。 |
| `point_coords` | 否 | JSON 数组，必须包含两个原图坐标；默认 `[[190,70],[460,280]]`。 |
| `point_labels` | 否 | JSON 数组，必须包含两个标签；默认 `[2,3]`。 |

提示标签遵循 SAM 约定：`0` 表示负点，`1` 表示正点，`2` 表示框左上角，`3` 表示框右下角，`-1` 表示填充点。当前转换后的解码器始终要求恰好两个坐标和两个标签。

框提示示例：

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilesam/predict \
  -F "file=@model/picture.jpg" \
  -F 'point_coords=[[190,70],[460,280]]' \
  -F 'point_labels=[2,3]'
```

响应包含 `iou_scores`、`selected_mask` 和 `mask_pixels`。选中的掩码会叠加到原图，并发布到 `GET /api/video_feed`。

### 2. 配置接口

`GET /api/config` 和 `POST /api/config` 是标准服务共用的兼容接口。当前 MobileSAM 运行时从每次推理请求读取提示，通用的 `threshold` 和 `topk` 参数不会改变选中的掩码。

### 3. 视频分析与服务接口

- `GET /api/health`：平台、就绪状态以及已加载的编码器/解码器名称。
- `GET /api/video_feed`：最新 MobileSAM 叠加结果的 MJPEG 流。
- `POST /api/video/upload`：通过 multipart 字段 `file` 上传一个 `.mp4`。
- `POST /api/video/analyze`：通过 multipart 字段 `filename` 处理已上传文件。
- `GET /api/video/status`：后台进度和错误信息。
- `GET /api/video/list`：已上传和已生成的 MP4 文件。
- `GET /api/video/download/{filename}`：下载结果。

上传视频分析使用默认框提示，因为当前视频任务接口不接收单次任务的提示字段。

## 开发者指南

编码器保持宽高比，将长边缩放到 448，再把图像填充为 448 × 448；提示坐标会同步缩放到该张量。解码器输出候选掩码和 IoU 分数，服务选取分数最高的掩码，恢复至原图尺寸后进行叠加。

转换脚本和 ONNX 源文件位于 `rknn_model_zoo/examples/mobilesam`。编码器与解码器必须转换为同一目标平台，并确保解码器输入形状继续满足 `task_runtime.py` 使用的固定双提示约定。
