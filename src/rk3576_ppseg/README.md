# RK3576 PPSeg Deployment Guide

[English] | [中文](./README_zh.md)

This directory packages the official PP-LiteSeg Cityscapes example as a standard reComputer RK-CV service for RK3576. It provides RKNN NPU inference, browser preview, REST APIs, asynchronous MP4 processing, and an ARM64 Docker image.

## Core Features

- Runs `ppseg.rknn` with RKNN-Toolkit-Lite2 on the RK3576 NPU.
- Produces semantic masks for the 19 Cityscapes classes and reports per-class pixel counts.
- Supports image inference, MJPEG result preview, and uploaded MP4 analysis.
- Provides FastAPI/OpenAPI endpoints and a health-checked Docker runtime.

## Directory Structure

- `model/`: RK3576 RKNN model and the warm-up image `test.png`.
- `video/`: sample video assets.
- `task_runtime.py`: 512 × 512 preprocessing, RKNN inference, Cityscapes postprocessing, and overlay rendering.
- `web_service.py`: Web UI, REST API, preview stream, and asynchronous video processing.
- `lib/` and `rknn-toolkit-lite2-packages/`: RKNN runtime dependencies.

## Quick Start

### 1. Run the Published Docker Image

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

Open `http://<BOARD_IP>:8000` for the preview page or `http://<BOARD_IP>:8000/docs` for OpenAPI. Because `--net=host` is used, change `--port` if port 8000 is occupied.

### 2. Analyze a Local Camera

Map the capture node and pass its numeric ID. This example uses `/dev/video0`:

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

For `/dev/video1`, map that node and use `--camera_id 1`. Frames are processed continuously and shown on the home page and `GET /api/video_feed`.

### 3. Analyze a Local Video

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

`--video` and `--video_path` are equivalent. Either option takes precedence over `--camera_id` and loops the file for continuous Web preview. The command above uses the image's built-in `video/test.mp4`.

### 4. Web-upload-only Mode

The image defaults to `--camera_id -1`, which starts the Web/API service without opening a camera. Images can be submitted on the home page or prediction API; MP4 files use the upload and analysis APIs.

### 5. Build Locally

From the repository root:

```bash
docker build -f docker/rk3576/ppseg.dockerfile \
  -t rk3576-ppseg:local src/rk3576_ppseg
```

### Startup Parameters

The Docker image supplies these parameters by default:

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | Target NPU; use `rk3576` for this directory. |
| `--model_dir` | No | `model` | Directory containing `ppseg.rknn` and `test.png`. |
| `--camera_id` | No | `-1` | Camera index `N` for `/dev/videoN`; `-1` disables camera capture. |
| `--video_path`, `--video` | No | — | Local video path. Overrides `--camera_id` and loops continuously. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. |
| `--port` | No | `8000` | Service port inside the container. |

`PYTHONUNBUFFERED=1` flushes service logs immediately. `RKNN_LOG_LEVEL=0` hides known harmless static-model initialization messages; remove it when diagnosing RKNN startup. `renderD129` is the NPU node verified on the RK3576 test board; check `/dev/dri/` if your board exposes a different node.

## API Documentation

### 1. Image Segmentation

**Endpoint:** `POST /api/models/ppseg/predict`

**Content type:** `multipart/form-data`

| Field | Required | Description |
| --- | --- | --- |
| `file` | Yes | Image decodable by OpenCV, such as JPEG or PNG. |

```bash
curl -X POST http://127.0.0.1:8000/api/models/ppseg/predict \
  -F "file=@model/test.png"
```

The response includes inference time, source dimensions, and each detected Cityscapes class with its pixel count. The rendered mask becomes the latest frame at `GET /api/video_feed`.

### 2. Configuration

`GET /api/config` and `POST /api/config` are retained for compatibility with the standard service interface. The current PP-LiteSeg postprocessing uses a fixed argmax mask, so the generic `threshold` and `topk` values do not change segmentation results.

### 3. Video Analysis and Service Endpoints

- `GET /api/health`: model name, platform, loaded RKNN files, and readiness.
- `GET /api/video_feed`: latest color-mask overlay as MJPEG.
- `POST /api/video/upload`: upload one `.mp4` through multipart field `file`.
- `POST /api/video/analyze`: start background processing through multipart field `filename`.
- `GET /api/video/status`: processing state, progress, current file, and error.
- `GET /api/video/list`: uploaded and generated MP4 files.
- `GET /api/video/download/{filename}`: download a generated result.

```bash
curl -X POST http://127.0.0.1:8000/api/video/upload -F "file=@video/test.mp4"
curl -X POST http://127.0.0.1:8000/api/video/analyze -F "filename=test.mp4"
curl http://127.0.0.1:8000/api/video/status
```

## Developer Guide

The service resizes input to 512 × 512, converts BGR to RGB, accepts either NCHW or NHWC 19-class output, restores the mask to source resolution with nearest-neighbor interpolation, and overlays the Cityscapes palette.

Model conversion sources remain in `rknn_model_zoo/examples/ppseg`. When replacing the model, keep the filename `ppseg.rknn` or update `task_runtime.py`, and verify that the output still contains the same 19 classes.
