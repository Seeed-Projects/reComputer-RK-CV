# RK3588 RetinaFace Deployment Guide

[English] | [中文](./README_zh.md)

This directory packages RetinaFace face detection as a standard reComputer RK-CV service for RK3588. It provides RKNN NPU inference, five-point facial landmarks, browser preview, REST APIs, asynchronous MP4 processing, and an ARM64 Docker image.

## Core Features

- Runs the 320 × 320 MobileNet-backbone RetinaFace RKNN model on the RK3588 NPU.
- Returns face confidence, bounding boxes, and five landmarks for every accepted face.
- Supports a configurable confidence threshold, image inference, MJPEG preview, and uploaded MP4 analysis.
- Provides FastAPI/OpenAPI endpoints and a health-checked Docker runtime.

## Directory Structure

- `model/retinaface_mobile.rknn`: model loaded by the current runtime.
- `model/retinaface_resnet50.rknn`: packaged alternative model; it is not selected automatically.
- `model/test.jpg`: startup warm-up and preview image.
- `py_utils/retinaface_official.py`: prior generation, box/landmark decoding, letterbox helpers, and NMS.
- `task_runtime.py`: RKNN inference, filtering, landmark rendering, and result serialization.
- `web_service.py`: Web UI, REST API, preview stream, and asynchronous MP4 processing.
- `video/`: sample video assets.

## Quick Start

### 1. Run the Published Docker Image

```bash
sudo docker run --rm --name rk3588-retinaface \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-retinaface:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --camera_id -1 --host 0.0.0.0 --port 8000
```

Open `http://<BOARD_IP>:8000` for image upload and annotated preview or `http://<BOARD_IP>:8000/docs` for OpenAPI. Because `--net=host` is used, change `--port` if port 8000 is occupied.

### 2. Analyze a Local Camera

Map the capture node and pass its numeric ID. This example uses `/dev/video0`:

```bash
sudo docker run --rm --name rk3588-retinaface-camera \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-retinaface:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

For `/dev/video1`, map that node and use `--camera_id 1`. Frames are processed continuously and shown on the home page and `GET /api/video_feed`.

### 3. Analyze a Local Video

```bash
sudo docker run --rm --name rk3588-retinaface-video \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-retinaface:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --video video/test.mp4 --host 0.0.0.0 --port 8000
```

`--video` and `--video_path` are equivalent. Either option takes precedence over `--camera_id` and loops the file for continuous Web preview. The command above uses the image's built-in `video/test.mp4`.

### 4. Web-upload-only Mode

The image defaults to `--camera_id -1`, which starts the Web/API service without opening a camera. Images can be submitted on the home page or prediction API; MP4 files use the upload and analysis APIs.

### 5. Build Locally

```bash
docker build -f docker/rk3588/retinaface.dockerfile \
  -t rk3588-retinaface:local src/rk3588_retinaface
```

### Startup Parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | Target NPU; use `rk3588` for this directory. |
| `--model_dir` | No | `model` | Directory containing RetinaFace RKNN files and `test.jpg`. |
| `--camera_id` | No | `-1` | Camera index `N` for `/dev/videoN`; `-1` disables camera capture. |
| `--video_path`, `--video` | No | — | Local video path. Overrides `--camera_id` and loops continuously. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. |
| `--port` | No | `8000` | Service port inside the container. |

`PYTHONUNBUFFERED=1` flushes service logs immediately. `RKNN_LOG_LEVEL=0` hides known harmless static-model initialization messages; remove it when diagnosing RKNN startup. `renderD129` is the NPU node verified on the RK3588 test board; check `/dev/dri/` if your board exposes a different node.

## API Documentation

### 1. Face Detection

**Endpoint:** `POST /api/models/retinaface/predict`

**Content type:** `multipart/form-data`

| Field | Required | Description |
| --- | --- | --- |
| `file` | Yes | Image decodable by OpenCV. |
| `threshold` | No | Per-request face confidence threshold; default is the global value `0.25`. |

```bash
curl -X POST http://127.0.0.1:8000/api/models/retinaface/predict \
  -F "file=@model/test.jpg" \
  -F "threshold=0.5"
```

Each item in `result.faces` contains `confidence`, `box` as `[x1,y1,x2,y2]`, and five `landmarks`. `result.count` reports the accepted face count. The annotated image becomes the latest frame at `GET /api/video_feed`.

### 2. Confidence Configuration

`GET /api/config` returns the global configuration. Update the threshold used by preview/video processing with JSON:

```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"threshold":0.5}'
```

`threshold` must be from 0 to 1. The generic `topk` field is reserved for standard-interface compatibility and is not used by RetinaFace postprocessing. NMS uses a fixed IoU threshold of 0.5.

### 3. Video Analysis and Service Endpoints

- `GET /api/health`: platform, readiness, and packaged model names.
- `GET /api/video_feed`: latest face boxes and landmarks as MJPEG.
- `POST /api/video/upload`: upload one `.mp4` through multipart field `file`.
- `POST /api/video/analyze`: process an upload through multipart field `filename`.
- `GET /api/video/status`: processing progress and errors.
- `GET /api/video/list`: uploaded and generated MP4 files.
- `GET /api/video/download/{filename}`: download a generated result.

```bash
curl -X POST http://127.0.0.1:8000/api/video/upload -F "file=@video/test.mp4"
curl -X POST http://127.0.0.1:8000/api/video/analyze -F "filename=test.mp4"
curl http://127.0.0.1:8000/api/video/status
```

## Developer Guide

Frames are letterboxed to 320 × 320, converted from BGR to RGB, and passed to the RKNN model. The runtime decodes prior-based boxes and five landmarks, restores source-image coordinates, applies NMS at 0.5, and filters results with the requested confidence threshold.

Conversion scripts and original ONNX models remain in `rknn_model_zoo/examples/RetinaFace`. The current constructor explicitly loads `retinaface_mobile.rknn`. Selecting `retinaface_resnet50.rknn` requires updating that path in `task_runtime.py` and verifying the same three-output layout.
