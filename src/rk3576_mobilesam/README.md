# RK3576 MobileSAM Deployment Guide

[English] | [中文](./README_zh.md)

This directory packages Mobile Segment Anything (MobileSAM) as a standard reComputer RK-CV service for RK3576. The encoder and prompt decoder run as RKNN models, while the browser preview and REST API expose prompt-based image segmentation and asynchronous MP4 processing.

## Core Features

- Runs the MobileSAM encoder and decoder on the RK3576 NPU with RKNN-Toolkit-Lite2.
- Accepts point or box prompts through `point_coords` and `point_labels`.
- Returns mask quality scores, the selected mask index, and mask pixel count.
- Supports image inference, MJPEG result preview, uploaded MP4 analysis, OpenAPI, and Docker deployment.

## Directory Structure

- `model/mobilesam_encoder.rknn`: image encoder.
- `model/mobilesam_decoder.rknn`: fixed two-prompt decoder.
- `model/picture.jpg`: startup warm-up and preview image.
- `task_runtime.py`: 448 × 448 preprocessing, prompt scaling, decoder inference, and mask rendering.
- `web_service.py`: Web UI, REST API, preview stream, and asynchronous MP4 processing.
- `video/`: sample video assets.
- `lib/` and `rknn-toolkit-lite2-packages/`: RKNN runtime dependencies.

## Quick Start

### 1. Run the Published Docker Image

```bash
sudo docker run --rm --name rk3576-mobilesam \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilesam:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id -1 --host 0.0.0.0 --port 8000
```

Open `http://<BOARD_IP>:8000` for image upload and result preview. Open `http://<BOARD_IP>:8000/docs` to supply custom prompts through OpenAPI. Because `--net=host` is used, change `--port` if port 8000 is occupied.

### 2. Analyze a Local Camera

Map the capture node and pass its numeric ID. This example uses `/dev/video0`:

```bash
sudo docker run --rm --name rk3576-mobilesam-camera \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilesam:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

For `/dev/video1`, map that node and use `--camera_id 1`. Frames are processed continuously and shown on the home page and `GET /api/video_feed`.

### 3. Analyze a Local Video

```bash
sudo docker run --rm --name rk3576-mobilesam-video \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilesam:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --video video/test.mp4 --host 0.0.0.0 --port 8000
```

`--video` and `--video_path` are equivalent. Either option takes precedence over `--camera_id` and loops the file for continuous Web preview. The command above uses the image's built-in `video/test.mp4`.

### 4. Web-upload-only Mode

The image defaults to `--camera_id -1`, which starts the Web/API service without opening a camera. Images can be submitted on the home page or prediction API; MP4 files use the upload and analysis APIs.

### 5. Build Locally

```bash
docker build -f docker/rk3576/mobilesam.dockerfile \
  -t rk3576-mobilesam:local src/rk3576_mobilesam
```

### Startup Parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | Target NPU; use `rk3576` for this directory. |
| `--model_dir` | No | `model` | Directory containing both RKNN files and `picture.jpg`. |
| `--camera_id` | No | `-1` | Camera index `N` for `/dev/videoN`; `-1` disables camera capture. |
| `--video_path`, `--video` | No | — | Local video path. Overrides `--camera_id` and loops continuously. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. |
| `--port` | No | `8000` | Service port inside the container. |

`PYTHONUNBUFFERED=1` flushes service logs immediately. `RKNN_LOG_LEVEL=0` hides known harmless static-model initialization messages; remove it when diagnosing RKNN startup. `renderD129` is the NPU node verified on the RK3576 test board; check `/dev/dri/` if your board exposes a different node.

## API Documentation

### 1. Prompt-based Segmentation

**Endpoint:** `POST /api/models/mobilesam/predict`

**Content type:** `multipart/form-data`

| Field | Required | Description |
| --- | --- | --- |
| `file` | Yes | Image decodable by OpenCV. |
| `point_coords` | No | JSON array containing exactly two source-image coordinates; default `[[190,70],[460,280]]`. |
| `point_labels` | No | JSON array containing exactly two labels; default `[2,3]`. |

Prompt labels follow SAM conventions: `0` is a negative point, `1` is a positive point, `2` is the top-left box corner, `3` is the bottom-right box corner, and `-1` is padding. This converted decoder always requires exactly two coordinates and two labels.

Box prompt example:

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilesam/predict \
  -F "file=@model/picture.jpg" \
  -F 'point_coords=[[190,70],[460,280]]' \
  -F 'point_labels=[2,3]'
```

The response contains `iou_scores`, `selected_mask`, and `mask_pixels`. The selected mask is overlaid on the source image and published at `GET /api/video_feed`.

### 2. Configuration

`GET /api/config` and `POST /api/config` are compatibility endpoints shared by the standard service. The current MobileSAM runtime takes its prompts from each prediction request; the generic `threshold` and `topk` values do not change the selected mask.

### 3. Video Analysis and Service Endpoints

- `GET /api/health`: platform, readiness, and the loaded encoder/decoder names.
- `GET /api/video_feed`: latest MobileSAM overlay as MJPEG.
- `POST /api/video/upload`: upload one `.mp4` through multipart field `file`.
- `POST /api/video/analyze`: process an uploaded file through multipart field `filename`.
- `GET /api/video/status`: background progress and errors.
- `GET /api/video/list`: uploaded and generated MP4 files.
- `GET /api/video/download/{filename}`: download the result.

Uploaded videos use the default box prompt because the video-analysis endpoint currently does not accept per-job prompt fields.

## Developer Guide

The encoder preserves aspect ratio, resizes the long side to 448, and pads the image to 448 × 448. Prompt coordinates are scaled into that tensor. The decoder produces candidate masks and IoU scores; the highest-scoring mask is restored to the source resolution and overlaid.

Conversion scripts and ONNX sources remain in `rknn_model_zoo/examples/mobilesam`. The encoder and decoder must be converted for the same target platform, and the decoder input shape must continue to match the fixed two-prompt contract used by `task_runtime.py`.
