# RK3588 DeepLabV3 Deployment Guide

[English] | [中文](./README_zh.md)

This directory provides a standardized DeepLabV3 semantic-segmentation deployment for RK3588, including RKNN NPU inference, browser preview, REST API, video processing, and Docker startup.

## Core Features

- **NPU acceleration**: Runs `deeplabv3.rknn` through RKNN Toolkit Lite2 on RK3588.
- **Semantic segmentation**: Produces a 21-class PASCAL VOC mask and reports the pixel count of every class present.
- **Visual preview**: Blends the color mask with the source image and exposes it as an MJPEG stream.
- **Standard service**: Provides health, prediction, configuration, video-analysis, and OpenAPI endpoints.

## Directory Structure

- `lib/`: RK3588 RKNN runtime library.
- `model/deeplabv3.rknn` and `model/test.jpg`: RKNN model and warm-up sample.
- `rknn_runtime.py`: Thread-safe RKNN loader and inference wrapper.
- `task_runtime.py`: 513×513 preprocessing, mask decoding, PASCAL VOC labels/colors, and preview generation.
- `web_service.py`: FastAPI service, browser UI, MJPEG preview, and video processing.
- `video/test.mp4`: Sample video.
- `requirements.txt` and `rknn-toolkit-lite2-packages/`: Runtime dependencies.

## Quick Start

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-deeplabv3:latest
```

Open `http://<BOARD_IP>:8000` for preview or `http://<BOARD_IP>:8000/docs` for OpenAPI.

Build locally from the repository root:

```bash
docker build -f docker/rk3588/deeplabv3.dockerfile \
  -t rk3588-deeplabv3:local src/rk3588_deeplabv3
```

## API Documentation

### Semantic-segmentation prediction

**Endpoint:** `POST /api/models/deeplabv3/predict`
**Content type:** `multipart/form-data`

- `file`: Required image.
- `threshold` and `topk` are accepted by the common service layer but are not used by the current argmax decoder.

```bash
curl -X POST http://127.0.0.1:8000/api/models/deeplabv3/predict \
  -F "file=@model/test.jpg"
```

```json
{
  "success": true,
  "model": "deeplabv3",
  "inference_time": 0.0842,
  "result": {
    "classes": [
      {"id": 0, "class": "background", "pixels": 180240},
      {"id": 15, "class": "person", "pixels": 26418}
    ],
    "width": 640,
    "height": 480
  }
}
```

The JSON contains mask statistics; the rendered overlay is available from `GET /api/video_feed`.

### Configuration and video endpoints

- `GET /api/health`: Model, platform, input type, model file, and readiness.
- `GET /api/config` and `POST /api/config`: Common configuration (`threshold` 0–1 and `topk` 1–100).
- `GET /api/video_feed`: Latest overlay as MJPEG.
- `POST /api/video/upload` and `POST /api/video/analyze`: Upload an MP4 and start per-frame segmentation with a multipart `filename` field.
- `GET /api/video/status`, `GET /api/video/list`, `GET /api/video/download/{filename}`: Query and retrieve results.

## Developer Guide

The pipeline resizes BGR input to 513×513, converts it to RGB, runs RKNN inference, restores logits to the original resolution, takes `argmax` over 21 PASCAL VOC classes, and overlays the color mask at 50% opacity.

To replace the model, put a RK3588-converted `deeplabv3.rknn` in `model/`. Preserve the 21-class output layout or update `LABELS` and output handling in `task_runtime.py`.
