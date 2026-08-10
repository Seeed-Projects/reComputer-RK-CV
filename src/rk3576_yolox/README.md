# RK3576 YOLOX Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains a YOLOX object detection service optimized for RK3576. It provides RKNN inference, browser preview, REST APIs, offline video processing, and a ready-to-run Docker image.

## Core Features

- **Hardware Acceleration**: Runs YOLOX RKNN models on the RK3576 NPU through RKNN-Toolkit-Lite2.
- **COCO Object Detection**: Uses `640 x 640` letterbox preprocessing and returns classes, confidence scores, and bounding boxes.
- **Multiple Models**: Includes YOLOX-S and YOLOX-M; `yolox_s.rknn` is used by default.
- **Model-specific Post-processing**: Implements YOLOX branch decoding, objectness/class score fusion, and class-aware NMS.
- **Flexible Input**: Supports image uploads, selected MP4 frames, local MP4 playback, and camera input.
- **Web and API Access**: Includes a browser preview, MJPEG stream, health check, and REST inference APIs.

## Directory Structure

- `lib/`: RK3576 `librknnrt.so` runtime library.
- `model/`: YOLOX RKNN models, COCO labels, and a sample image.
- `py_utils/`: Letterbox, coordinate restoration, drawing, and RKNN helper utilities.
- `video/`: Sample MP4 video used by the default container command.
- `web_detection.py`: Main application providing detection, Web preview, and FastAPI endpoints.
- `requirements.txt`: Python runtime dependencies.

Available models are `model/yolox_s.rknn` and `model/yolox_m.rknn`. The default is `model/yolox_s.rknn`.

## Quick Start

### 1. Run with Docker

Run the published image directly on an RK3576 board:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolox:latest
```

The default command continuously analyzes the bundled `video/test.mp4`. Open `http://<Board_IP>:8000` for the Web preview or `http://<Board_IP>:8000/docs` for interactive API documentation.

To use `yolox_m.rknn` instead:

```bash
sudo docker run --rm --privileged --net=host \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolox:latest \
    python3 web_detection.py --platform rk3576 \
    --model_path model/yolox_m.rknn --class_path model/coco_80_labels_list.txt \
    --video_path video/test.mp4
```

For upload-only Web mode, replace `--video_path video/test.mp4` with `--camera_id -1`.

### 2. Build the Image Locally

Run from the repository root:

```bash
docker build -f docker/rk3576/yolox.dockerfile \
    -t rk3576-yolox:local src/rk3576_yolox
```

## API Documentation

### 1. Model Inference Interface (Predict)

**Endpoint:** `POST /api/models/yolox/predict`

#### Request Parameters (`multipart/form-data`)

- `file`: Optional image file.
- `video`: Optional MP4 video file.
- `timestamp`: Optional timestamp in seconds for selecting a video frame; defaults to the first frame.
- `realtime`: Optional boolean. With no uploaded file, the current camera or local-video frame is used.
- `conf`: Optional confidence threshold for this request; the default is `0.25`.
- `iou`: Optional IoU threshold; the default exposed value is `0.45`. The `iou` field controls class-aware NMS for the current request.

#### Usage Examples

```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolox/predict" \
    -F "file=@model/bus.jpg" -F "conf=0.25" -F "iou=0.45"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolox/predict" \
    -F "video=@video/test.mp4" -F "timestamp=5.5"
```

#### Response Format

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "bus",
      "confidence": 0.91,
      "box": {"x1": 100, "y1": 120, "x2": 520, "y2": 430}
    }
  ],
  "image": {"width": 640, "height": 480}
}
```

### 2. System Configuration Interface (Config)

- `GET /api/config`: Returns `{"obj_thresh": 0.25, "nms_thresh": 0.45}`.
- `POST /api/config`: Accepts JSON such as `{"obj_thresh": 0.3, "nms_thresh": 0.5}`.

The global confidence threshold applies to the preview, offline analysis, and requests without `conf`. The `iou` field controls class-aware NMS for the current request.

### 3. Video and Service Interfaces

- `GET /api/video_feed`: Returns the annotated MJPEG preview stream.
- `POST /api/video/upload`: Uploads an MP4 file using the `file` form field.
- `POST /api/video/analyze`: Starts asynchronous processing using the uploaded `filename` form field.
- `GET /api/video/status`: Returns processing progress and error state.
- `GET /api/video/list`: Lists uploaded and generated videos.
- `GET /api/video/download/{filename}`: Downloads a generated MP4 result.
- `GET /api/health`: Returns platform, model file, and model readiness information.

## Developer Guide

### Code Description

- `web_detection.py` initializes the RK3576 NPU runtime, loops camera or video input, exposes FastAPI endpoints, and implements YOLOX branch decoding, objectness/class score fusion, and class-aware NMS.
- Input frames are letterboxed to `640 x 640`, converted from BGR to RGB, and detections are mapped back to the source resolution.
- Runtime uploads and outputs are written under `workspace/` by default. Set `RK_CV_WORKSPACE` to change this location.

### Replacing the Model

1. Place an RK3576-compatible YOLOX `.rknn` model in `model/`.
2. Ensure its input size and output tensors match the preprocessing and YOLOX post-processing in `web_detection.py`.
3. Update the class configuration if the model does not use the default COCO 80 classes.
4. Pass the new file with `--model_path model/<model_name>.rknn`.
