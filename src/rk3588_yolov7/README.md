# RK3588 YOLOv7 Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains a YOLOv7 object detection service optimized for RK3588. It provides RKNN inference, browser preview, REST APIs, offline video processing, and a ready-to-run Docker image.

## Core Features

- **Hardware Acceleration**: Runs YOLOv7 RKNN models on the RK3588 NPU through RKNN-Toolkit-Lite2.
- **COCO Object Detection**: Uses `640 x 640` letterbox preprocessing and returns classes, confidence scores, and bounding boxes.
- **Multiple Models**: Includes YOLOv7-Tiny and the full YOLOv7 model; YOLOv7-Tiny is used by default.
- **Anchor-based Decoding**: Loads `anchors_yolov7.txt` for YOLOv7 output decoding and NMS.
- **Flexible Input**: Supports image uploads, selected MP4 frames, local MP4 playback, and camera input.
- **Web and API Access**: Includes a browser preview, MJPEG stream, health check, and REST inference APIs.

## Directory Structure

- `lib/`: RK3588 `librknnrt.so` runtime library.
- `model/`: YOLOv7 RKNN models, anchor configuration, COCO labels, and a sample image.
- `py_utils/`: Letterbox, anchor loading, coordinate restoration, drawing, and RKNN helper utilities.
- `video/`: Sample MP4 video used by the default container command.
- `web_detection.py`: Main application providing detection, Web preview, and FastAPI endpoints.
- `requirements.txt`: Python runtime dependencies.

Available models are `model/yolov7-tiny.rknn` and `model/yolov7.rknn`. The default model is `model/yolov7-tiny.rknn`, with anchors loaded from `model/anchors_yolov7.txt`.

## Quick Start

### 1. Run with Docker

Run the published image directly on an RK3588 board:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov7:latest
```

The default command continuously analyzes the bundled `video/test.mp4`. Open `http://<Board_IP>:8000` for the Web preview or `http://<Board_IP>:8000/docs` for interactive API documentation.

To use the full YOLOv7 model instead:

```bash
sudo docker run --rm --privileged --net=host \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov7:latest \
    python3 web_detection.py --platform rk3588 \
    --model_path model/yolov7.rknn \
    --anchors model/anchors_yolov7.txt --video_path video/test.mp4
```

For upload-only Web mode, replace `--video_path video/test.mp4` with `--camera_id -1`.

### 2. Build the Image Locally

Run from the repository root:

```bash
docker build -f docker/rk3588/yolov7.dockerfile \
    -t rk3588-yolov7:local src/rk3588_yolov7
```

## API Documentation

### 1. Model Inference Interface (Predict)

**Endpoint:** `POST /api/models/yolov7/predict`

#### Request Parameters (`multipart/form-data`)

- `file`: Optional image file.
- `video`: Optional MP4 video file.
- `timestamp`: Optional timestamp in seconds for selecting a video frame; defaults to the first frame.
- `realtime`: Optional boolean. With no uploaded file, the current camera or local-video frame is used.
- `conf`: Optional confidence threshold for this request; the default is `0.25`.
- `iou`: Optional NMS IoU threshold for this request; the default is `0.45`.

#### Usage Examples

```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolov7/predict" \
    -F "file=@model/bus.jpg" -F "conf=0.25" -F "iou=0.45"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolov7/predict" \
    -F "video=@video/test.mp4" -F "timestamp=5.5"
```

#### Response Format

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "bus ",
      "confidence": 0.92,
      "box": {"x1": 100, "y1": 120, "x2": 520, "y2": 430}
    }
  ],
  "image": {"width": 640, "height": 480}
}
```

### 2. System Configuration Interface (Config)

- `GET /api/config`: Returns `{"obj_thresh": 0.25, "nms_thresh": 0.45}`.
- `POST /api/config`: Accepts JSON such as `{"obj_thresh": 0.3, "nms_thresh": 0.5}`.

The global thresholds apply to the preview, offline analysis, and requests that do not provide `conf` or `iou`.

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

- `web_detection.py` initializes the RK3588 NPU runtime, loops camera or video input, provides FastAPI endpoints, and implements YOLOv7 anchor decoding and NMS.
- Input frames are letterboxed to `640 x 640`, converted from BGR to RGB, and detections are mapped back to the source resolution.
- Runtime uploads and outputs are written under `workspace/` by default. Set `RK_CV_WORKSPACE` to change this location.

### Replacing the Model

1. Place an RK3588-compatible YOLOv7 `.rknn` model in `model/`.
2. Ensure its input size and output tensors match the YOLOv7 preprocessing and post-processing implemented in `web_detection.py`.
3. Provide a matching anchor file and pass it with `--anchors model/<anchor_file>.txt`.
4. Update the class configuration if the model does not use the default COCO 80 classes.
5. Pass the new model with `--model_path model/<model_name>.rknn`.
