# RK3588 PP-YOLOE Deployment Guide

[English] | [中文](./README_zh.md)

This directory provides a standardized PP-YOLOE deployment for RK3588: RKNN NPU inference, browser preview, REST API, and one-command Docker startup.

## Core Features

- **NPU acceleration**: Runs the PP-YOLOE RKNN model on the RK3588 NPU through RKNN Toolkit Lite2.
- **Multiple inputs**: Supports images, MP4 frames, the bundled sample video, and a camera.
- **Web service**: Includes browser preview, MJPEG streaming, health checks, threshold configuration, and asynchronous video analysis.
- **Standard output**: Returns class names, confidence scores, bounding boxes, and image dimensions as JSON.
- **Two model sizes**: Uses `ppyoloe_s.rknn` by default and also includes `ppyoloe_m.rknn`.

## Directory Structure

- `lib/`: RK3588 RKNN runtime library.
- `model/ppyoloe_s.rknn`: Default PP-YOLOE RKNN model.
- `model/ppyoloe_m.rknn`: Optional larger model.
- `model/coco_80_labels_list.txt`: COCO class names.
- `py_utils/`: RKNN executor and detection utilities.
- `video/test.mp4`: Default sample input.
- `web_detection.py`: Inference, post-processing, Web preview, and REST API service.
- `requirements.txt` and `rknn-toolkit-lite2-packages/`: Runtime dependencies.

## Quick Start

### Run the published image

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-ppyoloe:latest
```

Open `http://<BOARD_IP>:8000`. OpenAPI is available at `http://<BOARD_IP>:8000/docs`.

For a camera, add `--device /dev/video0:/dev/video0` and override the command:

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-ppyoloe:latest \
  python web_detection.py --platform rk3588 --model_path model/ppyoloe_s.rknn \
  --class_path model/coco_80_labels_list.txt --camera_id 0
```

### Build locally

```bash
docker build -f docker/rk3588/ppyoloe.dockerfile \
  -t rk3588-ppyoloe:local src/rk3588_ppyoloe
```

## API Documentation

### Model prediction

**Endpoint:** `POST /api/models/ppyoloe/predict`
**Content type:** `multipart/form-data`

- `file`: Optional image.
- `video`: Optional MP4; one frame is read.
- `timestamp`: Optional video timestamp in seconds.
- `realtime`: Optional camera-frame mode.
- `conf`: Optional confidence threshold; default `0.25`.
- `iou`: Optional class-aware NMS IoU threshold; default `0.45`.

Input priority is image, video, then the current camera/sample-video frame.

```bash
curl -X POST http://127.0.0.1:8000/api/models/ppyoloe/predict \
  -F "file=@model/bus.jpg" -F "conf=0.30" -F "iou=0.45"
```

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {"class": "bus", "confidence": 0.91,
     "box": {"x1": 120, "y1": 80, "x2": 520, "y2": 430}}
  ],
  "image": {"width": 640, "height": 480}
}
```

### Configuration and video endpoints

- `GET /api/health`: Runtime, platform, model readiness.
- `GET /api/config`: Returns `obj_thresh` and `nms_thresh`.
- `POST /api/config`: Updates thresholds with JSON, for example `{"obj_thresh": 0.3, "nms_thresh": 0.5}`.
- `GET /api/video_feed`: MJPEG preview stream.
- `POST /api/video/upload` and `POST /api/video/analyze`: Upload an MP4 and start analysis using a multipart `filename` field.
- `GET /api/video/status`, `GET /api/video/list`, `GET /api/video/download/{filename}`: Query and retrieve results.

## Developer Guide

`web_detection.py` performs 640×640 letterbox preprocessing, RKNN inference, PP-YOLOE DFL decoding, class-aware NMS, coordinate restoration, rendering, and API serialization.

To use another converted PP-YOLOE model, place it in `model/` and pass `--model_path`. Its output layout must remain compatible with the included DFL decoder.
