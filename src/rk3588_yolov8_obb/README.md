# RK3588 YOLOv8-OBB Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains YOLOv8-OBB (Oriented Bounding Box) inference code optimized for RK3588.

## Core Features
- **Hardware Acceleration**: Full utilization of RK3588's 6 TOPS NPU computing power.
- **OBB Detection**: Detects objects with oriented bounding boxes, ideal for aerial or satellite imagery (e.g., DOTA dataset).
- **High-performance Inference**: Real-time FPS display based on inference time calculation.
- **Web UI & API**: Real-time video stream preview, local video analysis, and RESTful API support.

## Directory Structure
- `lib/`: Contains `librknnrt.so` for RK3588.
- `model/`: Stores `.rknn` models converted for RK3588 (e.g., `yolov8n_obb.rknn`).
- `py_utils/`: Contains post-processing utilities for OBB detection.
- `web_detection.py`: Main program (supports Web preview and API).

## Quick Start

### 1. Run the Project (One command, Web preview)

This project supports preview via **Web Browser**. The program automatically serves a web interface for real-time detection or local video analysis.

#### Step A: Configure Display Permissions (Optional)
If you have a monitor connected and want to see the window locally:
```bash
xhost +local:docker
```

#### Step B: One-click Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video1:/dev/video1 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-obb:latest \
    python3 web_detection.py --model_path model/yolov8n_obb.rknn --video video/test.mp4
```
Access via: `http://<Board_IP>:8000`

> **Note**: If you want to test with a local video instead of a camera, use the `--video` parameter:
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    -v $(pwd)/video:/app/video \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-obb:latest \
    python3 web_detection.py --model_path model/yolov8n_obb.rknn --video video/test.mp4
```

---

## 🔌 API Documentation

This project provides RESTful interfaces compatible with the Ultralytics Cloud API standard, supporting OBB detection via image, video uploads or direct camera calls.

### 1. Model Inference Interface (Predict)

**Endpoint:** `POST /api/models/yolo_obb/predict`

#### Request Parameters (Multipart/Form-Data):
- `file`: (Optional) Image file to be detected.
- `video`: (Optional) MP4 video file to be detected.
- `timestamp`: (Optional) Timestamp in the video file (seconds), returns detection results for the frame at that point. Default is 0.
- `realtime`: (Optional) Boolean. If `true` or if no `file`/`video` parameters are provided, returns detection results for the current camera frame.
- `conf`: (Optional) Confidence threshold for a single request, range 0.0-1.0.
- `iou`: (Optional) NMS IOU threshold for a single request, range 0.0-1.0.

#### Usage Examples:

**1. Image Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo_obb/predict" -F "file=@/home/cat/001.jpg"
```

**2. Video Specific Frame Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo_obb/predict" -F "video=@/home/cat/test.mp4" -F "timestamp=5.5"
```

**3. Get Current Camera Frame Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo_obb/predict" -F "realtime=true"
# Or without file parameters
curl -X POST "http://127.0.0.1:8000/api/models/yolo_obb/predict"
```

#### Response Format (JSON):
```json
{
  "success": true,
  "source": "video frame at 5.5s",
  "predictions": [
    {
      "class": "plane",
      "confidence": 0.92,
      "poly": [
        [150, 210],
        [160, 205],
        [170, 215],
        [160, 220]
      ],
      "angle": 1.5708
    }
  ],
  "image": { "width": 1280, "height": 720 }
}
```

### 2. Local Video Analysis

- **Upload Video:** `POST /api/video/upload`
- **List Videos:** `GET /api/video/list`
- **Analyze Video:** `POST /api/video/analyze`
- **Check Status:** `GET /api/video/status`
- **Download Result:** `GET /api/video/download/{filename}`

### 3. System Configuration Interface (Config)

Used to dynamically adjust thresholds for real-time video streams and default inference.

#### Get Current Configuration
- **Endpoint:** `GET /api/config`
- **Response:** `{"obj_thresh": 0.25, "nms_thresh": 0.45, "camera_id": 1, "video_path": null}`

#### Update System Configuration
- **Endpoint:** `POST /api/config`
- **Request Body (JSON):** `{"obj_thresh": 0.3, "nms_thresh": 0.5}`
- **Response:** `{"status": "success"}`

### 4. Real-time Video Stream Interface (Video Feed)

Get real-time MJPEG video stream with OBB bounding boxes drawn, can be directly embedded in HTML `<img>` tags.

- **Endpoint:** `GET /api/video_feed`
- **Example Usage:** `<img src="http://<Board_IP>:8000/api/video_feed">`

---

## 🛠️ Developer Guide (Production Recommendations)
### Code Description
- `web_detection.py`:
    - **Web API**: Integrates FastAPI, supporting MJPEG streaming output, file upload, and video analysis.
    - **RKNN Inference**: Encapsulates RKNN initialization, model loading, and multi-core inference logic.
    - **Mode Switching**: Dynamically handles UI layout based on input parameters (`--camera_id`, `--video_path`).
- `py_utils/obb_utils.py`:
    - **Post-processing**: YOLOv8-OBB specific Box decoding, NMS logic, and angle processing.
    - **Visualization**: Draws oriented bounding boxes (polygons) with customized colors.

### Modifying Models
1. Place the trained and converted .rknn model into the `model/` directory.
2. Add the `--model_path` argument to the running command to point to the new model.
