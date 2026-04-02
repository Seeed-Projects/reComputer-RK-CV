# RK3588 YOLO Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains YOLOv11 inference code optimized for RK3588.

## Core Features
- **Hardware Acceleration**: Full utilization of RK3588's 6 TOPS NPU computing power.
- **Multi-core Support**: `NPU_CORE_0_1_2` full-core mode enabled by default.
- **High-performance Inference**: Real-time FPS display based on inference time calculation.

## Directory Structure
- `lib/`: Contains `librknnrt.so` for RK3588.
- `model/`: Stores `.rknn` models converted for RK3588.
- `web_detection.py`: Main program (supports Web preview and API).

## Quick Start

### 1. Run the Project (One command, dual-mode preview)

This project supports simultaneous preview via **Local GUI** and **Web Browser**. The program automatically detects the display environment and downgrades to Web mode if no display is connected.

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
    ghcr.io/Seeed-Projects/recomputer-rk-cv/rk3588-yolo:latest \
    python web_detection.py --model_path model/yolo11n.rknn --camera_id 1
```
Access via: `http://<Board_IP>:8000`


> **Note**: If you need custom classes, you can add `-v $(pwd)/class_config.txt:/app/class_config.txt \` mount and `--class_path` parameter. The program defaults to COCO 80 classes.

Example:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    -v $(pwd)/class_config.txt:/app/class_config.txt \
    --device /dev/video1:/dev/video1 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/Seeed-Projects/recomputer-rk-cv/rk3588-yolo:latest \
    python web_detection.py --model_path model/yolo11n.rknn --camera_id 1 --class_path class_config.txt
```

---

## 🔌 API Documentation

This project provides RESTful interfaces compatible with the Ultralytics Cloud API standard, supporting object detection via image, video uploads or direct camera calls.

### 1. Model Inference Interface (Predict)

**Endpoint:** `POST /api/models/yolo11/predict`

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
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict" -F "file=@/home/cat/001.jpg"
```

**2. Video Specific Frame Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict" -F "video=@/home/cat/test.mp4" -F "timestamp=5.5"
```

**3. Get Current Camera Frame Detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict" -F "realtime=true"
# Or without file parameters
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict"
```

#### Response Format (JSON):
```json
{
  "success": true,
  "source": "video frame at 5.5s",
  "predictions": [
    {
      "class": "person",
      "confidence": 0.92,
      "box": { "x1": 100, "y1": 200, "x2": 300, "y2": 500 }
    }
  ],
  "image": { "width": 1280, "height": 720 }
}
```

### 2. System Configuration Interface (Config)

Used to dynamically adjust thresholds for real-time video streams and default inference.

#### Get Current Configuration
- **Endpoint:** `GET /api/config`
- **Response:** `{"obj_thresh": 0.25, "nms_thresh": 0.45}`

#### Update System Configuration
- **Endpoint:** `POST /api/config`
- **Request Body (JSON):** `{"obj_thresh": 0.3, "nms_thresh": 0.5}`
- **Response:** `{"status": "success"}`

### 3. Real-time Video Stream Interface (Video Feed)

Get real-time MJPEG video stream with detection boxes drawn, can be directly embedded in HTML `<img>` tags.

- **Endpoint:** `GET /api/video_feed`
- **Example Usage:** `<img src="http://<Board_IP>:8000/api/video_feed">`

---

## 🛠️ Developer Guide (Production Recommendations)
### Code Description
- `web_detection.py`:
    - **Dual-mode Support**: Integrates FastAPI, supporting both local rendering and MJPEG streaming output.
    - **Environment Adaptive**: Automatically detects the `DISPLAY` environment variable, silently skipping GUI initialization if not present.
    - **RKNN Inference**: Encapsulates RKNN initialization, model loading, and multi-core inference logic.
    - **Dynamic Loading**: Supports dynamic class configuration loading via `--class_path`.
    - **Post-processing**: YOLOv11 specific Box decoding and NMS logic.

### Modifying Models
1. Place the trained and converted .rknn model into the `model/` directory.
2. Add the `--model_path` argument to the running command to point to the new model.
