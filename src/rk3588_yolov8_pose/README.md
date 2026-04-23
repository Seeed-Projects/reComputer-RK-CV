# RK3588 YOLOv8-Pose Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains YOLOv8-Pose inference code optimized for RK3588.

## Core Features
- **Hardware Acceleration**: Full utilization of RK3588's 6 TOPS NPU computing power.
- **Pose Estimation**: Detects 17 keypoints for human pose estimation.
- **High-performance Inference**: Real-time FPS display based on inference time calculation.
- **Web UI & API**: Real-time video stream preview, local video analysis, and RESTful API support.

## Directory Structure
- `lib/`: Contains `librknnrt.so` for RK3588.
- `model/`: Stores `.rknn` models converted for RK3588 (e.g., `yolov8n_pose.rknn`).
- `py_utils/`: Contains post-processing utilities for pose estimation.
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
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-pose:latest \
    python3 web_detection.py --model_path model/yolov8n_pose.rknn --video video/test.mp4
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
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-pose:latest \
    python3 web_detection.py --model_path model/yolov8n_pose.rknn --video video/test.mp4
```

---

## 🔌 API Documentation

This project provides RESTful interfaces compatible with the Ultralytics Cloud API standard, supporting pose estimation via image, video uploads or direct camera calls.

### 1. Model Inference Interface (Predict)

**Endpoint:** `POST /api/models/yolo11/predict` (compatible with original YOLO11 path design)

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
      "box": { "x1": 100, "y1": 200, "x2": 300, "y2": 500 },
      "keypoints": [
        { "x": 150.5, "y": 210.2, "conf": 0.88 },
        { "x": 160.1, "y": 205.5, "conf": 0.91 },
        { "x": 140.2, "y": 206.1, "conf": 0.85 },
        { "x": 170.8, "y": 215.3, "conf": 0.89 },
        { "x": 130.4, "y": 218.7, "conf": 0.82 },
        { "x": 180.2, "y": 250.4, "conf": 0.95 },
        { "x": 120.5, "y": 255.1, "conf": 0.93 },
        { "x": 190.6, "y": 300.2, "conf": 0.87 },
        { "x": 110.3, "y": 305.8, "conf": 0.86 },
        { "x": 200.1, "y": 350.5, "conf": 0.81 },
        { "x": 100.9, "y": 355.2, "conf": 0.80 },
        { "x": 175.4, "y": 400.1, "conf": 0.90 },
        { "x": 125.6, "y": 405.3, "conf": 0.88 },
        { "x": 185.2, "y": 450.6, "conf": 0.85 },
        { "x": 115.8, "y": 455.4, "conf": 0.84 },
        { "x": 195.7, "y": 500.2, "conf": 0.79 },
        { "x": 105.3, "y": 505.7, "conf": 0.78 }
      ]
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

Get real-time MJPEG video stream with pose skeletons drawn, can be directly embedded in HTML `<img>` tags.

- **Endpoint:** `GET /api/video_feed`
- **Example Usage:** `<img src="http://<Board_IP>:8000/api/video_feed">`

---

## 🛠️ Developer Guide (Production Recommendations)
### Code Description
- `web_detection.py`:
    - **Web API**: Integrates FastAPI, supporting MJPEG streaming output, file upload, and video analysis.
    - **RKNN Inference**: Encapsulates RKNN initialization, model loading, and multi-core inference logic.
    - **Mode Switching**: Dynamically handles UI layout based on input parameters (`--camera_id`, `--video_path`).
- `py_utils/pose_utils.py`:
    - **Post-processing**: YOLOv8-Pose specific Box decoding, NMS logic, and Keypoint extraction.
    - **Visualization**: Draws bounding boxes and pose skeletons with customized colors.

### Modifying Models
1. Place the trained and converted .rknn model into the `model/` directory.
2. Add the `--model_path` argument to the running command to point to the new model.
