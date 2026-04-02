# RK3588 YOLOv8-OBB Deployment Guide

This directory contains YOLOv8-OBB (Oriented Bounding Box) inference code optimized for RK3588.
It is based on the structure of the RK3588 YOLO example but adapted for rotated object detection.

## Core Features
- **Hardware Acceleration**: Optimized for RK3588's NPU.
- **OBB Support**: Supports rotated bounding boxes (x, y, w, h, angle).
- **Flexible Input**: Supports camera and local MP4 video input.
- **Web Preview**: Real-time web preview via FastAPI.
- **Video Download**: Automatically records the first loop of video processing and allows downloading via Web UI for easy debugging.

## Directory Structure
- `lib/`: Should contain `librknnrt.so` for RK3588 (if needed by C++ parts, here purely Python).
- `model/`: Place your `.rknn` models here.
- `py_utils/`: Utility functions for OBB processing and NMS.
- `web_detection.py`: Main program (supports Web preview and API).

## Quick Start

### 1. Prepare Model
Place your converted YOLOv8-OBB RKNN model in the `model/` directory.

### 2. Run the Project
```bash
# Install dependencies if needed
pip install shapely fastapi uvicorn

# Run
python web_detection.py --model_path model/yolov8_obb.rknn --camera_id 0

# Run with default video loop (video/test.mp4) to enable video download feature for debugging
python web_detection.py --model_path model/yolov8_obb.rknn --camera_id -1
```

Access via: `http://<Board_IP>:8000`

### 3. API Usage

**Predict Endpoint:** `POST /api/models/yolo_obb/predict`

Returns predictions with oriented bounding boxes (polygon points).

Example Response:
```json
{
  "success": true,
  "predictions": [
    {
      "class": "plane",
      "confidence": 0.92,
      "poly": [[100, 100], [200, 100], [200, 200], [100, 200]],
      "angle": 1.57
    }
  ]
}
```

## Notes
- The default input size is 640x640. If your model uses a different size, please modify `IMG_SIZE` in `web_detection.py`.
- Ensure `shapely` is installed for NMS (`pip install shapely`).
