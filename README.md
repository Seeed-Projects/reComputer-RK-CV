# reComputer-RK-CV

[English] | [中文](./README_zh.md)

This project aims to provide industrial-grade, high-performance Computer Vision (CV) application solutions for Rockchip series development boards. It currently supports **RK3588** and **RK3576** platforms, primarily integrating the YOLOv11 object detection model.

## Project Architecture

The project uses a multi-platform adaptation architecture, with code and environment configurations managed independently for each platform:

```text
reComputer-RK-CV/
├── docker/                 # Docker image configuration files
│   ├── rk3576/             # RK3576 specific Dockerfile
│   └── rk3588/             # RK3588 specific Dockerfile
├── src/                    # Source code directory
│   ├── rk3576/             # RK3576 source code, models, and dependencies
│   └── rk3588/             # RK3588 source code, models, and dependencies
└── .github/workflows/      # GitHub Actions automated build scripts
```

## Supported Platforms

| Platform | Chip | Computing Power | Image Name |
| :--- | :--- | :--- | :--- |
| **RK3588** | RK3588/RK3588S | 6 TOPS | `rk3588-yolo` |
| **RK3576** | RK3576 | 6 TOPS | `rk3576-yolo` |

## Quick Start

### 1. Install Docker

Run the following commands on the development board to install Docker:

```bash
# Download installation script
curl -fsSL https://get.docker.com -o get-docker.sh
# Install using Aliyun mirror source
sudo sh get-docker.sh --mirror Aliyun
# Start Docker and enable auto-start on boot
sudo systemctl enable docker
sudo systemctl start docker
```

### 2. Run the Project (One command, dual-mode preview)

This project supports simultaneous preview via **Local GUI** and **Web Browser**. The program automatically detects the display environment and downgrades to Web mode if no display is connected.

#### Step A: Configure Display Permissions (Optional)
If you have a monitor connected and want to see the window locally:
```bash
xhost +local:docker
```

#### Step B: Pull Images
```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolo11:latest
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolo11:latest
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-obb:latest
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolov8-obb:latest
```

#### Step C: Run with One Click

**For RK3588:**
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video1:/dev/video1 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolo11:latest \  
    python web_detection.py --model_path model/yolo11n.rknn --video video/test.mp4
```

**For RK3576:**
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolo11:latest \  
    python web_detection.py --model_path model/yolo11n.rknn --video video/test.mp4
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
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolo11:latest \  
    python web_detection.py --model_path model/yolo11n.rknn --video video/test.mp4 --class_path class_config.txt
```

---

## 🔌 API Documentation

This project provides RESTful interfaces compatible with the Ultralytics Cloud API standard, supporting object detection via image uploads using HTTP POST requests.

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

### 3. Command Line Arguments

`web_detection.py` supports the following arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model_path` | Path to RKNN model file | (Required) |
| `--camera_id` | Camera device ID (e.g., fill 1 for `/dev/video1`) | 1 |
| `--video_path` | Path to video file (overrides camera_id if provided) | None |
| `--class_path` | Path to custom class configuration file (class_config.txt) | None (Default COCO 80) |
| `--host` | Web server listening address | `0.0.0.0` |
| `--port` | Web server port | 8000 |

#### Custom Class Configuration (class_config.txt) Format:
Name classes with double quotes, separated by commas, for example:
`"person", "bicycle", "car", "motorbike"`

---

## Real-time Video Stream Interface (Video Feed)

Get real-time MJPEG video stream with detection boxes drawn, can be directly embedded in HTML `<img>` tags.

- **Endpoint:** `GET /api/video_feed`
- **Example Usage:** `<img src="http://<Board_IP>:8000/api/video_feed">`

## Detailed Platform Documentation

- [RK3588 Guide](src/rk3588/README.md)
- [RK3576 Guide](src/rk3576/README.md)

## Automated Build

This project supports automated multi-platform image building via GitHub Actions.
- Modifying the `src/rk3588/` directory automatically triggers the `rk3588-yolo` image build.
- Modifying the `src/rk3576/` directory automatically triggers the `rk3576-yolo` image build.
- Manual trigger is supported, with the option to specify `image_tag`.

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
1. Place the trained and converted .rknn model into the `model/` directory of the corresponding platform.
2. Add the `--model_path` argument to the running command to point to the new model (default already configured in Dockerfile).
