# reComputer-RK-CV

[English] | [中文](./README_zh.md)

This project provides deployable Computer Vision (CV) applications for Rockchip
development boards. It supports **RK3588** and **RK3576**, with object detection,
image classification, pose estimation, segmentation, and speech examples.

## Project Architecture

The project uses a multi-platform adaptation architecture, with code and environment configurations managed independently for each platform:

```text
reComputer-RK-CV/
├── docker/                 # Docker image configuration files
│   ├── rk3576/             # RK3576 specific Dockerfile
│   └── rk3588/             # RK3588 specific Dockerfile
├── src/                    # Source code directory
│   ├── rk3576_<model>/     # Independent RK3576 application build contexts
│   └── rk3588_<model>/     # Independent RK3588 application build contexts
└── .github/workflows/      # GitHub Actions automated build scripts
```

## Supported Platforms

| Platform | Chip | Computing Power | Image Name |
| :--- | :--- | :--- | :--- |
| **RK3588** | RK3588/RK3588S | 6 TOPS | `rk3588-yolo` |
| **RK3576** | RK3576 | 6 TOPS | `rk3576-yolo` |

Image-classification services:

| Platform | Model | Web/API entry point | Dockerfile |
| :--- | :--- | :--- | :--- |
| RK3588 | MobileNet | `POST /api/models/mobilenet/predict` | `docker/rk3588/mobilenet.dockerfile` |
| RK3588 | ResNet50V2 | `POST /api/models/resnet50v2/predict` | `docker/rk3588/resnet50v2.dockerfile` |
| RK3576 | MobileNet | `POST /api/models/mobilenet/predict` | `docker/rk3576/mobilenet.dockerfile` |
| RK3576 | ResNet50V2 | `POST /api/models/resnet50v2/predict` | `docker/rk3576/resnet50v2.dockerfile` |

New standard services are available for both RK3588 and RK3576:

| Category | Models |
| :--- | :--- |
| Segmentation | DeepLabV3, PPSeg, MobileSAM |
| Vision/OCR | RetinaFace, LPRNet, PPOCR (Det + Rec), CLIP |
| Text | Lite Transformer, MMS-TTS |
| Speech/audio | Wav2Vec2, Whisper, Zipformer, YAMNet |

Every service provides a browser page, `GET /api/health`, OpenAPI at `/docs`,
and `POST /api/models/<model>/predict`.

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
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolov8_airborne:latest
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

- [RK3588 MobileNet](src/rk3588_mobilenet/README.md)
- [RK3588 ResNet50V2](src/rk3588_resnet50v2/README.md)
- [RK3576 MobileNet](src/rk3576_mobilenet/README.md)
- [RK3576 ResNet50V2](src/rk3576_resnet50v2/README.md)
- [RK3588 PPOCR (Det + Rec)](src/rk3588_ppocr/README.md)
- [RK3576 PPOCR (Det + Rec)](src/rk3576_ppocr/README.md)
- [RK3588 YOLOv6](src/rk3588_yolov6/README.md)
- [RK3576 YOLOv6](src/rk3576_yolov6/README.md)
- [RK3588 YOLOv7](src/rk3588_yolov7/README.md)
- [RK3576 YOLOv7](src/rk3576_yolov7/README.md)
- [RK3588 YOLOv10](src/rk3588_yolov10/README.md)
- [RK3576 YOLOv10](src/rk3576_yolov10/README.md)
- [RK3588 YOLOX](src/rk3588_yolox/README.md)
- [RK3576 YOLOX](src/rk3576_yolox/README.md)
- [RK3588 PP-YOLOE](src/rk3588_ppyoloe/README.md)
- [RK3576 PP-YOLOE](src/rk3576_ppyoloe/README.md)
- [RK3588 YOLO-World](src/rk3588_yolo_world/README.md)
- [RK3576 YOLO-World](src/rk3576_yolo_world/README.md)
- [RK3576 standard segmentation, OCR, text and audio services](src/rk3576_deeplabv3/README.md)
- [RK3588 standard segmentation, OCR, text and audio services](src/rk3588_deeplabv3/README.md)
- [Python model project migration standard](src/PYTHON_PROJECT_STANDARD.md)

## Automated Build

This project supports automated multi-platform image building via GitHub Actions.
- Each `src/rk<platform>_<model>/` directory is an independent Docker build context.
- All standard services listed above are included in both RK3588 and RK3576 build matrices.
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

## RK182x PCIe accelerator · RKNN3 CV

These **11 standalone Docker projects and 26 variant configurations** have been validated on an RK3588 + RK1820 M.2 using RKNN3 1.1.0 and INT8 W8A8. Each bilingual project README includes measured FPS for every variant, methodology and raw JSON evidence.

**Validated on RK3588 + RK1820 M.2.** GitHub Actions builds all 11 ARM64 images and publishes `latest` and commit tags to GHCR. Check Actions for publication status. Each project guide includes GHCR deployment, local builds and measured benchmarks.

| Project | Variants | Image | Guide |
|---|---|---|---|
| Depth Anything V3 Base | `da3-base` | `rv182x-rk3588-depth-anything-v3` | [README](src/rv182x_rk3588_depth_anything_v3/README.md) |
| MobileNetV2 | `mobilenetv2-12` | `rv182x-rk3588-mobilenetv2` | [README](src/rv182x_rk3588_mobilenetv2/README.md) |
| QA-CLIP ViT-L/14 | `qaclip-vit-l14` | `rv182x-rk3588-qaclip` | [README](src/rv182x_rk3588_qaclip/README.md) |
| RAFT-Stereo | `raftstereo-eth3d-160x240`, `raftstereo-middlebury-160x240`, `raftstereo-sceneflow-160x240`, `raftstereo-realtime-160x240` | `rv182x-rk3588-raft-stereo` | [README](src/rv182x_rk3588_raft_stereo/README.md) |
| ResNet50-v2 | `resnet50-v2-7` | `rv182x-rk3588-resnet` | [README](src/rv182x_rk3588_resnet/README.md) |
| YOLO26 | `yolo26n`, `yolo26s`, `yolo26m` | `rv182x-rk3588-yolo26` | [README](src/rv182x_rk3588_yolo26/README.md) |
| YOLO26 Pose | `yolo26n-pose`, `yolo26s-pose`, `yolo26m-pose` | `rv182x-rk3588-yolo26-pose` | [README](src/rv182x_rk3588_yolo26_pose/README.md) |
| YOLO26 Segment | `yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg` | `rv182x-rk3588-yolo26-segment` | [README](src/rv182x_rk3588_yolo26_segment/README.md) |
| YOLOv5 | `yolov5n`, `yolov5s`, `yolov5m` | `rv182x-rk3588-yolov5` | [README](src/rv182x_rk3588_yolov5/README.md) |
| YOLOv6 | `yolov6n`, `yolov6s`, `yolov6m` | `rv182x-rk3588-yolov6` | [README](src/rv182x_rk3588_yolov6/README.md) |
| YOLOv8 | `yolov8n`, `yolov8s`, `yolov8m` | `rv182x-rk3588-yolov8` | [README](src/rv182x_rk3588_yolov8/README.md) |

Prepare the matching PCIe driver, firmware and RKNN3 proxy and confirm the card is Online. Normal startup does not flash firmware. YOLO26 example:

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolo26:latest
sudo docker run --rm --privileged --net=host --ipc=host \
  -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
  -v "$PWD/workspace:/app/workspace" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolo26:latest \
  python3 web_detection.py --variant yolo26n --model_path model/yolo26n.rknn --video video/test.mp4
```

Open http://<Board_IP>:8000; API documentation is /docs. Select a project variant using MODEL_VARIANT. Demonstrations share port 8000; stop the current one before switching. The supplied 30_1080p.mp4 is used throughout, while RAFT uses a genuine official stereo pair. Clear close-up people are recommended for pose-video evaluation; road stereo video requires synchronized, rectified left/right material.

QAClip caches text vectors and releases the text encoder before vision inference; its FPS measures frame processing with cached text. Depth Anything V3 uses one 280×280 view and three serial partitions. RAFT exports one iteration and counts stereo pairs. Each project separates synchronous runtime-call FPS from pipeline FPS including decode, task processing and preview encoding.

The dedicated workflow `.github/workflows/docker-build-rv182x.yml` builds an 11-model matrix on push and supports selecting one model with workflow_dispatch.
