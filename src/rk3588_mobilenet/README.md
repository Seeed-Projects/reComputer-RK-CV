# RK3588 MobileNetV2 Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains a MobileNetV2 image classification service optimized for RK3588. It provides an RKNN-based inference pipeline, browser preview, REST APIs, and a ready-to-run Docker image.

## Core Features

- **Hardware Acceleration**: Runs the RKNN model on the RK3588 NPU through RKNN-Toolkit-Lite2.
- **ImageNet Classification**: Returns configurable Top-K class labels and confidence scores using `synset.txt`.
- **Flexible Input**: Supports image uploads, a selected frame from an MP4 video, a local video file, and camera frames.
- **Web and API Access**: Includes a browser preview, MJPEG stream, health check, and REST inference APIs.

## Directory Structure

- `lib/`: RK3588 `librknnrt.so` runtime library.
- `model/`: RK3588 RKNN model, ImageNet labels, and sample image.
- `py_utils/`: MobileNetV2 preprocessing and Top-K post-processing utilities.
- `video/`: Sample MP4 video used by the default container command.
- `web_classification.py`: Main application providing RKNN inference, Web preview, and FastAPI endpoints.
- `requirements.txt`: Python runtime dependencies.

The default model is `model/rk3588_mobilenet_v2.rknn` and its input size is `224 x 224`.

## Quick Start

### 1. Run with Docker

Run the published image directly on an RK3588 board:

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-mobilenet:latest
```

The default command classifies the bundled `video/test.mp4`. Open `http://<Board_IP>:8000` for the Web preview. Interactive API documentation is available at `http://<Board_IP>:8000/docs`.

To run in upload-only Web mode without opening a camera or local video, override the command:

```bash
sudo docker run --rm --privileged --net=host \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-mobilenet:latest \
    python3 web_classification.py \
    --model_path model/rk3588_mobilenet_v2.rknn --camera_id -1
```

### 2. Build the Image Locally

Run the following command from the repository root:

```bash
docker build \
    -f docker/rk3588/mobilenet.dockerfile \
    -t rk3588-mobilenet:local \
    src/rk3588_mobilenet
```

## API Documentation

### 1. Model Inference Interface (Predict)

**Endpoint:** `POST /api/models/mobilenet/predict`

#### Request Parameters (`multipart/form-data`)

- `file`: Optional image file.
- `video`: Optional MP4 video file. It cannot be supplied together with `file`.
- `timestamp`: Optional non-negative timestamp in seconds for selecting a video frame; defaults to the first frame.
- `conf`: Optional confidence threshold for this request, from `0.0` to `1.0`.
- `topk`: Optional number of results, from `1` to `20`; defaults to `5`.

If neither `file` nor `video` is supplied, the service classifies the current camera or local-video frame when one is available.

#### Usage Examples

Image classification:

```bash
curl -X POST "http://127.0.0.1:8000/api/models/mobilenet/predict" \
    -F "file=@model/bell.jpg" -F "topk=5" -F "conf=0.01"
```

Classify a frame at 5.5 seconds in a video:

```bash
curl -X POST "http://127.0.0.1:8000/api/models/mobilenet/predict" \
    -F "video=@video/test.mp4" -F "timestamp=5.5" -F "topk=3"
```

#### Response Format

```json
{
  "success": true,
  "model": "mobilenet",
  "source": "uploaded image",
  "predictions": [
    {
      "class": "n02123045 tabby, tabby cat",
      "confidence": 0.91
    }
  ],
  "image": {
    "width": 640,
    "height": 480
  }
}
```

### 2. System Configuration Interface (Config)

- `GET /api/config`: Returns `{"conf_thresh": 0.0}`.
- `POST /api/config`: Accepts JSON such as `{"conf_thresh": 0.1}`. The value must be from `0.0` to `1.0`.

The global threshold is used by the preview stream and by prediction requests that do not provide `conf`.

### 3. Video and File Interfaces

- `GET /api/video_feed`: Returns the annotated MJPEG preview stream.
- `POST /api/video/upload`: Uploads a JPG, JPEG, PNG, BMP, or MP4 file using the `file` form field.
- `POST /api/video/analyze`: Starts asynchronous processing using the uploaded `filename` form field.
- `GET /api/video/status`: Returns processing progress and error state.
- `GET /api/video/list`: Lists uploaded and generated files.
- `GET /api/video/download/{filename}`: Downloads a generated result.
- `GET /api/health`: Returns service, platform, and model readiness information.

## Developer Guide

### Code Description

- `web_classification.py` loads the RKNN model, initializes the RK3588 NPU runtime, handles camera or video frames, and exposes the Web/API service.
- `py_utils/mobilenet_utils.py` resizes input images to `224 x 224`, preserves the BGR channel order required by this model, applies softmax, and filters Top-K results.
- Runtime uploads and outputs are written under `workspace/` by default. Set `RK_CV_WORKSPACE` to change this location.

### Replacing the Model

1. Place an RK3588-compatible `.rknn` model in `model/`.
2. Ensure its input layout, channel order, normalization, and output format match `MobileNet_helper`.
3. Update `model/synset.txt` if the class set changes.
4. Pass the new file with `--model_path model/<model_name>.rknn`.
