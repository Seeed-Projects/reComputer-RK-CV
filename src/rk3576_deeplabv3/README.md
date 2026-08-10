# RK3576 DeepLabV3 Deployment Guide

[English] | [中文](./README_zh.md)

This project packages DeepLabV3 semantic segmentation for RK3576 with RKNN NPU inference, a browser UI, REST APIs, MP4 processing, and Docker deployment.

## Important Device Mapping

On the supported reComputer boards, `/dev/dri/renderD129` is the RKNPU device; `renderD128` is the GPU. RKNN Toolkit Lite2 also checks the board-compatible file. Both paths must be mounted with `-v`, and `--privileged` is recommended for the runtime check.

## Quick Start

Run the published image:

```bash
sudo docker run --rm --name rk3576-deeplabv3 \
  --privileged \
  -p 8000:8000 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-deeplabv3:latest \
  python3 web_service.py \
    --platform rk3576 \
    --model_path /app/model/deeplabv3.rknn \
    --sample_path /app/model/test.jpg \
    --overlay_alpha 0.5 \
    --host 0.0.0.0 \
    --port 8000
```

Open `http://<BOARD_IP>:8000`. OpenAPI is at `http://<BOARD_IP>:8000/docs`.

If host port 8000 is occupied, only change the left side of `-p`:

```bash
-p 8001:8000
```

The service must still listen on container port 8000 so that its health check remains correct.

### Startup Parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | Target NPU: `rk3576` for this image. |
| `--model_path` | No | `model/deeplabv3.rknn` | DeepLabV3 RKNN file. |
| `--sample_path` | No | `model/test.jpg` | Image used for startup warm-up and initial Web preview. |
| `--overlay_alpha` | No | `0.5` | Initial color-mask opacity, from `0` to `1`. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. Keep `0.0.0.0` for external access. |
| `--port` | No | `8000` | Port inside the container. |

Build locally from the repository root:

```bash
docker build -f docker/rk3576/deeplabv3.dockerfile \
  -t rk3576-deeplabv3:local src/rk3576_deeplabv3
```

## API Documentation

### Predict

**Endpoint:** `POST /api/models/deeplabv3/predict`
**Content type:** `multipart/form-data`

| Field | Required | Description |
| --- | --- | --- |
| `file` | Yes | JPEG/PNG or another OpenCV-decodable image. |
| `overlay_alpha` | No | Per-request mask opacity, `0`–`1`; overrides the global configuration for this request only. |

```bash
curl -X POST http://127.0.0.1:8000/api/models/deeplabv3/predict \
  -F "file=@model/test.jpg" \
  -F "overlay_alpha=0.65"
```

The response reports inference time, image size, opacity, and all PASCAL VOC classes present with their pixel counts. The rendered overlay is available through `GET /api/video_feed`.

### Configuration

```bash
curl http://127.0.0.1:8000/api/config

curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"overlay_alpha":0.7}'
```

### Other Endpoints

- `GET /api/health`: model, platform, and readiness.
- `GET /api/video_feed`: latest segmentation overlay as MJPEG.
- `POST /api/video/upload`: upload an MP4 as multipart `file`.
- `POST /api/video/analyze`: start asynchronous processing with multipart `filename`.
- `GET /api/video/status`, `GET /api/video/list`, `GET /api/video/download/{filename}`: progress and result management.

## Processing and Model Replacement

Input is resized to 513×513 and converted from BGR to RGB. RKNN output may be NHWC or NCHW; the service validates a 21-class output, restores logits to the source resolution, applies `argmax`, and overlays the PASCAL VOC color map.

To replace the model, mount a compatible model and point `--model_path` to it. Models with a different class count require corresponding changes to `LABELS` and output processing in `task_runtime.py`.
