# RK3588 LPRNet Deployment Guide

[English] | [中文](./README_zh.md)

This directory packages the RKNN Model Zoo LPRNet example as a standard reComputer RK-CV service for RK3588. It supports RKNN NPU recognition, scene-level plate candidate localization, live browser preview, camera and local-video input, uploaded-video analysis, REST APIs, and an ARM64 Docker image.

## Core features

- Recognizes cropped Chinese license plates with the RKNN LPRNet model.
- Locates multiple plate candidates in a scene with OpenCV cascade, edge, and color analysis before recognition.
- Draws every accepted plate box and an indexed label in the MJPEG stream; the Web panel shows every complete Unicode plate string.
- Accepts `/dev/videoN`, a looping local MP4 through `--video`, images, and Web-uploaded MP4 files.
- Returns plate text, box coordinates, candidate score, and recognition score through the API.

## Directory structure

- `model/lprnet.rknn`: LPRNet model converted for RK3588.
- `model/test.jpg`: cropped license plate used for startup warm-up.
- `video/test.mp4`: sample scene video.
- `task_runtime.py`: candidate localization, RKNN recognition, and annotation.
- `web_service.py`: FastAPI service, Web UI, MJPEG stream, camera, and video processing.
- `rknn_runtime.py`: thread-safe RKNNLite wrapper.
- `py_utils/`: official character dictionary and CTC-style decoder.

## Quick start

### 1. Web upload mode

This mode starts the Web/API service without opening a camera:

```bash
sudo docker run --rm --name rk3588-lprnet \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-lprnet:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --camera_id -1 --host 0.0.0.0 --port 8000
```

Open `http://<BOARD_IP>:8000`. The interface is in English and supports image recognition and MP4 upload/analysis. Open `http://<BOARD_IP>:8000/docs` for OpenAPI.

### 2. Local camera

The camera ID `N` maps to `/dev/videoN`. This example uses `/dev/video0`:

```bash
sudo docker run --rm --name rk3588-lprnet-camera \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-lprnet:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

The annotated live view is available on the home page and at `GET /api/video_feed`. Use `--camera_id 1` and map `/dev/video1` when the capture device uses that node.

### 3. Local MP4 video

```bash
sudo docker run --rm --name rk3588-lprnet-video \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-lprnet:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --video /app/video/test.mp4 --host 0.0.0.0 --port 8000
```

`--video` and `--video_path` are aliases. A local video takes precedence over `--camera_id` and loops continuously for live Web preview. To analyze a host file, mount it read-only, for example `-v /path/input.mp4:/data/input.mp4:ro`, and pass `--video /data/input.mp4`.

### 4. Build locally

From the repository root:

```bash
docker build -f docker/rk3588/lprnet.dockerfile \
  -t rk3588-lprnet:local src/rk3588_lprnet
```

## Command-line parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | RKNN target; use `rk3588` in this directory. |
| `--model_dir` | No | `model` | Directory containing `lprnet.rknn` and `test.jpg`. |
| `--camera_id` | No | `-1` | Camera index `N` for `/dev/videoN`; `-1` enables Web-upload-only mode. |
| `--video`, `--video_path` | No | — | Local MP4 path. Overrides the camera and loops continuously. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. |
| `--port` | No | `8000` | FastAPI listen port. |

`PYTHONUNBUFFERED=1` flushes logs immediately. `RKNN_LOG_LEVEL=0` suppresses confirmed harmless RKNN static-shape initialization messages. `renderD128` is the node verified on the RK3588 test device; check `/dev/dri/` if your board exposes a different node.

## API

### Image or scene recognition

**Endpoint:** `POST /api/models/lprnet/predict`

**Content type:** `multipart/form-data`

| Field | Required | Description |
| --- | --- | --- |
| `file` | Yes | An image decodable by OpenCV. Use `@` before a local path with curl. |
| `whole_image` | No | Set `true` when the input is already a tightly cropped plate. Scene images use automatic candidate localization. |
| `min_score` | No | Per-request recognition-score filter from `0` to `1`. The score is useful for relative filtering, not a calibrated probability. |
| `max_plates` | No | Maximum candidate plates, from `1` to `32`. |
| `min_text_length` | No | Minimum accepted character count, from `1` to `8`; defaults to `6` to reject short false positives. |

```bash
curl -X POST http://127.0.0.1:8000/api/models/lprnet/predict \
  -F "file=@model/test.jpg" \
  -F "whole_image=true"
```

Example response:

```json
{
  "success": true,
  "model": "lprnet",
  "result": {
    "plates": [
      {
        "text": "京A12345",
        "recognition_score": 0.91,
        "candidate_score": 1.0,
        "box": [0, 0, 94, 24]
      }
    ],
    "count": 1,
    "source_mode": "plate_crop",
    "image": {"width": 94, "height": 24}
  }
}
```

### Runtime configuration

```bash
curl http://127.0.0.1:8000/api/config
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"min_score":0.0,"max_plates":8,"min_text_length":6}'
```

The configuration applies to camera frames, looping local videos, and uploaded-video processing.

### Video and service endpoints

- `GET /api/health`: model, platform, and current input-source status.
- `GET /api/results/latest`: all plates, boxes, and recognition text from the latest processed frame.
- `GET /api/video_feed`: annotated MJPEG stream.
- `POST /api/video/upload`: upload one `.mp4` through multipart field `file`.
- `POST /api/video/analyze`: start analysis through multipart field `filename`.
- `GET /api/video/status`: progress, output filename, and errors.
- `GET /api/video/list`: uploaded and generated MP4 files.
- `GET /api/video/download/{filename}`: download an analyzed MP4.

```bash
curl -X POST http://127.0.0.1:8000/api/video/upload \
  -F "file=@video/test.mp4"
curl -X POST http://127.0.0.1:8000/api/video/analyze \
  -F "filename=test.mp4"
curl http://127.0.0.1:8000/api/video/status
```

## Model scope and limitations

The packaged RKNN model is the official LPRNet recognition model trained for cropped Chinese license plates. It is not an end-to-end plate detector. Scene-level boxes are generated by lightweight OpenCV heuristics and then recognized by LPRNet; small, blurred, strongly tilted, occluded, or non-Chinese plates can be missed or misread. For production use, replace the heuristic locator with a dedicated plate detector and pass its crops to this LPRNet runtime.

Model conversion inputs and scripts remain in `rknn_model_zoo/examples/LPRNet`.
