# RK3576 LPRNet Deployment Guide

[English] | [中文](./README_zh.md)

This directory packages the RKNN Model Zoo LPRNet example as a standard reComputer RK-CV service for RK3576. It supports RKNN NPU recognition, scene-level plate candidate localization, live browser preview, camera and local-video input, uploaded-video analysis, REST APIs, and an ARM64 Docker image.

## Core features

- Recognizes Chinese colored plates with RKNN LPRNet and automatically routes light/international plates to RKNN PP-OCR recognition.
- Provides a browser four-point selection tool. Each corner can be dragged independently, and the selected quadrilateral is perspective-rectified before recognition.
- Locates multiple plate candidates with cascade, edge, blue/yellow/green masks, and light/red-text analysis; bright plate candidates are ranked ahead of traffic lights and tail lights.
- Draws every accepted plate box and an indexed label in the MJPEG stream; the Web panel shows every complete Unicode plate string.
- Accepts `/dev/videoN`, a looping local MP4 through `--video`, images, and Web-uploaded MP4 files.
- Returns plate text, box coordinates, selected recognizer, candidate score, rejection reason, and recognition score through the API.

## Directory structure

- `model/lprnet.rknn`: LPRNet model converted for RK3576.
- `model/ppocr_rec.rknn`: PP-OCR recognition fallback converted for RK3576.
- `model/ppocr_keys_v1.txt`: PP-OCR character dictionary.
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
sudo docker run --rm --name rk3576-lprnet \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-lprnet:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id -1 --host 0.0.0.0 --port 8000
```

Open `http://<BOARD_IP>:8000`. The interface is in English and supports image recognition and MP4 upload/analysis. Open `http://<BOARD_IP>:8000/docs` for OpenAPI.

### 2. Local camera

The camera ID `N` maps to `/dev/videoN`. This example uses `/dev/video0`:

```bash
sudo docker run --rm --name rk3576-lprnet-camera \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-lprnet:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --camera_id 0 --host 0.0.0.0 --port 8000
```

The annotated live view is available on the home page and at `GET /api/video_feed`. Use `--camera_id 1` and map `/dev/video1` when the capture device uses that node.

### 3. Local MP4 video

```bash
sudo docker run --rm --name rk3576-lprnet-video \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-lprnet:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --video /app/video/test.mp4 --host 0.0.0.0 --port 8000
```

`--video` and `--video_path` are aliases. A local video takes precedence over `--camera_id` and loops continuously for live Web preview. Every video frame must already be a cropped plate. The service processes every frame, ignores scene localization, and resizes that complete frame for recognition regardless of source frame rate. To analyze a host file, mount it read-only, for example `-v /path/input.mp4:/data/input.mp4:ro`, and pass `--video /data/input.mp4`.

### 4. Build locally

From the repository root:

```bash
docker build -f docker/rk3576/lprnet.dockerfile \
  -t rk3576-lprnet:local src/rk3576_lprnet
```

## Command-line parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | RKNN target; use `rk3576` in this directory. |
| `--model_dir` | No | `model` | Directory containing both RKNN recognition models, the PP-OCR dictionary, and `test.jpg`. |
| `--camera_id` | No | `-1` | Camera index `N` for `/dev/videoN`; `-1` enables Web-upload-only mode. |
| `--video`, `--video_path` | No | — | Local MP4 path. Overrides the camera and loops continuously. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. |
| `--port` | No | `8000` | FastAPI listen port. |

`PYTHONUNBUFFERED=1` flushes logs immediately. `RKNN_LOG_LEVEL=0` suppresses confirmed harmless RKNN static-shape initialization messages. `renderD129` is the node verified on the RK3576 test device; check `/dev/dri/` if your board exposes a different node.

## API

### Image or scene recognition

**Endpoint:** `POST /api/models/lprnet/predict`

**Content type:** `multipart/form-data`

| Field | Required | Description |
| --- | --- | --- |
| `file` | Yes | An image decodable by OpenCV. Use `@` before a local path with curl. |
| `whole_image` | No | Set `true` when the input is already a tightly cropped plate. Scene images use automatic candidate localization. |
| `plate_layout` | No | `auto` selects LPRNet for Chinese colored plates and PP-OCR for light/international plates; `chinese` or `international` forces one route. |
| `manual_quad` | No | Four original-image points as JSON: `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`. The region is perspective-rectified and recognized as one plate. |
| `manual_box` | No | JSON array `[x1,y1,x2,y2]` in original-image coordinates. It bypasses automatic localization. |
| `min_score` | No | Per-request recognition-score filter from `0` to `1`. The score is useful for relative filtering, not a calibrated probability. |
| `max_plates` | No | Maximum candidate plates, from `1` to `32`. |
| `min_text_length` | No | Minimum accepted character count, from `1` to `8`; defaults to `6` to reject short false positives. |

```bash
curl -X POST http://127.0.0.1:8000/api/models/lprnet/predict \
  -F "file=@vehicle.jpg" \
  -F 'manual_quad=[[120,210],[410,205],[418,286],[112,292]]' \
  -F "plate_layout=chinese"
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
        "recognizer": "lprnet",
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
  -d '{"min_score":0.0,"max_plates":8,"min_text_length":6,"plate_layout":"auto"}'
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

Neither bundled model is an end-to-end plate detector. The official LPRNet recognition model is trained for cropped Chinese plates; the PP-OCR fallback improves alphanumeric text on light/international plates but is general text recognition rather than a plate-specific model. For uploaded images, use the Web four-point tool or `manual_quad` to provide an exact perspective-corrected region. Videos are intentionally treated as sequences of already-cropped plate images. Camera scene boxes still come from lightweight OpenCV heuristics and can be unreliable; production camera deployments should use a dedicated plate detector.

Model conversion inputs and scripts remain in `rknn_model_zoo/examples/LPRNet`.
