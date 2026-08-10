# RK3576 YOLO-World Deployment Guide

[English] | [中文](./README_zh.md)

This project provides YOLO-World open-vocabulary detection on RK3576. It runs both the YOLO-World detector and CLIP text encoder on the NPU, so Web and API users can change detection categories with natural-language prompts without rebuilding the model.

> The bundled CLIP model is trained primarily for English. English noun phrases such as `person`, `red bus`, or `traffic light` are recommended.

## Core Features

- Dynamic natural-language prompts through the Web UI, startup arguments, or REST API.
- Up to 80 prompt classes, separated by `|`.
- Offline BPE tokenization through the bundled `clip_vocab.txt`; no Hugging Face download is needed at runtime.
- Image, MP4-frame, sample-video, and camera inference.
- MJPEG preview, threshold configuration, and asynchronous video analysis.

## Quick Start

`/dev/dri/renderD129` is the NPU device on the supported reComputer boards. Run:

```bash
sudo docker run --rm --name rk3576-yolo_world \
  --privileged \
  -p 8000:8000 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolo_world:latest \
  python3 web_detection.py \
    --platform rk3576 \
    --model_path /app/model/yolo_world_v2s_i8.rknn \
    --text_model /app/model/clip_text_fp16.rknn \
    --text_features /app/model/coco_text_outp.npy \
    --vocab_path /app/model/clip_vocab.txt \
    --class_path /app/model/detect_classes.txt \
    --video_path /app/video/test.mp4 \
    --host 0.0.0.0 \
    --port 8000
```

Open `http://<BOARD_IP>:8000`. Enter prompts such as `person|bus|red car`, then apply them to the stream or upload an image for one-off search.

Use `-p 8001:8000` if host port 8000 is occupied. For a camera, replace `--video_path` with `--camera_id 0` and add `--device /dev/video0:/dev/video0`.

### Startup Parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | Target NPU: `rk3576`. |
| `--model_path` | Yes | — | Quantized YOLO-World RKNN detector. |
| `--text_model` | No | `model/clip_text_fp16.rknn` | FP16 CLIP text encoder used for dynamic prompts. |
| `--text_features` | No | `model/coco_text_outp.npy` | Precomputed `(1,80,512)` COCO features used before prompts are changed. |
| `--vocab_path` | No | `model/clip_vocab.txt` | Offline CLIP BPE merge vocabulary. |
| `--class_path` | No | built-in COCO labels | Labels matching the precomputed feature rows. |
| `--prompts` | No | COCO 80 classes | Initial dynamic prompts, for example `"person|red bus|bicycle"`. |
| `--video_path` | No | — | Looping local MP4 input; takes precedence over the camera. |
| `--camera_id` | No | `1` | Camera index; `-1` enables video-analysis-only mode. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. |
| `--port` | No | `8000` | Container service port. |

Build locally:

```bash
docker build -f docker/rk3576/yolo_world.dockerfile \
  -t rk3576-yolo_world:local src/rk3576_yolo_world
```

## Natural-Language Prompt API

### Read or replace active stream prompts

```bash
curl http://127.0.0.1:8000/api/prompts

curl -X POST http://127.0.0.1:8000/api/prompts \
  -H "Content-Type: application/json" \
  -d '{"text":"person|red bus|traffic light"}'
```

`POST /api/prompts` encodes every phrase with the CLIP RKNN model and changes the categories used by the live stream and asynchronous video analysis. A JSON list can also be sent as `{"prompts":["person","red bus"]}`.

### One-off image search

**Endpoint:** `POST /api/models/yolo_world/predict`
**Content type:** `multipart/form-data`

- `file`: image upload.
- `video` and `timestamp`: MP4 upload and frame time in seconds.
- `realtime`: use the current stream frame.
- `text`: optional pipe-separated prompts for this request only; it does not replace stream prompts.
- `conf`: confidence threshold, default `0.25`.
- `iou`: class-aware NMS IoU threshold, default `0.45`.

```bash
curl -X POST http://127.0.0.1:8000/api/models/yolo_world/predict \
  -F "file=@model/bus.jpg" \
  -F "text=person|bus|red vehicle" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

The response includes the prompts actually used and returns each matching prompt phrase as the detection class.

## Prompt Rules and Processing

- At least 1 and at most 80 non-empty prompts are accepted.
- Use `|` as the separator because commas may be part of a natural-language phrase.
- Each CLIP prompt is truncated to a 20-token input; short noun phrases work best.
- Dynamic embeddings fill the first rows of the detector's fixed 80-row text input. Post-processing ignores padded rows, preventing false detections from unused classes.
- Repeated prompt sets are cached in memory.

Other endpoints: `GET /api/health`, `GET/POST /api/config`, `GET /api/video_feed`, and the `/api/video/*` upload, analysis, status, list, and download interfaces.
