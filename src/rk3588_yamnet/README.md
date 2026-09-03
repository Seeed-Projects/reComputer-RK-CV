# RK3588 YAMNet Deployment Guide

[English] | [中文](./README_zh.md)

YAMNet audio event classification across 521 AudioSet classes, accelerated by the RK3588 NPU.

## Features

- Upload audio or record from the browser microphone
- Overall Top-K classes and a sliding-window event timeline
- Configurable score threshold and 0.25–3 second hop
- REST API and one-shot CLI

## Directory Structure

- `model/yamnet.rknn`: fixed three-second model
- `model/yamnet_class_map.txt`: 521 class labels
- `samples/test.wav`: sample audio
- `task_runtime.py`: audio conversion, windowing, ranking, and event merging
- `web_service.py`, `inference.py`, `web/`: API, CLI, and English UI

## Quick Start

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yamnet:latest
```

Open `http://<Board_IP>:8000`.

### Command Line

```bash
python inference.py --platform rk3588 --model_dir model --file samples/test.wav \
    --topk 5 --threshold 0.1 --hop_seconds 1
```

## API

```bash
curl -X POST http://127.0.0.1:8000/api/models/yamnet/predict \
    -F 'file=@samples/test.wav' -F 'topk=5' \
    -F 'threshold=0.1' -F 'hop_seconds=1'
```

The runtime converts audio to mono 16 kHz and analyzes fixed 3-second windows. `topk` is 1–20, `threshold` is 0–1, `hop_seconds` is 0.25–3, and input duration is limited to 300 seconds. Results include overall `predictions`, per-window `segments`, and merged `events`. OpenAPI documentation is at `/docs`.
