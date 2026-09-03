# RK3588 Zipformer Deployment Guide

[English] | [中文](./README_zh.md)

Chinese-English streaming speech recognition with Zipformer encoder, decoder, and joiner RKNN models on RK3588.

## Features

- Upload an audio file for complete transcription
- Stream browser microphone PCM over WebSocket and receive partial transcripts
- Token IDs, token timestamps, and processed duration
- REST API, WebSocket API, and one-shot CLI

## Directory Structure

- `model/`: encoder, decoder, joiner, and bilingual vocabulary
- `samples/test.wav`: sample audio
- `task_runtime.py`: Kaldi-compatible filterbank and streaming state cache
- `web_service.py`, `inference.py`, `web/`: API, CLI, and English UI

## Quick Start

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-zipformer:latest
```

Open `http://<Board_IP>:8000`. Browser microphone capture does not require `/dev/snd` mapping.

### Command Line

```bash
python inference.py --platform rk3588 --model_dir model --file samples/test.wav
```

## APIs

```bash
curl -X POST http://127.0.0.1:8000/api/models/zipformer/predict \
    -F 'file=@samples/test.wav'
```

- REST: `POST /api/models/zipformer/predict`, multipart field `file`.
- Streaming: `WS /api/models/zipformer/stream`; send mono 16 kHz signed 16-bit little-endian PCM binary chunks, then `{"action":"stop"}`.
- Server messages are `ready`, `partial`, `final`, or `error` JSON objects.

File audio is decoded and resampled to 16 kHz. OpenAPI documents the REST routes at `/docs`; WebSocket framing is described above.
