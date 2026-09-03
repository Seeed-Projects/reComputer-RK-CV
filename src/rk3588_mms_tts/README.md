# RK3588 MMS-TTS Deployment Guide

[English] | [中文](./README_zh.md)

English speech synthesis using `facebook/mms-tts-eng` encoder and decoder models converted for the RK3588 NPU.

## Features

- English text entry with immediate validation
- Browser WAV playback and download
- REST API and CLI output file
- Generated duration, sample rate, and sample count

## Directory Structure

- `model/`: MMS-TTS encoder and decoder RKNN models
- `task_runtime.py`: vocabulary, duration alignment, inference, and WAV encoding
- `web_service.py`: FastAPI service and generated-audio endpoint
- `inference.py`: one-shot CLI
- `web/`: English browser interface

## Quick Start

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-mms_tts:latest
```

Open `http://<Board_IP>:8000`.

### Command Line

```bash
python inference.py --platform rk3588 --model_dir model \
    --text 'hello from the edge' --output output.wav
```

## API

```bash
curl -X POST http://127.0.0.1:8000/api/models/mms_tts/predict \
    -F 'text=hello from the edge'
curl -o speech.wav http://127.0.0.1:8000/api/audio/<audio_filename>
```

The prediction response contains `audio_url` and `audio_filename`; generated files are retained temporarily and the newest 20 are kept. Input is English, at most 99 supported characters. Unsupported characters are rejected rather than silently dropped. OpenAPI documentation is at `/docs`.
