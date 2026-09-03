# RK3588 Wav2Vec2 Deployment Guide

[English] | [中文](./README_zh.md)

English automatic speech recognition using `facebook/wav2vec2-base-960h` converted for the RK3588 NPU.

## Features

- Upload WAV or another libsndfile-supported audio file
- Record from the browser microphone without mapping a host audio device
- Audio player, waveform preview, transcript, and timing information
- REST API and one-shot CLI

## Directory Structure

- `model/wav2vec2.rknn`: fixed-shape ASR model
- `samples/test.wav`: sample audio
- `task_runtime.py`: mono conversion, 16 kHz resampling, CTC decoding
- `web_service.py`, `inference.py`, `web/`: API, CLI, and English UI

## Quick Start

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-wav2vec2:latest
```

Open `http://<Board_IP>:8000`. Browser microphone permission requires HTTPS or a trusted/localhost context in many browsers; file upload always works.

### Command Line

```bash
python inference.py --platform rk3588 --model_dir model --file samples/test.wav
```

## API

```bash
curl -X POST http://127.0.0.1:8000/api/models/wav2vec2/predict \
    -F 'file=@samples/test.wav'
```

Audio is converted to mono 16 kHz. The model has a fixed 20-second input: shorter audio is padded and longer audio is explicitly reported as truncated. The model recognizes English speech only. OpenAPI documentation is at `/docs`.
