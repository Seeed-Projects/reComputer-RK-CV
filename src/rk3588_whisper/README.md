# RK3588 Whisper Deployment Guide

[English] | [中文](./README_zh.md)

This directory contains OpenAI's Whisper ASR (Automatic Speech Recognition) inference code optimized for RK3588.

## Core Features
- **Hardware Acceleration**: Full utilization of RK3588's 6 TOPS NPU computing power for audio transcription.
- **Sliding Window Inference**: Robust support for long audio processing using overlapping window strategies.
- **Hot-Swapping**: Dynamically load models and switch languages without restarting the service.
- **Web UI & API**: Real-time Web preview and RESTful API support for both synchronous short audio and asynchronous long audio transcription.

## Directory Structure
- `lib/`: Contains `librknnrt.so` for RK3588.
- `model/`: Stores `.rknn` encoder and decoder models converted for RK3588 (e.g., `whisper_encoder_base_20s.rknn`, `whisper_decoder_base_20s.rknn`).
- `py_utils/`: Contains utilities for log-mel spectrogram calculation and tokenization.
- `web_service.py`: Main program (supports Web UI and API).

## Quick Start

### 1. Run the Project (One command, Web preview)

This project supports access via **Web Browser**. The program automatically serves a web interface for speech recognition.

#### One-click Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-whisper:latest \
    python3 web_service.py
```
Access via: `http://<Board_IP>:8000`

---

## 🔌 API Documentation

This project provides RESTful interfaces for ASR tasks, supporting synchronous and asynchronous transcription of audio files.

### 1. Synchronous Transcription Interface (Short Audio)

**Endpoint:** `POST /api/models/whisper/predict`

Suitable for audio files under 20 seconds.

#### Request Parameters (Multipart/Form-Data):
- `file`: (Required) Audio file to be transcribed (e.g., .wav, .mp3).
- `language`: (Optional) Target language code (e.g., `en`, `zh`). If different from the current model, it will hot-swap the tokenizer.

#### Usage Examples:

```bash
curl -X POST "http://127.0.0.1:8000/api/models/whisper/predict" \
     -F "file=@/home/user/audio/test_en.wav" \
     -F "language=en"
```

#### Response Format (JSON):
```json
{
  "status": "success",
  "data": {
    "text": "Hello world, this is a test.",
    "language": "en",
    "duration": 3.5,
    "inference_time": 0.8
  }
}
```

### 2. Asynchronous Transcription Interface (Long Audio)

**Endpoint:** `POST /api/models/whisper/task`

Creates an asynchronous task for processing longer audio/video files.

#### Usage Examples:
```bash
curl -X POST "http://127.0.0.1:8000/api/models/whisper/task" \
     -F "file=@/home/user/audio/long_podcast.wav" \
     -F "language=zh"
```

#### Response Format (JSON):
```json
{
  "status": "success",
  "data": {
    "task_id": "29c7b932-a77f-480c-a18b-8a958c7911c3",
    "message": "Task created successfully. Poll /api/models/whisper/task/{task_id} for status."
  }
}
```

### 3. Task Status Polling

**Endpoint:** `GET /api/models/whisper/task/{task_id}`

#### Usage Examples:
```bash
curl "http://127.0.0.1:8000/api/models/whisper/task/29c7b932-a77f-480c-a18b-8a958c7911c3"
```

### 4. System Configuration Interface (Config)

Used to dynamically switch models and languages.

#### Get Current System Status
- **Endpoint:** `GET /api/system/status`
- **Response:** `{"status": "success", "data": {"model_size": "base", "language": "en", "max_tokens": 12, "rknn_lite_available": true}}`

#### Update System Configuration (Hot-Swap)
- **Endpoint:** `POST /api/system/config`
- **Request Parameters (Form-Data):** `model_size=base`, `language=zh`
- **Response:** `{"status": "success", "message": "Successfully loaded base model."}`

---

## 🛠️ Developer Guide (Production Recommendations)
### Code Description
- `web_service.py`:
    - **Web API**: Integrates FastAPI, supporting audio upload, async task queuing, and model hot-swapping.
    - **RKNN Inference**: Encapsulates RKNN initialization for both Encoder and Decoder models. Implements autoregressive generation loop.
- `py_utils/whisper_utils.py`:
    - **Audio Processing**: Calculates Log-Mel spectrograms aligned with OpenAI's implementation.
    - **Tokenizer**: Handles BPE tokenization and vocabulary management.

### Modifying Models
1. Place the trained and converted `.rknn` encoder and decoder models into the `model/` directory.
2. The service automatically loads models based on the size parameter (e.g., `whisper_encoder_base_20s.rknn`). Ensure file naming conventions are maintained.
