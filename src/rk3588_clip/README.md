# RK3588 CLIP Deployment Guide

[English] | [中文](./README_zh.md)

OpenAI CLIP ViT-B/32 demo accelerated by the RK3588 NPU. It demonstrates both zero-shot image classification and natural-language image retrieval.

## Features

- Classify one image using user-defined candidate labels
- Rank up to 32 uploaded images with a natural-language query
- Browser previews, REST APIs, and one-shot CLI
- 224 × 224 image preprocessing and normalized cosine similarity

## Directory Structure

- `model/`: RKNN image/text encoders and CLIP vocabulary
- `samples/`: example image and prompt text
- `task_runtime.py`: tokenizer, image preprocessing, and similarity ranking
- `web_service.py`, `inference.py`, `web/`: API, CLI, and English UI

## Quick Start

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-clip:latest
```

Open `http://<Board_IP>:8000`.

### Command Line

```bash
python inference.py --platform rk3588 --model_dir model \
    --file samples/dog_224x224.jpg \
    --prompts 'a photo of a dog|a photo of a cat'

python inference.py --platform rk3588 --model_dir model \
    --files image1.jpg image2.jpg --query 'a red car' --topk 2
```

## API

```bash
curl -X POST http://127.0.0.1:8000/api/models/clip/predict \
    -F 'file=@samples/dog_224x224.jpg' \
    -F $'prompts=a photo of a dog\na photo of a cat'

curl -X POST http://127.0.0.1:8000/api/models/clip/retrieve \
    -F 'files=@image1.jpg' -F 'files=@image2.jpg' \
    -F 'text=a red car' -F 'topk=2'
```

`prompts` accepts a JSON array, newline-separated values, or `|`-separated values. Retrieval accepts 1–32 images. Text is limited by the converted CLIP model's fixed 20-token input. OpenAPI documentation is at `/docs`.
