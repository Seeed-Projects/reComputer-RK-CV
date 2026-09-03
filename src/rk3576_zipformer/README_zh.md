# RK3576 Zipformer 部署指南

[English](./README.md) | [中文]

本项目在 RK3576 上运行 Zipformer 编码器、解码器和 Joiner RKNN 模型，支持中英文流式语音识别。

## 核心功能

- 上传完整音频进行转写
- 浏览器麦克风通过 WebSocket 发送 PCM，并接收实时部分结果
- 输出 Token、时间戳和处理时长
- REST API、WebSocket API 和命令行推理

## 目录结构

- `model/`：编码器、解码器、Joiner 和中英文词表
- `samples/test.wav`：测试音频
- `task_runtime.py`：Kaldi 风格 FBank 与流式缓存
- `web_service.py` / `inference.py` / `web/`：API、CLI 与英文界面

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-zipformer:latest
```

访问 `http://<开发板IP>:8000`，浏览器采集麦克风不需要映射 `/dev/snd`。

```bash
python inference.py --platform rk3576 --model_dir model --file samples/test.wav

curl -X POST http://127.0.0.1:8000/api/models/zipformer/predict \
    -F 'file=@samples/test.wav'
```

流式端点为 `WS /api/models/zipformer/stream`：发送单声道 16 kHz、有符号 16 位小端 PCM 二进制块，结束时发送 `{"action":"stop"}`。服务返回 `ready`、`partial`、`final` 或 `error` JSON。REST 文档见 `/docs`。
