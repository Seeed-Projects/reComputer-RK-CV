# RK3588 Wav2Vec2 部署指南

[English](./README.md) | [中文]

本项目将 `facebook/wav2vec2-base-960h` 转换为 RKNN，在 RK3588 NPU 上进行英文语音识别。

## 核心功能

- 上传 WAV 等 libsndfile 支持的音频，或使用浏览器麦克风录音
- Web 音频播放器、波形、英文转写和耗时信息
- REST API、命令行推理和 Docker 一键部署

## 目录结构

- `model/wav2vec2.rknn`：固定输入模型
- `samples/test.wav`：测试音频
- `task_runtime.py`：单声道转换、16 kHz 重采样和 CTC 解码
- `web_service.py` / `inference.py` / `web/`：API、CLI 与英文界面

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-wav2vec2:latest
```

访问 `http://<开发板IP>:8000`。许多浏览器只在 HTTPS 或 localhost 安全上下文允许麦克风，文件上传不受影响。

```bash
python inference.py --platform rk3588 --model_dir model --file samples/test.wav

curl -X POST http://127.0.0.1:8000/api/models/wav2vec2/predict \
    -F 'file=@samples/test.wav'
```

音频会转为单声道 16 kHz。模型固定输入 20 秒：短音频补零，长音频只识别前 20 秒并在响应中标记 `truncated`。仅支持英文语音，完整接口见 `/docs`。
