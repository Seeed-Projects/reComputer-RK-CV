# RK3588 Wav2Vec2 部署指南

[English](./README.md) | [中文]

本项目将 `facebook/wav2vec2-base-960h` 转换为 RKNN，在 RK3588 NPU 上进行英文语音识别。

## 核心功能

- 上传 WAV 等 libsndfile 支持的音频，或使用浏览器麦克风录音
- 浏览器录音完成后统一重采样，后端执行静音裁剪、音量归一化和削波检测
- CTC Beam Search 解码；超过 20 秒的音频自动重叠分段并合并文本
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
python inference.py --platform rk3588 --model_dir model \
    --file samples/test.wav --beam_width 8

curl -X POST http://127.0.0.1:8000/api/models/wav2vec2/predict \
    -F 'file=@samples/test.wav' -F 'beam_width=8'
```

音频会转为单声道 16 kHz，并在推理前进行静音裁剪和音量归一化。RKNN 模型仍为固定 20 秒输入；长音频会重叠分段后合并文本，不再直接截断。`beam_width` 范围为 1-32：`1` 是速度最快的贪心解码，默认值 `8` 会使用少量 CPU 换取更好的解码结果。模型仅支持英文语音，完整接口见 `/docs`。
