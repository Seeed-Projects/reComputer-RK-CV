# RK3588 MMS-TTS 部署指南

[English](./README.md) | [中文]

本项目将 `facebook/mms-tts-eng` 编码器和解码器转换为 RKNN，在 RK3588 NPU 上合成英文语音。

## 核心功能

- 英文文本输入与字符校验
- Web、API 和 CLI 支持 0.6x 至 1.4x 语速调节
- 浏览器播放、下载生成的 WAV
- REST API 和可指定输出文件的 CLI
- 输出音频时长、采样率和采样数

## 目录结构

- `model/`：MMS-TTS 编码器和解码器
- `task_runtime.py`：词表、时长对齐、推理和 WAV 编码
- `web_service.py`：FastAPI 与生成音频下载接口
- `inference.py` / `web/`：单次 CLI 与英文界面

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-mms_tts:latest
```

访问 `http://<开发板IP>:8000`。

```bash
python inference.py --platform rk3588 --model_dir model \
    --text 'hello from the edge' --speaking_rate 0.8 --output output.wav

curl -X POST http://127.0.0.1:8000/api/models/mms_tts/predict \
    -F 'text=hello from the edge' -F 'speaking_rate=0.8'
curl -o speech.wav http://127.0.0.1:8000/api/audio/<audio_filename>
```

`speaking_rate` 范围是 `0.6` 至 `1.4`，默认 `1.0`；小于 1 表示更慢。该参数在解码前缩放音素时长，不会像 WAV 重采样一样改变音高。较长文本使用低语速时可能达到解码器时长上限，响应会返回 `duration_clipped` 和警告。预测响应包含 `audio_url` 和 `audio_filename`，服务只保留最新 20 个临时结果。输入最多 99 个模型支持的英文字符；不支持的字符会明确报错。完整接口见 `/docs`。
