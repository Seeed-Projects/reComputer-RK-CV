# RK3576 Lite Transformer 部署指南

[English](./README.md) | [中文]

本项目是在 RK3576 NPU 上运行的英译中演示。Web 端同时显示译文以及模型实际处理的 BPE Token ID。

## 核心功能

- 英文输入、中文输出；提供 Web、REST API 和单次命令行推理
- 明确校验最多 15 个源 BPE Token，不会静默截断
- Docker 一键部署

## 目录结构

- `model/`：编码器、解码器、分词字典和嵌入文件
- `task_runtime.py`：预处理、RKNN 推理和解码
- `web_service.py` / `inference.py`：Web/API 服务与命令行程序
- `web/`：英文 Web 界面

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-lite_transformer:latest
```

访问 `http://<开发板IP>:8000`，不需要映射摄像头、显示器或音频设备。

```bash
python inference.py --platform rk3576 --model_dir model --text "thank you"

curl -X POST http://127.0.0.1:8000/api/models/lite_transformer/predict \
    -F 'text=thank you'
```

响应包含 `translation`、输入/输出 Token 和数量。固定编码器最多接收 15 个 BPE Token 加 EOS，一个英文单词可能拆成多个 Token。本模型适合短句演示，完整接口见 `/docs`。
