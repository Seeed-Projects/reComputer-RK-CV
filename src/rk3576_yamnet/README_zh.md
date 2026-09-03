# RK3576 YAMNet 部署指南

[English](./README.md) | [中文]

本项目在 RK3576 NPU 上运行 YAMNet，对 AudioSet 的 521 类声音事件进行分类。

## 核心功能

- 上传音频或使用浏览器麦克风录音
- 输出全局 Top-K 以及三秒滑动窗口事件时间线
- 可配置分数阈值和 0.25–3 秒步长
- REST API、命令行推理和 Docker 一键部署

## 目录结构

- `model/yamnet.rknn`：固定三秒输入模型
- `model/yamnet_class_map.txt`：521 类标签
- `samples/test.wav`：测试音频
- `task_runtime.py`：音频转换、滑窗、排序和事件合并
- `web_service.py` / `inference.py` / `web/`：API、CLI 与英文界面

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yamnet:latest
```

访问 `http://<开发板IP>:8000`。

```bash
python inference.py --platform rk3576 --model_dir model --file samples/test.wav \
    --topk 5 --threshold 0.1 --hop_seconds 1

curl -X POST http://127.0.0.1:8000/api/models/yamnet/predict \
    -F 'file=@samples/test.wav' -F 'topk=5' \
    -F 'threshold=0.1' -F 'hop_seconds=1'
```

音频会转为单声道 16 kHz；窗口固定 3 秒。`topk` 范围 1–20，`threshold` 为 0–1，`hop_seconds` 为 0.25–3，最长音频 300 秒。响应含全局 `predictions`、窗口 `segments` 和合并后的 `events`，完整接口见 `/docs`。
