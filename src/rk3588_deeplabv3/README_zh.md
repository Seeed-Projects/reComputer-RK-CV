# RK3588 DeepLabV3 部署指南

[English](./README.md) | [中文]

本目录提供面向 RK3588 的标准化 DeepLabV3 语义分割部署，包含 RKNN NPU 推理、浏览器预览、REST API、视频处理和 Docker 一键运行。

## 核心特性

- **NPU 加速**：通过 RKNN Toolkit Lite2 在 RK3588 上运行 `deeplabv3.rknn`。
- **语义分割**：生成 21 类 PASCAL VOC 掩码，并统计当前图像中每个类别的像素数。
- **可视化预览**：将彩色掩码与原图叠加，通过 MJPEG 视频流输出。
- **标准服务**：提供健康检查、模型推理、配置、视频分析和 OpenAPI 接口。

## 目录结构

- `lib/`：RK3588 RKNN 运行时库。
- `model/deeplabv3.rknn` 和 `model/test.jpg`：RKNN 模型和预热示例。
- `rknn_runtime.py`：线程安全的 RKNN 加载和推理封装。
- `task_runtime.py`：513×513 预处理、掩码解码、PASCAL VOC 标签/颜色和预览生成。
- `web_service.py`：FastAPI 服务、浏览器界面、MJPEG 预览和视频处理。
- `video/test.mp4`：示例视频。
- `requirements.txt` 和 `rknn-toolkit-lite2-packages/`：运行依赖。

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-deeplabv3:latest
```

访问 `http://<开发板IP>:8000` 使用预览页面，或访问 `http://<开发板IP>:8000/docs` 查看 OpenAPI。

在仓库根目录进行本地构建：

```bash
docker build -f docker/rk3588/deeplabv3.dockerfile \
  -t rk3588-deeplabv3:local src/rk3588_deeplabv3
```

## API 接口文档

### 语义分割推理

**接口：** `POST /api/models/deeplabv3/predict`
**类型：** `multipart/form-data`

- `file`：必填，待分割图片。
- 通用服务层也接受 `threshold` 和 `topk`，但当前 argmax 解码器不使用这两个参数。

```bash
curl -X POST http://127.0.0.1:8000/api/models/deeplabv3/predict \
  -F "file=@model/test.jpg"
```

```json
{
  "success": true,
  "model": "deeplabv3",
  "inference_time": 0.0842,
  "result": {
    "classes": [
      {"id": 0, "class": "background", "pixels": 180240},
      {"id": 15, "class": "person", "pixels": 26418}
    ],
    "width": 640,
    "height": 480
  }
}
```

JSON 包含掩码统计；渲染后的叠加图可通过 `GET /api/video_feed` 查看。

### 配置与视频接口

- `GET /api/health`：模型、平台、输入类型、模型文件和就绪状态。
- `GET /api/config` 与 `POST /api/config`：通用配置（`threshold` 范围 0–1，`topk` 范围 1–100）。
- `GET /api/video_feed`：以 MJPEG 输出最新叠加图。
- `POST /api/video/upload` 和 `POST /api/video/analyze`：上传 MP4，并通过 multipart `filename` 字段启动逐帧分割。
- `GET /api/video/status`、`GET /api/video/list`、`GET /api/video/download/{filename}`：查询并获取结果。

## 开发者指南

处理流程将 BGR 输入缩放到 513×513 并转换为 RGB，执行 RKNN 推理，将 logits 恢复到原始分辨率，对 21 个 PASCAL VOC 类别执行 `argmax`，再以 50% 透明度叠加彩色掩码。

更换模型时，将针对 RK3588 转换的 `deeplabv3.rknn` 放入 `model/`。模型应保持 21 类输出布局，否则需同步修改 `task_runtime.py` 中的 `LABELS` 和输出处理。
