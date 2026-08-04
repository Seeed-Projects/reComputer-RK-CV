# RK3576 MobileNetV2 部署指南

[English](./README.md) | [中文]

本目录包含针对 RK3576 优化的 MobileNetV2 图像分类服务，提供基于 RKNN 的推理流程、浏览器预览、REST API，以及可直接运行的 Docker 镜像。

## 核心特性

- **硬件加速**：通过 RKNN-Toolkit-Lite2 在 RK3576 NPU 上运行 RKNN 模型。
- **ImageNet 分类**：结合 `synset.txt` 返回可配置的 Top-K 类别标签和置信度。
- **灵活输入**：支持上传图片、提取 MP4 指定时间帧、本地视频和摄像头画面。
- **Web 与 API**：提供浏览器预览、MJPEG 视频流、健康检查和 REST 推理接口。

## 目录结构

- `lib/`：RK3576 版 `librknnrt.so` 运行时库。
- `model/`：RK3576 RKNN 模型、ImageNet 标签和示例图片。
- `py_utils/`：MobileNetV2 预处理与 Top-K 后处理工具。
- `video/`：容器默认命令使用的示例 MP4 视频。
- `web_classification.py`：提供 RKNN 推理、Web 预览和 FastAPI 接口的主程序。
- `requirements.txt`：Python 运行依赖。

默认模型为 `model/rk3576_mobilenet_v2.rknn`，输入尺寸为 `224 x 224`。

## 快速开始

### 1. 使用 Docker 运行

在 RK3576 开发板上直接运行已发布镜像：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilenet:latest
```

默认命令会分析镜像内置的 `video/test.mp4`。通过 `http://<开发板IP>:8000` 访问 Web 预览，通过 `http://<开发板IP>:8000/docs` 查看交互式 API 文档。

如果只需要上传文件，不打开摄像头或本地视频，可覆盖容器启动命令：

```bash
sudo docker run --rm --privileged --net=host \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-mobilenet:latest \
    python3 web_classification.py \
    --model_path model/rk3576_mobilenet_v2.rknn --camera_id -1
```

### 2. 本地构建镜像

在仓库根目录执行：

```bash
docker build \
    -f docker/rk3576/mobilenet.dockerfile \
    -t rk3576-mobilenet:local \
    src/rk3576_mobilenet
```

## API 接口文档

### 1. 模型推理接口（Predict）

**Endpoint：** `POST /api/models/mobilenet/predict`

#### 请求参数（`multipart/form-data`）

- `file`：可选，待分类的图片文件。
- `video`：可选，待提取画面的 MP4 文件，不能与 `file` 同时传入。
- `timestamp`：可选，提取视频帧的非负时间戳，单位为秒；默认读取第一帧。
- `conf`：可选，本次请求的置信度阈值，范围为 `0.0` 至 `1.0`。
- `topk`：可选，返回结果数量，范围为 `1` 至 `20`，默认值为 `5`。

如果未提供 `file` 和 `video`，服务会在摄像头或本地视频帧可用时对当前帧进行分类。

#### 调用示例

图片分类：

```bash
curl -X POST "http://127.0.0.1:8000/api/models/mobilenet/predict" \
    -F "file=@model/bell.jpg" -F "topk=5" -F "conf=0.01"
```

分类视频 5.5 秒处的画面：

```bash
curl -X POST "http://127.0.0.1:8000/api/models/mobilenet/predict" \
    -F "video=@video/test.mp4" -F "timestamp=5.5" -F "topk=3"
```

#### 响应格式

```json
{
  "success": true,
  "model": "mobilenet",
  "source": "uploaded image",
  "predictions": [
    {
      "class": "n02123045 tabby, tabby cat",
      "confidence": 0.91
    }
  ],
  "image": {
    "width": 640,
    "height": 480
  }
}
```

### 2. 系统配置接口（Config）

- `GET /api/config`：返回 `{"conf_thresh": 0.0}`。
- `POST /api/config`：接受 `{"conf_thresh": 0.1}` 格式的 JSON，取值范围为 `0.0` 至 `1.0`。

全局阈值用于预览视频流，以及未单独提供 `conf` 的推理请求。

### 3. 视频与文件接口

- `GET /api/video_feed`：返回叠加分类结果的 MJPEG 预览流。
- `POST /api/video/upload`：通过 `file` 表单字段上传 JPG、JPEG、PNG、BMP 或 MP4 文件。
- `POST /api/video/analyze`：通过已上传的 `filename` 表单字段启动异步处理。
- `GET /api/video/status`：返回处理进度和错误状态。
- `GET /api/video/list`：列出已上传文件和生成结果。
- `GET /api/video/download/{filename}`：下载生成结果。
- `GET /api/health`：返回服务、平台和模型就绪状态。

## 开发者指南

### 代码说明

- `web_classification.py` 负责加载 RKNN 模型、初始化 RK3576 NPU 运行时、处理摄像头或视频帧，并提供 Web/API 服务。
- `py_utils/mobilenet_utils.py` 将输入缩放至 `224 x 224`，保留该模型要求的 BGR 通道顺序，执行 softmax 并筛选 Top-K 结果。
- 运行时上传文件和输出结果默认写入 `workspace/`，可通过 `RK_CV_WORKSPACE` 修改目录。

### 替换模型

1. 将兼容 RK3576 的 `.rknn` 模型放入 `model/`。
2. 确认模型输入布局、通道顺序、归一化方式和输出格式与 `MobileNet_helper` 一致。
3. 如果类别发生变化，请同步更新 `model/synset.txt`。
4. 使用 `--model_path model/<模型文件名>.rknn` 指定新模型。
