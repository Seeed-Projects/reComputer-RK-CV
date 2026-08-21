# RK3576 PPOCR 端到端 OCR 部署指南

[English](./README.md) | [中文]

本项目将 PPOCR-Det 与 PPOCR-Rec 融合为一个适用于 RK3576 的服务。用户可以从 Web 页面、REST API 或 `--image` 命令行参数输入完整图片。服务会自动检测文字区域、按四点坐标进行透视裁剪、逐区域识别，并返回文字、坐标和置信度。

## 核心功能

- 使用 RKNNLite 在 RK3576 NPU 上同时运行 `ppocr_det.rknn` 和 `ppocr_rec.rknn`。
- 检测文档和自然场景图片中的多个文字区域。
- 按阅读顺序排列区域，并执行四点透视矫正。
- Web 端展示标注图片、识别全文、裁剪缩略图、置信度和分阶段耗时。
- 提供标准图片预测 API 和 OpenAPI 文档。
- 支持启动时分析本地图片以及纯命令行单次推理。
- 不包含视频上传、视频分析或本地摄像头代码。

## 处理流程

```text
静态图片 -> PPOCR-Det -> 排序后的四点文字框 -> 透视裁剪
         -> PPOCR-Rec -> 文字、置信度、坐标 -> 标注结果图
```

## 目录结构

- `model/ppocr_det.rknn`：RK3576 文字检测模型。
- `model/ppocr_rec.rknn`：RK3576 文字识别模型。
- `model/ppocr_keys_v1.txt`：CTC 字符表。
- `model/simfang.ttf`：Unicode 标注字体。
- `model/test.jpg`：启动预热图片。
- `task_runtime.py`：检测、排序、透视裁剪、识别和标注。
- `web_service.py`：Web 页面、REST API 和命令行图片入口。
- `rknn_runtime.py`：线程安全的 RKNNLite 封装。
- `utils/`：PPOCR 预处理与后处理工具。

## Docker 快速启动

```bash
sudo docker run --rm --name rk3576-ppocr \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-ppocr:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --host 0.0.0.0 --port 8000
```

访问 `http://<开发板IP>:8000`，OpenAPI 地址为 `http://<开发板IP>:8000/docs`。

在仓库根目录构建：

```bash
docker build -f docker/rk3576/ppocr.dockerfile \
  -t rk3576-ppocr:local src/rk3576_ppocr
```

## Web 使用流程

1. 上传一张包含一个或多个文字区域的 JPG、PNG、BMP 或 WEBP 图片。
2. 根据需要调整检测阈值、文字框阈值、扩张比例和识别阈值。
3. 点击 **Analyze image**。
4. 在结果图片中查看文字多边形与识别标签。
5. 在右侧查看按阅读顺序排列的裁剪图、文字、置信度、坐标和耗时。

页面只接受静态图片，已经主动移除视频和摄像头交互入口。

## 使用命令行分析本地图片

启动时分析图片，并继续运行 Web 服务：

```bash
python web_service.py --platform rk3576 --model_dir model \
  --image /data/document.jpg --host 0.0.0.0 --port 8000
```

执行一次推理、输出 JSON、保存标注图片后退出：

```bash
python web_service.py --platform rk3576 --model_dir model \
  --image /data/document.jpg --output /data/document_result.jpg --no_server
```

在 Docker 中使用本地图片时需要挂载图片目录：

```bash
sudo docker run --rm --privileged \
  -e RKNN_LOG_LEVEL=0 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  -v "$PWD/data:/data" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-ppocr:latest \
  python web_service.py --platform rk3576 --model_dir /app/model \
    --image /data/document.jpg --output /data/document_result.jpg --no_server
```

## 命令行参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--platform` | 是 | — | RKNN 目标平台，本目录使用 `rk3576`。 |
| `--model_dir` | 否 | `model` | 同时包含两个模型、字符表、字体和样例的目录。 |
| `--image` | 否 | — | 服务启动前分析的本地静态图片。 |
| `--output` | 否 | — | 保存 `--image` 标注结果的 JPG/PNG 路径。 |
| `--no_server` | 否 | 关闭 | 处理 `--image` 后直接退出。 |
| `--host` | 否 | `0.0.0.0` | FastAPI 监听地址。 |
| `--port` | 否 | `8000` | FastAPI 监听端口。 |

## REST API

**接口：** `POST /api/models/ppocr/predict`
**类型：** `multipart/form-data`

| 字段 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `file` | 是 | — | JPG、PNG、BMP 或 WEBP 图片；curl 本地路径前必须添加 `@`。 |
| `det_threshold` | 否 | `0.3` | 像素概率图检测阈值，范围 0～1。 |
| `box_threshold` | 否 | `0.6` | DB 文字框最低分数，范围 0～1。 |
| `unclip_ratio` | 否 | `1.5` | 文字多边形扩张比例，范围 0.1～5。 |
| `drop_score` | 否 | `0.5` | 计入组合全文的最低识别置信度。 |
| `max_results` | 否 | `100` | 送入 Rec 的最大区域数，范围 1～1000。 |
| `include_crops` | 否 | `false` | 是否在每行结果中返回 Base64 JPEG 裁剪缩略图。 |

```bash
curl -X POST http://127.0.0.1:8000/api/models/ppocr/predict \
  -F "file=@/data/document.jpg" \
  -F "drop_score=0.5" \
  -F "include_crops=false"
```

主要响应字段：

```json
{
  "success": true,
  "model": "ppocr",
  "inference_time": 0.1532,
  "result": {
    "text": "第一行识别文字\n第二行识别文字",
    "lines": [
      {
        "index": 1,
        "text": "第一行识别文字",
        "confidence": 0.9621,
        "accepted": true,
        "box": [[42, 31], [286, 29], [287, 72], [43, 74]],
        "crop_size": {"width": 245, "height": 43}
      }
    ],
    "count": 2,
    "detected_count": 2,
    "processed_count": 2,
    "timing_ms": {"detection": 54.2, "recognition": 71.4, "total": 129.8}
  }
}
```

其他接口：

- `GET /api/health`：模型与平台状态。
- `GET /api/config`、`POST /api/config`：读取或修改默认阈值。
- `GET /api/results/latest`：最新结构化 OCR 结果。
- `GET /api/results/latest/image`：最新 JPEG 标注结果。
- `GET /docs`：交互式 OpenAPI 文档。

## 模型范围与限制

- Rec 针对经过透视矫正的单行文字图片，Det 的定位质量会直接影响识别结果。
- 极小、模糊、弯曲、竖排、低对比度或大角度旋转文字可能漏检或识别错误。
- 当前使用简单规则按从上到下、从左到右排序；复杂多栏文档可能需要额外版面分析模型。
- `include_crops=true` 会增大 API 响应，主要供 Web 结果面板使用。

模型转换源文件位于 `rknn_model_zoo/examples/PPOCR/PPOCR-Det` 和 `PPOCR/PPOCR-Rec`。
