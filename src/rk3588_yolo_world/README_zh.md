# RK3588 YOLO-World 部署指南

[English](./README.md) | [中文]

本项目在 RK3588 上提供 YOLO-World 开放词汇目标检测。YOLO-World 检测模型和 CLIP 文本编码模型均在 NPU 上运行，Web 和 API 用户无需重新构建模型即可用自然语言动态切换检测类别。

> 内置 CLIP 模型主要使用英文训练，建议使用 `person`、`red bus`、`traffic light` 等英文名词短语。

## 核心特性

- 通过 Web、启动参数或 REST API 动态设置自然语言检索词。
- 最多支持 80 个检索类别，使用 `|` 分隔。
- 使用内置 `clip_vocab.txt` 离线完成 BPE 分词，运行时无需下载 Hugging Face 文件。
- 支持图片、MP4 指定帧、示例视频和摄像头推理。
- 支持 MJPEG 预览、阈值配置和异步视频分析。

## 快速开始

支持的 reComputer 开发板使用 `/dev/dri/renderD129` 作为 NPU 设备：

```bash
sudo docker run --rm --name rk3588-yolo_world \
  --privileged \
  -p 8000:8000 \
  -v /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolo_world:latest \
  python3 web_detection.py \
    --platform rk3588 \
    --model_path /app/model/yolo_world_v2s_i8.rknn \
    --text_model /app/model/clip_text_fp16.rknn \
    --text_features /app/model/coco_text_outp.npy \
    --vocab_path /app/model/clip_vocab.txt \
    --class_path /app/model/detect_classes.txt \
    --video_path /app/video/test.mp4 \
    --host 0.0.0.0 \
    --port 8000
```

访问 `http://<开发板IP>:8000`，输入 `person|bus|red car` 等检索词，可应用到实时视频流，或上传图片进行单次检索。

主机 8000 端口被占用时使用 `-p 8001:8000`。使用摄像头时，将 `--video_path` 替换为 `--camera_id 0`，并增加 `--device /dev/video0:/dev/video0`。

### 启动参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--platform` | 是 | — | 目标 NPU：`rk3588`。 |
| `--model_path` | 是 | — | 量化后的 YOLO-World RKNN 检测模型。 |
| `--text_model` | 否 | `model/clip_text_fp16.rknn` | 动态检索词使用的 FP16 CLIP 文本编码模型。 |
| `--text_features` | 否 | `model/coco_text_outp.npy` | 更改检索词前使用的 `(1,80,512)` COCO 预计算特征。 |
| `--vocab_path` | 否 | `model/clip_vocab.txt` | 离线 CLIP BPE 合并词表。 |
| `--class_path` | 否 | 内置 COCO 标签 | 与预计算特征各行顺序对应的类别名称。 |
| `--prompts` | 否 | COCO 80 类 | 初始动态检索词，例如 `"person|red bus|bicycle"`。 |
| `--video_path` | 否 | — | 循环播放的本地 MP4，优先于摄像头。 |
| `--camera_id` | 否 | `1` | 摄像头编号；`-1` 表示仅使用离线视频分析模式。 |
| `--host` | 否 | `0.0.0.0` | FastAPI 监听地址。 |
| `--port` | 否 | `8000` | 容器内部服务端口。 |

本地构建：

```bash
docker build -f docker/rk3588/yolo_world.dockerfile \
  -t rk3588-yolo_world:local src/rk3588_yolo_world
```

## 自然语言检索 API

### 查询或替换视频流检索词

```bash
curl http://127.0.0.1:8000/api/prompts

curl -X POST http://127.0.0.1:8000/api/prompts \
  -H "Content-Type: application/json" \
  -d '{"text":"person|red bus|traffic light"}'
```

`POST /api/prompts` 使用 CLIP RKNN 模型编码每个短语，并替换实时视频流和异步视频分析使用的类别。也可以传入 JSON 数组：`{"prompts":["person","red bus"]}`。

### 单次图片检索

**接口：** `POST /api/models/yolo_world/predict`
**类型：** `multipart/form-data`

- `file`：上传图片。
- `video` 和 `timestamp`：上传 MP4 和读取时间点（秒）。
- `realtime`：使用当前视频流帧。
- `text`：可选，本次请求使用的 `|` 分隔检索词，不会替换视频流检索词。
- `conf`：置信度阈值，默认 `0.25`。
- `iou`：分类 NMS IoU 阈值，默认 `0.45`。

```bash
curl -X POST http://127.0.0.1:8000/api/models/yolo_world/predict \
  -F "file=@model/bus.jpg" \
  -F "text=person|bus|red vehicle" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

响应会返回实际使用的检索词，并将匹配的自然语言短语作为检测类别。

## 检索词规则与处理逻辑

- 至少 1 个、最多 80 个非空检索词。
- 使用 `|` 分隔，因为逗号可能是自然语言短语的一部分。
- 每个 CLIP 输入最多保留 20 个 token，建议使用简短名词短语。
- 动态特征写入检测模型固定 80 行文本输入的前若干行；后处理会忽略填充行，避免未使用类别产生误检。
- 重复使用的检索词组合会在内存中缓存。

其他接口包括 `GET /api/health`、`GET/POST /api/config`、`GET /api/video_feed`，以及 `/api/video/*` 上传、分析、状态、列表和下载接口。
