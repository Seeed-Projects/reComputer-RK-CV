# reComputer-RK-CV

[English](./README.md) | [中文]

本项目为瑞芯微（Rockchip）开发板提供可直接部署的计算机视觉（CV）应用。
目前支持 **RK3588** 和 **RK3576**，涵盖目标检测、图像分类、姿态估计、
实例分割和语音等案例。

## 项目架构

项目采用多平台适配架构，各平台代码和环境配置独立管理：

```text
reComputer-RK-CV/
├── docker/                 # Docker 镜像配置文件
│   ├── rk3576/             # RK3576 专属 Dockerfile
│   └── rk3588/             # RK3588 专属 Dockerfile
├── src/                    # 源码目录
│   ├── rk3576_<模型>/      # 独立的 RK3576 应用构建上下文
│   └── rk3588_<模型>/      # 独立的 RK3588 应用构建上下文
└── .github/workflows/      # GitHub Actions 自动化构建脚本
```

## 支持平台

| 平台 | 芯片 | 算力 | 镜像名称 |
| :--- | :--- | :--- | :--- |
| **RK3588** | RK3588/RK3588S | 6 TOPS | `rk3588-yolo` |
| **RK3576** | RK3576 | 6 TOPS | `rk3576-yolo` |

图像分类服务：

| 平台 | 模型 | Web/API 入口 | Dockerfile |
| :--- | :--- | :--- | :--- |
| RK3588 | MobileNet | `POST /api/models/mobilenet/predict` | `docker/rk3588/mobilenet.dockerfile` |
| RK3588 | ResNet50V2 | `POST /api/models/resnet50v2/predict` | `docker/rk3588/resnet50v2.dockerfile` |
| RK3576 | MobileNet | `POST /api/models/mobilenet/predict` | `docker/rk3576/mobilenet.dockerfile` |
| RK3576 | ResNet50V2 | `POST /api/models/resnet50v2/predict` | `docker/rk3576/resnet50v2.dockerfile` |

以下标准服务均提供 RK3588 和 RK3576 两个平台版本：

| 类别 | 模型 |
| :--- | :--- |
| 图像分割 | DeepLabV3、PPSeg、MobileSAM |
| 视觉与 OCR | RetinaFace、LPRNet、PPOCR（检测 + 识别）、CLIP |
| 文本 | Lite Transformer、MMS-TTS |
| 语音与音频 | Wav2Vec2、Whisper、Zipformer、YAMNet |

每个服务均提供 Web 页面、`GET /api/health`、`/docs` OpenAPI 文档和
`POST /api/models/<模型>/predict` 推理接口。

## 快速开始

### 1. 安装 Docker

在开发板上执行以下命令安装 Docker：

```bash
# 下载安装脚本
curl -fsSL https://get.docker.com -o get-docker.sh
# 使用阿里云镜像源安装
sudo sh get-docker.sh --mirror Aliyun
# 启动 Docker 并设置开机自启
sudo systemctl enable docker
sudo systemctl start docker
```

### 2. 运行项目 (一条命令，双模预览)

本项目支持 **本地 GUI** 与 **Web 浏览器** 双模式同时预览。程序会自动检测显示器环境，无显示器时自动降级为 Web 模式。

#### 步骤 A：配置显示权限 (可选)
如果您连接了显示器并希望在本地看到窗口：
```bash
xhost +local:docker
```

#### 步骤 B：拉取镜像
```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolo11:latest
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolo11:latest
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolov8-obb:latest
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolov8-obb:latest
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolov8_airborne:latest
```

#### 步骤 C：一键运行

**针对 RK3588:**
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video1:/dev/video1 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolo11:latest \  
    python web_detection.py --model_path model/yolo11n.rknn --video video/test.mp4
```

**针对 RK3576:**
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolo11:latest \  
    python web_detection.py --model_path model/yolo11n.rknn --video video/test.mp4
```

访问方式：`http://<开发板IP>:8000`

> **注意**: 如果需要自定义类别，可以增加 `-v $(pwd)/class_config.txt:/app/class_config.txt \` 挂载和 `--class_path` 参数，程序默认使用 COCO 80 类。

例如：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    -v $(pwd)/class_config.txt:/app/class_config.txt \
    --device /dev/video1:/dev/video1 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-yolo11:latest \  
    python web_detection.py --model_path model/yolo11n.rknn --video video/test.mp4 --class_path class_config.txt
```

---

## 🔌 API 接口文档

本项目提供了兼容 Ultralytics Cloud API 标准的 RESTful 接口，支持通过 HTTP POST 请求上传图片进行目标检测。

### 1. 模型推理接口 (Predict)

**Endpoint:** `POST /api/models/yolo11/predict`

#### 请求参数 (Multipart/Form-Data):
- `file`: (可选) 待检测的图片文件。
- `video`: (可选) 待检测的 MP4 视频文件。
- `timestamp`: (可选) 视频文件的时间戳（单位：秒），返回该时间点的视频帧检测结果。默认为 0。
- `realtime`: (可选) 布尔值。若为 `true` 或未提供 `file`/`video` 参数，则返回摄像头当前帧的检测结果。
- `conf`: (可选) 单次请求的置信度阈值，范围 0.0-1.0。
- `iou`: (可选) 单次请求的 NMS IOU 阈值，范围 0.0-1.0。

#### 调用示例:

**1. 图片检测:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict" -F "file=@/home/cat/001.jpg"
```

**2. 视频特定时间帧检测:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict" -F "video=@/home/cat/test.mp4" -F "timestamp=5.5"
```

**3. 获取摄像头当前帧检测:**
```bash
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict" -F "realtime=true"
# 或者不传文件参数
curl -X POST "http://127.0.0.1:8000/api/models/yolo11/predict"
```

#### 响应格式 (JSON):
```json
{
  "success": true,
  "source": "video frame at 5.5s",
  "predictions": [
    {
      "class": "person",
      "confidence": 0.92,
      "box": { "x1": 100, "y1": 200, "x2": 300, "y2": 500 }
    }
  ],
  "image": { "width": 1280, "height": 720 }
}
```

### 2. 系统配置接口 (Config)

用于动态调整实时视频流和默认推理的阈值。

#### 获取当前配置
- **Endpoint:** `GET /api/config`
- **响应:** `{"obj_thresh": 0.25, "nms_thresh": 0.45}`

#### 更新系统配置
- **Endpoint:** `POST /api/config`
- **请求体 (JSON):** `{"obj_thresh": 0.3, "nms_thresh": 0.5}`
- **响应:** `{"status": "success"}`

### 3. 命令行参数说明

运行 `web_detection.py` 时支持以下参数：

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--model_path` | RKNN 模型文件路径 | (必填) |
| `--camera_id` | 摄像头设备 ID (如 `/dev/video1` 则填 1) | 1 |
| `--video_path` | 视频文件路径 (若提供则忽略 camera_id) | None |
| `--class_path` | 自定义类别配置文件路径 (class_config.txt) | None (默认 COCO 80类) |
| `--host` | Web 服务器监听地址 | `0.0.0.0` |
| `--port` | Web 服务器端口 | 8000 |

#### 自定义类别配置 (class_config.txt) 格式：
使用双引号命名分类，不同分类之间使用逗号隔开，例如：
`"person", "bicycle", "car", "motorbike"`

---

## 实时视频流接口 (Video Feed)

获取带有检测框绘制的实时 MJPEG 视频流，可直接嵌入 HTML `<img>` 标签。

- **Endpoint:** `GET /api/video_feed`
- **使用示例:** `<img src="http://<开发板IP>:8000/api/video_feed">`

## 平台详细文档

- [RK3588 MobileNet](src/rk3588_mobilenet/README_zh.md)
- [RK3588 ResNet50V2](src/rk3588_resnet50v2/README_zh.md)
- [RK3576 MobileNet](src/rk3576_mobilenet/README_zh.md)
- [RK3576 ResNet50V2](src/rk3576_resnet50v2/README_zh.md)
- [RK3588 PPOCR（检测 + 识别）](src/rk3588_ppocr/README_zh.md)
- [RK3576 PPOCR（检测 + 识别）](src/rk3576_ppocr/README_zh.md)
- [RK3588 YOLOv6](src/rk3588_yolov6/README_zh.md)
- [RK3576 YOLOv6](src/rk3576_yolov6/README_zh.md)
- [RK3588 YOLOv7](src/rk3588_yolov7/README_zh.md)
- [RK3576 YOLOv7](src/rk3576_yolov7/README_zh.md)
- [RK3588 YOLOv10](src/rk3588_yolov10/README_zh.md)
- [RK3576 YOLOv10](src/rk3576_yolov10/README_zh.md)
- [RK3588 YOLOX](src/rk3588_yolox/README_zh.md)
- [RK3576 YOLOX](src/rk3576_yolox/README_zh.md)
- [RK3588 PP-YOLOE](src/rk3588_ppyoloe/README_zh.md)
- [RK3576 PP-YOLOE](src/rk3576_ppyoloe/README_zh.md)
- [RK3588 YOLO-World](src/rk3588_yolo_world/README_zh.md)
- [RK3576 YOLO-World](src/rk3576_yolo_world/README_zh.md)
- [RK3576 分割、OCR、文本与音频标准服务](src/rk3576_deeplabv3/README_zh.md)
- [RK3588 分割、OCR、文本与音频标准服务](src/rk3588_deeplabv3/README_zh.md)
- [Python 模型项目迁移规范](src/PYTHON_PROJECT_STANDARD.md)

## 自动化构建

本项目支持通过 GitHub Actions 自动构建多平台镜像。
- 每个 `src/rk<平台>_<模型>/` 目录都是独立的 Docker 构建上下文。
- 上述标准服务均已加入 RK3588、RK3576 构建矩阵。
- 支持手动触发构建，并可指定 `image_tag`。

---

## 🛠️ 开发者指南 (量产建议)
### 代码说明
- `web_detection.py`:
    - **双模支持**: 集成 FastAPI，同时支持本地渲染和 MJPEG 流式输出。
    - **环境自适应**: 自动检测 `DISPLAY` 环境变量，无环境时静默跳过 GUI 初始化。
    - **RKNN 推理**: 封装了 RKNN 初始化、加载模型、多核推理逻辑。
    - **动态加载**: 支持通过 `--class_path` 动态加载类别配置。
    - **后处理**: YOLOv11 专用的 Box 解码与 NMS 逻辑。

### 修改模型
1. 将训练好并转换完成的 .rknn 模型放入相应平台的 `model/` 目录。
2. 运行命令时可添加 `--model_path` 参数指向新模型（默认已在 Dockerfile 中配置）。

## RK182x PCIe 加速卡 · RKNN3 CV

以下 **11 个独立 Docker 项目、26 个型号配置** 已在 RK3588 + RK1820 M.2 上完成实机验证，均使用 RKNN3 1.1.0、INT8 W8A8。每个项目的中英文 README 包含全部型号的实测 benchmark FPS、测试条件和原始 JSON 链接。

**已通过 RK3588 + RK1820 M.2 实机验证。** GitHub Actions 构建全部 11 个 ARM64 镜像，发布 `latest` 和提交号标签到 GHCR；发布状态见 Actions。各项目文档均包含 GHCR 一键部署、本地构建和实测 benchmark。

| 项目 | 覆盖型号 | 镜像名称 | 文档 |
|---|---|---|---|
| Depth Anything V3 Base | `da3-base` | `rv182x-rk3588-depth-anything-v3` | [README](src/rv182x_rk3588_depth_anything_v3/README_zh.md) |
| MobileNetV2 | `mobilenetv2-12` | `rv182x-rk3588-mobilenetv2` | [README](src/rv182x_rk3588_mobilenetv2/README_zh.md) |
| QA-CLIP ViT-L/14 | `qaclip-vit-l14` | `rv182x-rk3588-qaclip` | [README](src/rv182x_rk3588_qaclip/README_zh.md) |
| RAFT-Stereo | `raftstereo-eth3d-160x240`, `raftstereo-middlebury-160x240`, `raftstereo-sceneflow-160x240`, `raftstereo-realtime-160x240` | `rv182x-rk3588-raft-stereo` | [README](src/rv182x_rk3588_raft_stereo/README_zh.md) |
| ResNet50-v2 | `resnet50-v2-7` | `rv182x-rk3588-resnet` | [README](src/rv182x_rk3588_resnet/README_zh.md) |
| YOLO26 | `yolo26n`, `yolo26s`, `yolo26m` | `rv182x-rk3588-yolo26` | [README](src/rv182x_rk3588_yolo26/README_zh.md) |
| YOLO26 Pose | `yolo26n-pose`, `yolo26s-pose`, `yolo26m-pose` | `rv182x-rk3588-yolo26-pose` | [README](src/rv182x_rk3588_yolo26_pose/README_zh.md) |
| YOLO26 Segment | `yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg` | `rv182x-rk3588-yolo26-segment` | [README](src/rv182x_rk3588_yolo26_segment/README_zh.md) |
| YOLOv5 | `yolov5n`, `yolov5s`, `yolov5m` | `rv182x-rk3588-yolov5` | [README](src/rv182x_rk3588_yolov5/README_zh.md) |
| YOLOv6 | `yolov6n`, `yolov6s`, `yolov6m` | `rv182x-rk3588-yolov6` | [README](src/rv182x_rk3588_yolov6/README_zh.md) |
| YOLOv8 | `yolov8n`, `yolov8s`, `yolov8m` | `rv182x-rk3588-yolov8` | [README](src/rv182x_rk3588_yolov8/README_zh.md) |

主机先安装匹配的 PCIe 驱动、固件和 RKNN3 proxy，并确认加速卡 Online。普通启动不刷写固件。以 YOLO26 为例：

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolo26:latest
sudo docker run --rm --privileged --net=host --ipc=host \
  -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
  -v "$PWD/workspace:/app/workspace" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolo26:latest \
  python3 web_detection.py --variant yolo26n --model_path model/yolo26n.rknn --video video/test.mp4
```

访问 `http://<开发板IP>:8000`，API 文档 `/docs`。通过 `MODEL_VARIANT` 选择该项目中的型号；多个演示共用 8000 端口，切换前先停止当前容器。新测试视频统一为用户提供的 `30_1080p.mp4`；RAFT 使用官方真实左右目图片对。若要进行姿态视频效果验收，建议补充近景人物视频；道路双目视频需要同步、校正的左右目素材。

QAClip 先缓存文本向量并释放文本编码器，再运行视觉编码器；其 FPS 对应缓存文本后的逐帧处理。Depth Anything V3 固定单 view、280×280，并串行运行三段模型。RAFT 固定 1 次迭代，FPS 按左右目图片对计数。每个项目区分同步推理调用 FPS 与包含解码、后处理和展示编码的 Pipeline FPS。

专用工作流 `.github/workflows/docker-build-rv182x.yml` 在 push 后构建全部 11 个模型，支持通过 workflow_dispatch 手动选择单个模型。
