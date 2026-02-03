# reComputer-RK-CV

本项目旨在为瑞芯微（Rockchip）系列开发板提供工业级、高性能的计算机视觉（CV）应用方案。目前已支持 **RK3588** 和 **RK3576** 平台，主要集成了 YOLOv11 目标检测模型。

## 项目架构

项目采用多平台适配架构，各平台代码和环境配置独立管理：

```text
reComputer-RK-CV/
├── docker/                 # Docker 镜像配置文件
│   ├── rk3576/             # RK3576 专属 Dockerfile
│   └── rk3588/             # RK3588 专属 Dockerfile
├── src/                    # 源码目录
│   ├── rk3576/             # RK3576 源码、模型及依赖库
│   └── rk3588/             # RK3588 源码、模型及依赖库
└── .github/workflows/      # GitHub Actions 自动化构建脚本
```

## 支持平台

| 平台 | 芯片 | 算力 | 镜像名称 |
| :--- | :--- | :--- | :--- |
| **RK3588** | RK3588/RK3588S | 6 TOPS | `rk3588-yolo` |
| **RK3576** | RK3576 | 6 TOPS | `rk3576-yolo` |

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
sudo docker pull ghcr.io/litxaohu/recomputer-rk-cv/rk3588-yolo:latest
sudo docker pull ghcr.io/litxaohu/recomputer-rk-cv/rk3576-yolo:latest
```

#### 步骤 C：一键运行

**针对 RK3588:**
```bash
sudo docker run --rm --privileged --net=host --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/bus/usb:/dev/bus/usb \
    --device /dev/video1:/dev/video1 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/litxaohu/recomputer-rk-cv/rk3588-yolo:latest
    python realtime_detection.py --model_path model/yolo11n.rknn --camera_id 1
```

**针对 RK3576:**
```bash
sudo docker run --rm --privileged --net=host --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/bus/usb:/dev/bus/usb \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/litxaohu/recomputer-rk-cv/rk3576-yolo:latest
    python realtime_detection.py --model_path model/yolo11n.rknn --camera_id 0
```

#### 如何预览：
1.  **本地显示器**：自动弹出实时检测窗口（需连接显示器并执行了 xhost）。
2.  **Web 浏览器**：在局域网内访问 `http://<开发板IP>:8000` 即可实时预览。

#### 常见问题排查：
**问题：SSH 远程无屏幕运行报错 `qt.qpa.xcb: could not connect to display`**
解决方案：在运行命令末尾添加 `--no_gui` 参数，强制关闭本地窗口初始化。
```bash
# 示例 (在原有命令末尾追加):
... python realtime_detection.py --model_path model/yolo11n.rknn --camera_id 0 --no_gui
```

### 3. 独立 Web 预览模式 (仅浏览器查看)

如果您只需要通过 Web 浏览器查看预览画面（例如在远程服务器或无显示器环境下运行），可以使用专用的 Web 预览脚本：

**针对 RK3588:**
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video1:/dev/video1 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/litxaohu/recomputer-rk-cv/rk3588-yolo:latest \
    python web_detection.py --model_path model/yolo11n.rknn --camera_id 1
```

**针对 RK3576:**
```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/litxaohu/recomputer-rk-cv/rk3576-yolo:latest \
    python web_detection.py --model_path model/yolo11n.rknn --camera_id 0
```

访问方式：`http://<开发板IP>:8000`

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

### 3. 实时视频流接口 (Video Feed)

获取带有检测框绘制的实时 MJPEG 视频流，可直接嵌入 HTML `<img>` 标签。

- **Endpoint:** `GET /api/video_feed`
- **使用示例:** `<img src="http://<开发板IP>:8000/api/video_feed">`

## 平台详细文档

- [RK3588 使用指南](src/rk3588/README.md)
- [RK3576 使用指南](src/rk3576/README.md)

## 自动化构建

本项目支持通过 GitHub Actions 自动构建多平台镜像。
- 当修改 `src/rk3588/` 目录时，会自动触发 `rk3588-yolo` 镜像的构建。
- 当修改 `src/rk3576/` 目录时，会自动触发 `rk3576-yolo` 镜像的构建。
- 支持手动触发构建，并可指定 `image_tag`。

---

## �️ 开发者指南 (量产建议)
### 代码说明
- `realtime_detection.py`:
    - **双模支持**: 集成 FastAPI，同时支持本地渲染和 MJPEG 流式输出。
    - **环境自适应**: 自动检测 `DISPLAY` 环境变量，无环境时静默跳过 GUI 初始化。
    - **RKNN 推理**: 封装了 RKNN 初始化、加载模型、多核推理逻辑。
    - **后处理**: YOLOv11 专用的 Box 解码与 NMS 逻辑。

### 修改模型
1. 将训练好并转换完成的 .rknn 模型放入相应平台的 `model/` 目录。
2. 运行命令时可添加 `--model_path` 参数指向新模型（默认已在 Dockerfile 中配置）。
