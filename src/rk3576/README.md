# RK3576 YOLO 部署指南

本目录包含针对 RK3576 优化的 YOLOv11 推理代码。

## 核心特性
- **硬件加速**：针对 RK3576 的 2 TOPS NPU 架构进行了优化。
- **最新驱动**：集成支持 RK3576 的第 5 代 NPU 运行时库。
- **灵活输入**：支持摄像头和本地 MP4 视频输入。

## 目录结构
- `lib/`：包含 RK3576 版 `librknnrt.so`。
- `model/`：存放针对 RK3576 转换的 `.rknn` 模型。
- `realtime_detection.py`：主程序。

## 快速开始

### 1. 运行项目 (一条命令，双模预览)

本项目支持 **本地 GUI** 与 **Web 浏览器** 双模式同时预览。程序会自动检测显示器环境，无显示器时自动降级为 Web 模式。

#### 步骤 A：配置显示权限 (可选)
如果您连接了显示器并希望在本地看到窗口：
```bash
xhost +local:docker
```

#### 步骤 B：一键运行
```bash
sudo docker run --rm --privileged --net=host --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/bus/usb:/dev/bus/usb \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolo:latest \
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

### 2. 独立 Web 预览模式 (仅浏览器查看)

如果您只需要通过 Web 浏览器查看预览画面（例如在远程服务器或无显示器环境下运行），可以使用专用的 Web 预览脚本：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e RKNN_LOG_LEVEL=0 \
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD128:/dev/dri/renderD128 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-yolo:latest \
    python web_detection.py --model_path model/yolo11n.rknn --camera_id 0
```
访问方式：`http://<开发板IP>:8000`

---

## 🔌 API 接口文档

本项目提供了兼容 Ultralytics Cloud API 标准的 RESTful 接口，支持通过 HTTP POST 请求上传图片、视频或直接调用摄像头进行目标检测。

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

---

## ️ 开发者指南 (量产建议)
### 代码说明
- `realtime_detection.py`:
    - **双模支持**: 集成 FastAPI，同时支持本地渲染和 MJPEG 流式输出。
    - **环境自适应**: 自动检测 `DISPLAY` 环境变量，无环境时静默跳过 GUI 初始化。
    - **RKNN 推理**: 封装了 RKNN 初始化、加载模型、多核推理逻辑。
    - **后处理**: YOLOv11 专用的 Box 解码与 NMS 逻辑。

### 修改模型
1. 将训练好并转换完成的 .rknn 模型放入 `model/` 目录。
2. 运行命令时可添加 `--model_path` 参数指向新模型。
