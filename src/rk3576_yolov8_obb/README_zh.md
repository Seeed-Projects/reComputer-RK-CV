# RK3576 YOLOv8-OBB 部署指南

本目录包含针对 RK3576 优化的 YOLOv8-OBB (Oriented Bounding Box，旋转目标检测) 推理代码。
基于 RK3576 YOLO 示例架构，适配了旋转框检测逻辑，支持 DOTA 等航拍数据集场景。

## 核心特性
- **硬件加速**：针对 RK3576 的 NPU 架构进行了优化。
- **OBB 支持**：支持旋转边界框检测 (x, y, w, h, angle)，适用于细长、倾斜物体的检测。
- **灵活输入**：支持摄像头和本地 MP4 视频输入。
- **Web 预览**：集成 FastAPI，支持浏览器实时预览检测结果。

## 目录结构
- `lib/`：预留目录，可存放 C++ 依赖库。
- `model/`：存放针对 RK3576 转换的 `.rknn` 模型。
- `py_utils/`：包含 OBB 后处理、NMS 及图像预处理工具。
- `web_detection.py`：主程序（支持 Web 预览与 API）。
- `requirements.txt`：Python 依赖列表。

## 快速开始

### 1. 准备模型
请将训练并转换好的 YOLOv8-OBB `.rknn` 模型放入 `model/` 目录。
> 注意：模型输入尺寸默认为 640x640。如需修改，请调整 `web_detection.py` 中的 `IMG_SIZE` 常量。

### 2. 安装依赖
本项目额外依赖 `shapely` 库用于计算多边形 IoU。

```bash
pip install -r requirements.txt
```

### 3. 运行项目

**基本运行（使用摄像头）：**
```bash
python web_detection.py --model_path model/yolov8n-obb.rknn --camera_id 0
```

**检测视频文件：**
```bash
python web_detection.py --model_path model/yolov8n-obb.rknn --video_path test.mp4
```

**指定自定义类别文件：**
```bash
python web_detection.py --model_path model/custom.rknn --class_path class_config.txt
```

访问方式：打开浏览器访问 `http://<开发板IP>:8000`

---

## 🔌 API 接口文档

本项目提供了 RESTful 接口，支持通过 HTTP POST 请求获取旋转框检测结果。

### 1. 模型推理接口 (Predict)

**Endpoint:** `POST /api/models/yolo_obb/predict`

#### 请求参数 (Multipart/Form-Data):
- `file`: (可选) 待检测的图片文件。
- `video`: (可选) 待检测的 MP4 视频文件。
- `realtime`: (可选) 布尔值。若为 `true`，则返回摄像头当前帧的检测结果。
- `conf`: (可选) 置信度阈值。
- `iou`: (可选) NMS IoU 阈值。

#### 响应示例 (JSON):
```json
{
  "success": true,
  "source": "realtime camera frame",
  "predictions": [
    {
      "class": "plane",
      "confidence": 0.92,
      "poly": [[100, 100], [200, 100], [200, 200], [100, 200]],
      "angle": 1.57
    }
  ],
  "image": { "width": 1280, "height": 720 }
}
```
* `poly`: 包含 4 个顶点坐标 `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`。
* `angle`: 旋转角度（弧度）。

### 2. 实时视频流 (Video Feed)
**Endpoint:** `GET /api/video_feed`
可直接嵌入 HTML：`<img src="http://<IP>:8000/api/video_feed">`

## 开发说明
- **OBB 后处理**：核心逻辑位于 `py_utils/obb_utils.py`，实现了旋转框解码与多边形 NMS。
- **坐标还原**：程序会自动处理 Letterbox 填充带来的坐标偏移，确保 API 返回的坐标对应原始图像分辨率。
