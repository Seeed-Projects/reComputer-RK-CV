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
    --device /dev/video0:/dev/video0 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    ghcr.io/litxaohu/recomputer-rk-cv/rk3588-yolo:latest
    python realtime_detection.py --model_path model/yolo11n.rknn --camera_id 0
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

---

## 平台详细文档

- [RK3588 使用指南](src/rk3588/README.md)
- [RK3576 使用指南](src/rk3576/README.md)

## 自动化构建

本项目支持通过 GitHub Actions 自动构建多平台镜像。
- 当修改 `src/rk3588/` 目录时，会自动触发 `rk3588-yolo` 镜像的构建。
- 当修改 `src/rk3576/` 目录时，会自动触发 `rk3576-yolo` 镜像的构建。
- 支持手动触发构建，并可指定 `image_tag`。

## 💻 二次开发指南
### 代码说明
- `realtime_detection.py`:
    - **双模支持**: 集成 FastAPI，同时支持本地渲染和 MJPEG 流式输出。
    - **环境自适应**: 自动检测 `DISPLAY` 环境变量，无环境时静默跳过 GUI 初始化。
    - **RKNN 推理**: 封装了 RKNN 初始化、加载模型、多核推理逻辑。
    - **后处理**: YOLOv11 专用的 Box 解码与 NMS 逻辑。

### 修改模型
1. 将训练好并转换完成的 .rknn 模型放入相应平台的 `model/` 目录。
2. 运行命令时可添加 `--model_path` 参数指向新模型（默认已在 Dockerfile 中配置）。
