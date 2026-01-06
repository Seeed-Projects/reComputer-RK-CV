# RKNN Toolkit2 Lite 实时目标检测

Please run: 
`pip install rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl` to install RKNN Toolkit2 Lite on the target device.

### 安装依赖
`pip install -r requirements.txt` 

```bash
# 基本用法
python realtime_detection.py --model_path your_model.rknn

# 指定摄像头（如果video1不可用）
python realtime_detection.py --model_path your_model.rknn --camera_id 0

# 设置目标FPS
python realtime_detection.py --model_path your_model.rknn --fps 25

# 使用本地视频文件
python realtime_detection.py --model_path your_model.rknn --video_path video/test.mp4
```

### Docker 运行 (GitHub Workflow)

本项目支持通过 Docker 快速运行。

1. **拉取镜像**：
```bash
docker pull ghcr.io/<your-username>/<repo-name>:latest
```

2. **运行容器**：

> 注意：为了显示视频窗口，需要配置 X11 转发。

**Linux (支持 X11):**
```bash
xhost +local:docker
docker run --rm --net=host --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/video0:/dev/video0 \
    -v $(pwd)/model:/app/model \
    ghcr.io/<your-username>/<repo-name>:latest
```

**仅运行推理 (无显示窗口):**
如果不需要显示窗口，可以使用以下命令（需修改代码适配 headless 模式）：
```bash
docker run --rm --device /dev/video0:/dev/video0 -v $(pwd)/model:/app/model ghcr.io/<your-username>/<repo-name>:latest
```
