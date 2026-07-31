# RK3588 MobileNet 图像分类

[English](README.md)

本项目将 RKNN Model Zoo 的 MobileNet 示例封装为独立的 RK3588 Web 应用，
提供浏览器预览、标准 REST API、图片/视频分析和 Docker 一键运行能力。

## Docker 一键运行

在仓库根目录构建：

```bash
docker build \
  -f docker/rk3588/mobilenet.dockerfile \
  -t recomputer-rk3588-mobilenet \
  src/rk3588_mobilenet
```

运行内置测试视频：

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3588-mobilenet
```

浏览器访问 `http://<开发板IP>:8000`。如果只需要网页上传和 API，不启动视频源：

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3588-mobilenet \
  python web_classification.py --camera_id -1
```

## API

启动后可访问 `/docs` 查看 OpenAPI 交互文档。

- `GET /api/health`：服务及模型就绪状态
- `GET|POST /api/config`：读取或更新 `conf_thresh`
- `POST /api/models/mobilenet/predict`：对图片、视频指定帧或实时帧分类
- `GET /api/video_feed`：MJPEG 实时预览
- `POST /api/video/upload`、`POST /api/video/analyze`、
  `GET /api/video/status`：异步文件分析

图片分类示例：

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilenet/predict \
  -F file=@src/rk3588_mobilenet/model/bell.jpg \
  -F topk=5 \
  -F conf=0.0
```

视频指定帧分类示例：

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilenet/predict \
  -F video=@src/rk3588_mobilenet/video/test.mp4 \
  -F timestamp=2.5
```

## 开发板原生运行

在 RK3588 的 Python 3.11 环境中：

```bash
pip install -r requirements.txt
pip install rknn-toolkit-lite2-packages/*.whl
python web_classification.py --video_path video/test.mp4
```

运行目录仅保留部署所需的 RK3588 模型、标签、示例媒体、RKNN Lite wheel
和 RK3588 runtime library，不包含模型转换期的 ONNX 文件。
