# RK3588 ResNet50V2 图像分类

[English](README.md)

本项目将 RKNN Model Zoo 的 ResNet50V2 示例封装为独立的 RK3588 Web 应用，
提供浏览器预览、标准 REST API、图片/视频分析和 Docker 一键运行能力。

## Docker 一键运行

在仓库根目录构建：

```bash
docker build \
  -f docker/rk3588/resnet50v2.dockerfile \
  -t recomputer-rk3588-resnet50v2 \
  src/rk3588_resnet50v2
```

运行内置测试视频：

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3588-resnet50v2
```

浏览器访问 `http://<开发板IP>:8000`。如果只需要网页上传和 API：

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3588-resnet50v2 \
  python web_classification.py --camera_id -1
```

## API

启动后可访问 `/docs` 查看 OpenAPI 交互文档。

- `GET /api/health`：服务及模型就绪状态
- `GET|POST /api/config`：读取或更新 `conf_thresh`
- `POST /api/models/resnet50v2/predict`：对图片、视频指定帧或实时帧分类
- `GET /api/video_feed`：MJPEG 实时预览
- `POST /api/video/upload`、`POST /api/video/analyze`、
  `GET /api/video/status`：异步文件分析

图片分类示例：

```bash
curl -X POST http://127.0.0.1:8000/api/models/resnet50v2/predict \
  -F file=@src/rk3588_resnet50v2/model/dog_224x224.jpg \
  -F topk=5 \
  -F conf=0.0
```

## 开发板原生运行

在 RK3588 的 Python 3.11 环境中：

```bash
pip install -r requirements.txt
pip install rknn-toolkit-lite2-packages/*.whl
python web_classification.py --video_path video/test.mp4
```

本目录只包含 RK3588 模型。源案例中的 RK3576 模型需要与 RK3576 runtime
library 一起迁移到独立的 `rk3576_resnet50v2` 项目，不能混用。
