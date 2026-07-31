# RK3576 ResNet50V2 图像分类

[English](README.md)

本项目将 RKNN Model Zoo 的 ResNet50V2 示例封装为独立的 RK3576 Web 服务，
提供浏览器预览、REST/OpenAPI 推理、异步文件分析和 Docker 一键部署。

## Docker 运行

在仓库根目录执行：

```bash
docker build \
  -f docker/rk3576/resnet50v2.dockerfile \
  -t recomputer-rk3576-resnet50v2 \
  src/rk3576_resnet50v2

sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-resnet50v2
```

浏览器访问 `http://<开发板IP>:8000`。只启动网页上传和 API：

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-resnet50v2 \
  python web_classification.py --camera_id -1
```

## API

启动后可通过 `/docs` 使用 OpenAPI 交互文档。

- `GET /api/health`
- `GET|POST /api/config`
- `POST /api/models/resnet50v2/predict`
- `GET /api/video_feed`
- `POST /api/video/upload`、`POST /api/video/analyze`、
  `GET /api/video/status`

```bash
curl -X POST http://127.0.0.1:8000/api/models/resnet50v2/predict \
  -F file=@src/rk3576_resnet50v2/model/dog_224x224.jpg \
  -F topk=5 \
  -F conf=0.0
```

## 模型转换

运行模型来自：

- ONNX：`rknn_model_zoo/examples/resnet/model/resnet50-v2-7.onnx`
- 转换脚本：`rknn_model_zoo/examples/resnet/python/resnet.py`

在安装 RKNN-Toolkit2 的 x86 转换环境中执行：

```bash
cd rknn_model_zoo/examples/resnet/python
python3 resnet.py \
  ../model/resnet50-v2-7.onnx \
  rk3576 \
  i8 \
  ../model/rk3576_resnet50-v2-7.rknn
```

脚本会设置 ImageNet mean/std 和 INT8 量化。转换前需确保脚本中的校准数据集
路径真实存在。
