# RK3576 MobileNet 图像分类

[English](README.md)

本项目将 RKNN Model Zoo 的 MobileNet 示例封装为独立的 RK3576 Web 服务，
提供浏览器预览、REST/OpenAPI 推理、异步文件分析和 Docker 一键部署。

## Docker 运行

在仓库根目录执行：

```bash
docker build \
  -f docker/rk3576/mobilenet.dockerfile \
  -t recomputer-rk3576-mobilenet \
  src/rk3576_mobilenet

sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-mobilenet
```

浏览器访问 `http://<开发板IP>:8000`。只启动网页上传和 API：

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-mobilenet \
  python web_classification.py --camera_id -1
```

## API

启动后可通过 `/docs` 使用 OpenAPI 交互文档。

- `GET /api/health`
- `GET|POST /api/config`
- `POST /api/models/mobilenet/predict`
- `GET /api/video_feed`
- `POST /api/video/upload`、`POST /api/video/analyze`、
  `GET /api/video/status`

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilenet/predict \
  -F file=@src/rk3576_mobilenet/model/bell.jpg \
  -F topk=5 \
  -F conf=0.0
```

## 模型转换

运行模型来自：

- ONNX：`rknn_model_zoo/examples/mobilenet/model/mobilenetv2-12.onnx`
- 转换脚本：`rknn_model_zoo/examples/mobilenet/python/mobilenet.py`

在安装 RKNN-Toolkit2 的 x86 转换环境中执行：

```bash
cd rknn_model_zoo/examples/mobilenet/python
python3 mobilenet.py \
  --target rk3576 \
  --model ../model/mobilenetv2-12.onnx \
  --output_path ../model/rk3576_mobilenet_v2.rknn
```

脚本会设置 ImageNet mean/std 和 INT8 量化。转换前需确保脚本中的校准数据集
路径真实存在。
