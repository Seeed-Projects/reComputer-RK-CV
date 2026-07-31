# ResNet50V2 on RK3576

[中文](README_zh.md)

This project packages the RKNN Model Zoo ResNet50V2 example as an independent
RK3576 web service with browser preview, REST/OpenAPI inference, asynchronous
file analysis, and Docker deployment.

## Docker

From the repository root:

```bash
docker build \
  -f docker/rk3576/resnet50v2.dockerfile \
  -t recomputer-rk3576-resnet50v2 \
  src/rk3576_resnet50v2

sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-resnet50v2
```

Open `http://<device-ip>:8000`. For upload/API-only mode:

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-resnet50v2 \
  python web_classification.py --camera_id -1
```

## API

Interactive OpenAPI documentation is available at `/docs`.

- `GET /api/health`
- `GET|POST /api/config`
- `POST /api/models/resnet50v2/predict`
- `GET /api/video_feed`
- `POST /api/video/upload`, `POST /api/video/analyze`,
  `GET /api/video/status`

```bash
curl -X POST http://127.0.0.1:8000/api/models/resnet50v2/predict \
  -F file=@src/rk3576_resnet50v2/model/dog_224x224.jpg \
  -F topk=5 \
  -F conf=0.0
```

## Model conversion

The runtime model was generated from:

- ONNX: `rknn_model_zoo/examples/resnet/model/resnet50-v2-7.onnx`
- Script: `rknn_model_zoo/examples/resnet/python/resnet.py`

Run the conversion in an RKNN-Toolkit2 x86 environment:

```bash
cd rknn_model_zoo/examples/resnet/python
python3 resnet.py \
  ../model/resnet50-v2-7.onnx \
  rk3576 \
  i8 \
  ../model/rk3576_resnet50-v2-7.rknn
```

The conversion script configures ImageNet mean/std normalization and INT8
quantization. Its calibration dataset path must exist before conversion.
