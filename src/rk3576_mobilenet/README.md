# MobileNet on RK3576

[中文](README_zh.md)

This project packages the RKNN Model Zoo MobileNet example as an independent
RK3576 web service with browser preview, REST/OpenAPI inference, asynchronous
file analysis, and Docker deployment.

## Docker

From the repository root:

```bash
docker build \
  -f docker/rk3576/mobilenet.dockerfile \
  -t recomputer-rk3576-mobilenet \
  src/rk3576_mobilenet

sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-mobilenet
```

Open `http://<device-ip>:8000`. For upload/API-only mode:

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3576-mobilenet \
  python web_classification.py --camera_id -1
```

## API

Interactive OpenAPI documentation is available at `/docs`.

- `GET /api/health`
- `GET|POST /api/config`
- `POST /api/models/mobilenet/predict`
- `GET /api/video_feed`
- `POST /api/video/upload`, `POST /api/video/analyze`,
  `GET /api/video/status`

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilenet/predict \
  -F file=@src/rk3576_mobilenet/model/bell.jpg \
  -F topk=5 \
  -F conf=0.0
```

## Model conversion

The runtime model was generated from:

- ONNX: `rknn_model_zoo/examples/mobilenet/model/mobilenetv2-12.onnx`
- Script: `rknn_model_zoo/examples/mobilenet/python/mobilenet.py`

Run the conversion in an RKNN-Toolkit2 x86 environment:

```bash
cd rknn_model_zoo/examples/mobilenet/python
python3 mobilenet.py \
  --target rk3576 \
  --model ../model/mobilenetv2-12.onnx \
  --output_path ../model/rk3576_mobilenet_v2.rknn
```

The conversion script configures ImageNet mean/std normalization and INT8
quantization. Its calibration dataset path must exist before conversion.
