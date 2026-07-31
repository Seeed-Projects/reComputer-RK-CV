# MobileNet on RK3588

[中文](README_zh.md)

This project packages the RKNN Model Zoo MobileNet example as a standalone
RK3588 web application. It provides a browser preview, an OpenAPI-compatible
REST API, image/video analysis, and a one-command Docker deployment.

## Run with Docker

Build from the repository root:

```bash
docker build \
  -f docker/rk3588/mobilenet.dockerfile \
  -t recomputer-rk3588-mobilenet \
  src/rk3588_mobilenet
```

Run the bundled sample video:

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3588-mobilenet
```

Open `http://<device-ip>:8000`. To start in upload-only mode, override the
command:

```bash
sudo docker run --rm --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  recomputer-rk3588-mobilenet \
  python web_classification.py --camera_id -1
```

## API

Interactive documentation is available at `/docs`.

- `GET /api/health`: service and model readiness
- `GET|POST /api/config`: get or update `conf_thresh`
- `POST /api/models/mobilenet/predict`: classify an image, video frame, or the
  current realtime frame
- `GET /api/video_feed`: MJPEG preview
- `POST /api/video/upload`, `POST /api/video/analyze`,
  `GET /api/video/status`: asynchronous file analysis

Image request:

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilenet/predict \
  -F file=@src/rk3588_mobilenet/model/bell.jpg \
  -F topk=5 \
  -F conf=0.0
```

Video-frame request:

```bash
curl -X POST http://127.0.0.1:8000/api/models/mobilenet/predict \
  -F video=@src/rk3588_mobilenet/video/test.mp4 \
  -F timestamp=2.5
```

## Native run

Use Python 3.11 on an RK3588 device:

```bash
pip install -r requirements.txt
pip install rknn-toolkit-lite2-packages/*.whl
python web_classification.py --video_path video/test.mp4
```

The runtime assets are intentionally limited to the RK3588 model, label file,
sample media, RKNN Lite wheel, and RK3588 runtime library.
