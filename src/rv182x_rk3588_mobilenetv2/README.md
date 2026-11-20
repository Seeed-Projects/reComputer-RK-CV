# MobileNetV2 · RK1820 / RK3588

English | [中文](README_zh.md)

Run **MobileNetV2 RKNN3** on an RK1820 M.2 accelerator connected to an RK3588 host over PCIe. This standalone Docker application provides a web preview, image/video-frame inference, video upload, offline processing and result download.

## Deployment status

Validated on RK3588 + RK1820 M.2. GitHub Actions publishes ARM64 images to GHCR; check the repository Actions for publication status.

- Host: RK3588, Linux aarch64, Docker / Compose.
- Accelerator: RK1820 M.2, PCIe, firmware / RKNN3 Lite **1.1.0**.
- Quantization: **INT8 W8A8**, `do_quantization=True`, no manually assigned W16A16 subgraphs.
- Variants: `mobilenetv2-12`.
- Sample video: `video/test.mp4`, supplied `30_1080p.mp4` (traffic).

## Directory layout

```text
rv182x_rk3588_mobilenetv2/
├── web_detection.py        # FastAPI + serialized preview/offline workers
├── engine.py               # RKNN3 inference + task decoding/rendering
├── index.html              # Live preview + local video analysis
├── project.json            # Variants, task and artifact metadata
├── model/                  # Paired .rknn/.weight files and labels
├── video/test.mp4
├── lib/                    # Matched ARM64 runtime libraries
├── rknn3-toolkit-lite-packages/
├── host/                   # Driver / proxy / matching M.2 firmware
├── scripts/setup_host.sh
├── conversion/             # Conversion config and provenance
├── benchmark.py
├── validation/             # Measured JSON + preview images
└── compose.yaml
```

## Prepare the RK3588 host

```bash
cd src/rv182x_rk3588_mobilenetv2
sudo apt-get install -y build-essential linux-headers-$(uname -r)
sudo bash scripts/setup_host.sh
```

Use `--flash-m2` only when the matched M.2 firmware needs updating. Normal startup does not require flashing. Inference uses the PCIe accelerator through the host RKNN3 proxy.

## Quick deployment with GHCR

Prerequisites: Docker on Linux ARM64 RK3588, an Online RK1820 M.2 PCIe card, and the matching SDK 1.1.0 host driver / RKNN3 proxy installed using the host preparation below.

#### Step A: Pull the image

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-mobilenetv2:latest
```

#### Step B: One-command run

Stop other demos using port 8000 or the accelerator first. Run from a writable directory; `workspace` stores uploaded files and analysis results.

```bash
sudo docker run --rm --name rv182x-rk3588-mobilenetv2 \
  --privileged --net=host --ipc=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  -v "$(pwd)/workspace:/app/workspace" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-mobilenetv2:latest \
  python3 web_detection.py --variant mobilenetv2-12 --model_path model/mobilenetv2-12.rknn --weight_path model/mobilenetv2-12.weight --video video/test.mp4
```

Open `http://<Board_IP>:8000`; API documentation: `http://<Board_IP>:8000/docs`. Press Ctrl+C to stop the foreground container; add `-d` after `run` for background operation.

Available variants: `mobilenetv2-12`. Recreate the container after changing `--variant`; for single-file models, update the paired `--model_path` / `--weight_path` too.

Bundled samples need no camera or DRM device mappings. Host IPC is required for RKNN3 proxy / DMA shared memory; inference runs on RK1820. For a camera, add the matching `--device /dev/video0:/dev/video0` and replace `--video video/test.mp4` with `--camera_id 0`.

#### Custom display labels

Defaults: 1000 ImageNet classes. Create one name per line in `class_config.txt`, preserving the original class order and count. Renaming labels does not change recognition capabilities; new classes require a trained, converted model.

```bash
sudo docker run --rm --name rv182x-rk3588-mobilenetv2 \
  --privileged --net=host --ipc=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  -v "$(pwd)/class_config.txt:/app/class_config.txt:ro" \
  -v "$(pwd)/workspace:/app/workspace" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-mobilenetv2:latest \
  python3 web_detection.py --variant mobilenetv2-12 --model_path model/mobilenetv2-12.rknn --weight_path model/mobilenetv2-12.weight --video video/test.mp4 --class_path /app/class_config.txt
```

## Build the image

From the repository root:

```bash
docker build --network host --platform linux/arm64 \
  -f docker/rv182x-rk3588/mobilenetv2.dockerfile \
  -t rv182x-rk3588-mobilenetv2:local src/rv182x_rk3588_mobilenetv2
```

For a regional package mirror, add `--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn`.

## One-command run

```bash
docker run -d --name rv182x-rk3588-mobilenetv2 \
  --privileged --network host --ipc host --restart unless-stopped \
  -e MODEL_VARIANT=mobilenetv2-12 \
  -v "$PWD/workspace:/app/workspace" \
  rv182x-rk3588-mobilenetv2:local
```

Open **http://192.168.100.145:8000** (or your device IP).

Select a listed size using `MODEL_VARIANT` and recreate the container. Demonstrations share port 8000; stop the current demonstration before starting another.

```bash
cd src/rv182x_rk3588_mobilenetv2
MODEL_VARIANT=mobilenetv2-12 docker compose up -d --build
docker compose logs -f
docker compose down
```

`--privileged`, host networking and host IPC support the RKNN3 proxy, PCIe DMA and shared memory. Startup loads the model; check health before using inference.

## Labels and custom models

The official model has 1000 classes. To rename display labels, replace model/synset.txt, one label per line with the original count/order, or use --class_path. Renaming does not retrain the model. A replacement of the same architecture requires a matching .rknn/.weight pair; use --model_path and --weight_path.

## API usage

Interactive API documentation: `/docs`.

```bash
curl http://192.168.100.145:8000/api/health
curl -X POST http://192.168.100.145:8000/api/models/mobilenetv2/predict \
  -F file=@sample.jpg -F conf=0.25 -F iou=0.45
curl -X POST http://192.168.100.145:8000/api/predict \
  -F video=@sample.mp4 -F timestamp=1.0
curl -X POST http://192.168.100.145:8000/api/config \
  -H 'Content-Type: application/json' -d '{"obj_thresh":0.25,"nms_thresh":0.45}'
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Model, runtime, call count and state |
| `GET/POST /api/config` | Detection thresholds, range 0–1 |
| `POST /api/predict` | Image, timestamped video frame or current preview; JSON results |
| `GET /api/video_feed` | MJPEG preview |
| `GET /api/snapshot` | JPEG snapshot |
| `POST /api/video/upload` | multipart `file` |
| `GET /api/video/list` | Input/output filenames |
| `POST /api/video/analyze` | form `filename` |
| `GET /api/video/status` | Progress and errors |
| `GET /api/video/download/{filename}` | Download completed video |

Provide either `file` or `video`; otherwise the current preview frame is used. Coordinates refer to the original image. Classification returns Top-5; segmentation adds `segments` polygons; pose adds 17 `keypoints` (x, y, confidence). Classification does not use detection thresholds. Invalid parameters: 422; invalid media: 400; conflicting jobs: 409; unavailable: 503.

## On-device benchmark

Only successful measurements from this RK3588 + RK1820 M.2 are listed. `—` means no passing measurement is available.

| Variant | Quant | NHWC input | Runtime FPS | Pipeline FPS | Runtime p95 ms | Warmup / frames | Evidence / status |
|---|---|---|---:|---:|---:|---:|---|
| mobilenetv2-12 | W8A8 | 1×224×224×3 | 196.95 | 37.42 | 5.63 | 5/30 | [JSON](validation/benchmark-mobilenetv2-12.json) |

Conditions: RKNN3 Lite / firmware 1.1.0, core mask `0x1`, one serial process, other demonstration containers stopped. Runtime FPS = completed frames / accumulated synchronous Lite call time, including PCIe transfer and synchronization. Pipeline FPS includes video decoding, preprocessing, inference, task decoding, drawing and JPEG encoding up to 1280 pixels wide; excludes HTTP, network and source-FPS pacing. Live playback also depends on source FPS and other processing overhead.

```bash
mkdir -p validation
docker run --rm --privileged --network host --ipc host \
  -v "$PWD/validation:/app/validation" rv182x-rk3588-mobilenetv2:local \
  python benchmark.py --variant mobilenetv2-12 --warmup 5 --frames 30 \
  --output validation/benchmark-mobilenetv2-12.json
```

A speed test is not a dataset accuracy evaluation. Traffic video suits detection and vehicle segmentation; ImageNet classes are not road-specific labels, and clear close-up people are needed for meaningful pose visualization.

## Model provenance and conversion

- [RKNN3 official example](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/mobilenet_v2)
- Per-variant conversion records: `conversion/*.json`.

See [conversion/README.md](conversion/README.md) for reproducible conversion commands and calibration data.
- Model and weight files must be a matched pair. Do not mix RKNN2/RKNN3 artifacts or use an unmatched 1.2 runtime with 1.1 firmware.
- Labels: `model/synset.txt`.
- Checksums: `sha256sum -c SHA256SUMS`.

## Troubleshooting

On failed health checks, inspect `docker logs`, the host driver, card Online state and proxy. Verify paired model/weight files and that the selected variant has been generated. On overlayroot hosts, Docker and containerd data must reside on a writable ext4 filesystem supporting overlay2.
