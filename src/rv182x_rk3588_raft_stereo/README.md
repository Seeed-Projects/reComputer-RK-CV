# RAFT-Stereo · RK1820 / RK3588

English | [中文](README_zh.md)

Stereo disparity estimation on an RK1820 M.2 accelerator connected to an RK3588 over PCIe. This standalone Docker application provides a matching web layout, paired-image upload, disparity visualization and a JSON API.

## Status and features

- Validated on RK3588 + RK1820 M.2. GitHub Actions publishes ARM64 images to GHCR; check the repository Actions for publication status.
- RKNN3 Toolkit / Lite / firmware **1.1.0**, target `rk1820`, core mask `0x1`.
- **INT8 W8A8**, `do_quantization=True`, no manually assigned W16A16 subgraphs.
- Includes all four official variants: eth3d, middlebury, sceneflow and realtime.
- Each call takes two 160×240 RGB images, resized with aspect ratio preserved and black padding on the right/bottom.
- Exports one fixed GRU iteration, matching the official exporter default. These FPS numbers are not comparable to a 32-iteration configuration; a different iteration count requires re-export and conversion.

Inputs must be a genuine synchronized, rectified left/right pair of the same scene. A monocular traffic video cannot provide stereo input. Validation uses the official xt_left.jpg / xt_right.jpg pair. The supplied traffic video is stored in video/test.mp4 but is not used for stereo validation. Road stereo-video validation requires synchronized, rectified left/right videos. This service exposes a paired-image interface. Disparity is not converted to metric distance without baseline/focal-length calibration.

## Directory layout

```text
rv182x_rk3588_raft_stereo/
├── web_detection.py       # Paired-input FastAPI service
├── special_engine.py      # RKNN3 inference + disparity visualization
├── index.html             # Left/right upload + preview
├── project.json           # Four variants and iteration count
├── model/                 # .rknn/.weight pairs + xt_left/right.jpg
├── lib/
├── rknn3-toolkit-lite-packages/
├── host/                  # SDK driver / proxy / M.2 firmware
├── scripts/setup_host.sh
├── conversion/
├── benchmark.py
├── verify_stereo.py
├── validation/
└── compose.yaml
```

## Host preparation

```bash
cd src/rv182x_rk3588_raft_stereo
sudo apt-get install -y build-essential linux-headers-$(uname -r)
sudo bash scripts/setup_host.sh
```

Normal startup does not require flashing. Use --flash-m2 only when updating matched M.2 firmware is necessary.

## Quick deployment with GHCR

Prerequisites: Docker on Linux ARM64 RK3588, an Online RK1820 M.2 PCIe card, and the matching SDK 1.1.0 host driver / RKNN3 proxy installed using the host preparation below.

#### Step A: Pull the image

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-raft-stereo:latest
```

#### Step B: One-command run

Stop other demos using port 8000 or the accelerator first. Run from a writable directory; `workspace` stores uploaded files and analysis results.

```bash
sudo docker run --rm --name rv182x-rk3588-raft-stereo \
  --privileged --net=host --ipc=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  -v "$(pwd)/workspace:/app/workspace" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-raft-stereo:latest \
  python3 web_detection.py --variant raftstereo-eth3d-160x240
```

Open `http://<Board_IP>:8000`; API documentation: `http://<Board_IP>:8000/docs`. Press Ctrl+C to stop the foreground container; add `-d` after `run` for background operation.

Available variants: `raftstereo-eth3d-160x240`, `raftstereo-middlebury-160x240`, `raftstereo-sceneflow-160x240`, `raftstereo-realtime-160x240`. Recreate the container after changing `--variant`; for single-file models, update the paired `--model_path` / `--weight_path` too.

Bundled samples need no camera or DRM device mappings. Host IPC is required for RKNN3 proxy / DMA shared memory; inference runs on RK1820. The page automatically runs the bundled stereo pair. Upload synchronized, rectified left/right images of equal dimensions; monocular video and live stereo camera capture are not supported by this service.

The four variants are independently selectable complete models. Output is disparity, without metric-distance conversion, at a fixed one iteration.

## Build the image

```bash
# Repository root
docker build --network host --platform linux/arm64 \
  -f docker/rv182x-rk3588/raft-stereo.dockerfile \
  -t rv182x-rk3588-raft-stereo:local src/rv182x_rk3588_raft_stereo
```

Optional regional mirrors: --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple.

## One-command run and variants

```bash
docker run -d --name rv182x-rk3588-raft-stereo \
  --privileged --network host --ipc host --restart unless-stopped \
  -e MODEL_VARIANT=raftstereo-eth3d-160x240 \
  rv182x-rk3588-raft-stereo:local
```

Open **http://192.168.100.145:8000** (or your device IP).

Available variants: `raftstereo-eth3d-160x240`, `raftstereo-middlebury-160x240`, `raftstereo-sceneflow-160x240`, `raftstereo-realtime-160x240`.

```bash
cd src/rv182x_rk3588_raft_stereo
MODEL_VARIANT=raftstereo-realtime-160x240 docker compose up -d --build
docker compose logs -f
docker compose down
```

Recreate the container after selecting a variant. Demonstrations share port 8000; stop the current one before switching. Privileged mode, host networking and host IPC support PCIe DMA, the host RKNN3 proxy and shared memory.

## API

```bash
curl http://192.168.100.145:8000/api/health
curl -X POST http://192.168.100.145:8000/api/models/raft-stereo/predict \
  -F left=@left.jpg -F right=@right.jpg
curl http://192.168.100.145:8000/api/sample -o disparity.jpg
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Variant, iterations and call count |
| `POST /api/predict` | Alias for model-specific prediction; multipart left/right |
| `POST /api/models/raft-stereo/predict` | Paired-image inference; dimensions must match |
| `GET /api/sample` | Run official sample; return a JPEG disparity preview |
| `GET /docs` | OpenAPI |

Prediction returns predictions[0].visualization_png (Base64 PNG), disparity minimum/maximum, valid_fraction, and timing.runtime_ms / total_ms. A missing input returns 422; invalid media or mismatched dimensions returns 400.

## On-device benchmark

Uses the official stereo pair, two NHWC 1×160×240×3 inputs, core mask 0x1, one iteration per call, with other inference demonstrations stopped. A frame represents one completed stereo-pair inference.

| Variant | Quant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Warmup / pairs | Evidence |
|---|---|---:|---:|---:|---:|---|
| raftstereo-eth3d-160x240 | W8A8 | 4.75 | 1.84 | 211.66 | 5/30 | [JSON](validation/benchmark-raftstereo-eth3d-160x240.json) |
| raftstereo-middlebury-160x240 | W8A8 | 4.76 | 1.87 | 210.81 | 5/30 | [JSON](validation/benchmark-raftstereo-middlebury-160x240.json) |
| raftstereo-sceneflow-160x240 | W8A8 | 4.77 | 1.86 | 210.28 | 5/30 | [JSON](validation/benchmark-raftstereo-sceneflow-160x240.json) |
| raftstereo-realtime-160x240 | W8A8 | 16.48 | 2.56 | 61.58 | 5/30 | [JSON](validation/benchmark-raftstereo-realtime-160x240.json) |

Runtime FPS includes synchronous Lite calls, PCIe transfer and synchronization. Pipeline FPS additionally includes reading/decoding both JPEGs, resize/padding, disparity coloring and JPEG encoding up to 1280 pixels wide. HTTP/network are excluded. FPS is completed pairs divided by accumulated measured time. A variant without a successful record shows —.

```bash
mkdir -p validation
docker run --rm --privileged --network host --ipc host \
  -v "$PWD/validation:/app/validation" rv182x-rk3588-raft-stereo:local \
  python benchmark.py --variant raftstereo-eth3d-160x240 \
  --warmup 5 --frames 30 --output validation/benchmark-raftstereo-eth3d-160x240.json
```

## Conversion and provenance

See conversion/README.md. Conversion uses x86 RKNN3 Toolkit 1.1.0, fixed 160×240 inputs and iters=1 for all four checkpoints. The official converter maps its i8 option to W8A16; this project explicitly uses W8A8 instead.

- [RKNN3 official RAFT-Stereo example](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/RAFT-Stereo)
- Per-variant configuration and ONNX checksums: `conversion/*.json`.
- Model/weight pairs must match. Verify runtime files: `sha256sum -c SHA256SUMS`.

## Troubleshooting

Check docker logs, the host driver, card Online state and RKNN3 proxy. Images must have matching dimensions and stereo rectification. The color map does not replace ground-truth error evaluation. On overlayroot hosts, place Docker/containerd data on writable ext4 storage.
