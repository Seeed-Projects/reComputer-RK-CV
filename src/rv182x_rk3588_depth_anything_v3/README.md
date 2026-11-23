# Depth Anything V3 Base · RK1820 / RK3588

English | [中文](README_zh.md)

Run **Depth Anything V3 Base RKNN3** on an RK1820 M.2 accelerator connected to an RK3588 host over PCIe. This standalone Docker application provides a web preview, image/video-frame inference, video upload, offline processing and result download.

## Deployment status

Validated on RK3588 + RK1820 M.2. GitHub Actions publishes ARM64 images to GHCR; check the repository Actions for publication status.

- Host: RK3588, Linux aarch64, Docker / Compose.
- Accelerator: RK1820 M.2, PCIe, firmware / RKNN3 Lite **1.1.0**.
- Quantization: **INT8 W8A8**, `do_quantization=True`, no manually assigned W16A16 subgraphs.
- Variants: `da3-base`.
- Sample video: `video/test.mp4`, supplied `30_1080p.mp4` (traffic).

## Directory layout

```text
rv182x_rk3588_depth_anything_v3/
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

## Optional manual host setup (legacy images / native host use)

```bash
cd src/rv182x_rk3588_depth_anything_v3
sudo apt-get install -y build-essential linux-headers-$(uname -r)
sudo bash scripts/setup_host.sh
```

Use `--flash-m2` only when the matched M.2 firmware needs updating. Normal startup does not require flashing. Inference uses the PCIe accelerator through the host RKNN3 proxy.

## Quick deployment with GHCR

Prerequisites: Docker on Linux ARM64 RK3588 and a correctly installed RK1820 M.2 PCIe card. The shared runtime now prepares/checks the driver and proxy, waits for Online / OK and performs real W8A8 self-test inference before starting this application. Verified kernel: `6.1.172-vendor-seeed-rk3588`; other kernels need matching headers and a working build environment.

#### Step A: Pull the image

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-depth-anything-v3:latest
```

#### Step B: One-command run

Stop other demos using port 8000 or the accelerator first. No host project directory is required. Workspace persistence is optional; download results before removing the container if no volume is mounted.

```bash
sudo docker run --rm --name rv182x-rk3588-depth-anything-v3 \
  --privileged --net=host --ipc=host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-depth-anything-v3:latest \
  python3 web_detection.py --variant da3-base --video video/test.mp4
```

Open `http://<Board_IP>:8000`; API documentation: `http://<Board_IP>:8000/docs`. Press Ctrl+C to stop the foreground container; add `-d` after `run` for background operation.

Available variants: `da3-base`. Recreate the container after changing `--variant`; for single-file models, update the paired `--model_path` / `--weight_path` too.

Bundled samples need no camera or DRM device mappings. Host IPC is required for RKNN3 proxy / DMA shared memory; inference runs on RK1820. For a camera, add the matching `--device /dev/video0:/dev/video0` and replace `--video video/test.mp4` with `--camera_id 0`.

Local, Global and Head are loaded automatically. COCO class files and detection thresholds do not apply to depth estimation.

## Build the image

From the repository root:

```bash
docker build --network host --platform linux/arm64 \
  -f docker/rv182x-rk3588/depth-anything-v3.dockerfile \
  -t rv182x-rk3588-depth-anything-v3:local src/rv182x_rk3588_depth_anything_v3
```

Configure regional package mirrors when building the shared base, not the model layer; see the runtime instructions below.

## One-command run

```bash
docker run -d --name rv182x-rk3588-depth-anything-v3 \
  --privileged --network host --ipc host --restart unless-stopped --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e MODEL_VARIANT=da3-base \
  rv182x-rk3588-depth-anything-v3:local
```

Open **http://192.168.100.145:8000** (or your device IP).

Select a listed size using `MODEL_VARIANT` and recreate the container. Demonstrations share port 8000; stop the current demonstration before starting another.

```bash
cd src/rv182x_rk3588_depth_anything_v3
MODEL_VARIANT=da3-base docker compose up -d --build
docker compose logs -f
docker compose down
```

`--privileged`, host networking and host IPC support the RKNN3 proxy, PCIe DMA and shared memory. Startup loads the model; check health before using inference.

## Depth output and input specification

Uses the official DA3-BASE example, fixed to one view at 280×280. The original image is resized directly to the square input without padding, matching the official C++ example. Local, Global and Head all use W8A8, with floating-point tokens/features between partitions. Each frame runs all three sequentially; Runtime FPS includes all three synchronous calls, PCIe transfer and inter-stage handling. Outputs are relative depth/confidence, without camera intrinsics, extrinsics or metric distance. Per-frame 2%–98% inverse-depth quantiles color near regions red and far regions blue; colors across frames do not represent a shared absolute distance scale. The API returns a Base64 PNG in visualization_png, depth range and confidence_mean.

## API usage

Interactive API documentation: `/docs`.

```bash
curl http://192.168.100.145:8000/api/health
curl -X POST http://192.168.100.145:8000/api/models/depth-anything-v3/predict \
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
| da3-base | W8A8 | 1×280×280×3 | 2.92 | 2.56 | 369.29 | 5/30 | [JSON](validation/benchmark-da3-base.json) |

Conditions: RKNN3 Lite / firmware 1.1.0, core mask `0x1`, one serial process, other demonstration containers stopped. Runtime FPS = completed frames / accumulated synchronous Lite call time, including PCIe transfer and synchronization. Pipeline FPS includes video decoding, preprocessing, inference, task decoding, drawing and JPEG encoding up to 1280 pixels wide; excludes HTTP, network and source-FPS pacing. Live playback also depends on source FPS and other processing overhead.

```bash
mkdir -p validation
docker run --rm --privileged --network host --ipc host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v "$PWD/validation:/app/validation" rv182x-rk3588-depth-anything-v3:local \
  python benchmark.py --variant da3-base --warmup 5 --frames 30 \
  --output validation/benchmark-da3-base.json
```

A speed test is not a dataset accuracy evaluation. Traffic video suits detection and vehicle segmentation; ImageNet classes are not road-specific labels, and clear close-up people are needed for meaningful pose visualization.

## Model provenance and conversion

- [RKNN3 official example](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/depth_anything_v3)
- Per-variant conversion records: `conversion/*.json`.

See [conversion/README.md](conversion/README.md) for reproducible conversion commands and calibration data.
- Model and weight files must be a matched pair. Do not mix RKNN2/RKNN3 artifacts or use an unmatched 1.2 runtime with 1.1 firmware.
- Labels: `model/labels.txt`.
- Checksums: `sha256sum -c SHA256SUMS`.

## Troubleshooting

On failed health checks, inspect `docker logs`, the host driver, card Online state and proxy. Verify paired model/weight files and that the selected variant has been generated. On overlayroot hosts, Docker and containerd data must reside on a writable ext4 filesystem supporting overlay2.

## Depth visual comparison

The fixed 280×280 single-view INT8 result has visible banding and limited detail for distant small objects. Functional/performance checks passed; depth dataset accuracy is not certified.

[Native INT8 preview](validation/benchmark-da3-base.jpg) · [FP32 ONNX reference, same frame](validation/fp32-reference-frame34.jpg)

## Shared runtime and automatic initialization

All model images inherit `ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0`. CI publishes this layer first, then builds models against its immutable digest. It includes firmware, SDK tools, driver source / verified profile and a self-test model, with no startup model/firmware downloads.

Startup: verify assets → detect one RK1820 → load/build the driver → reuse/start proxy → wait for Online / OK → run the bundled MobileNetV2 W8A8 sample → start this model. Pass `doctor` as the image command for read-only diagnostics.

- Share `/run/lock` for cross-container exclusion; stop the previous model before switching. One RK1820 card per host is supported.
- Mount `/lib/modules` and `/usr/src` read-only for kernel build information. The validated board kernel has a bundled module. Missing headers, ABI mismatch or build failures are reported; forced module loading is never used.
- Kernel modules affect the host; `--privileged` is required. The entrypoint creates the matching container device node after loading the module; mounting all of host `/dev` is unnecessary.
- Only a proxy started by this entrypoint is cleaned up. Existing external proxies are reused without being killed. A container-owned proxy lives for the application's lifetime.
- Firmware flashing is disabled by default. Driver errors and Online self-test/model failures do not trigger flashing.
- Upload, analysis and download work without a workspace mount. Add `-v "$PWD/workspace:/app/workspace"` only for persistence after container removal.

Only after confirming the M.2 board and stopping all card users and external proxies may an operator add `-e RK182X_ALLOW_FIRMWARE_RECOVERY=1 -e RK182X_BOARD=ext-m2`. An Offline card then gets at most one verified recovery attempt. Healthy cards skip recovery. Shared locking is mandatory. Keep the 300-second stop timeout and do not interrupt power or forcibly stop firmware writes.

Build the runtime locally from the repository root:

```bash
docker build --platform linux/arm64 -f docker/rv182x-rk3588/runtime.dockerfile \
  -t ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0 \
  runtime/rv182x-rk3588
```

`APT_MIRROR` / `PIP_INDEX_URL` configure the base build. Select it in model builds with `--build-arg RK182X_BASE_IMAGE=<base reference>`. Ordinary users pull only the model image; Docker fetches shared layers automatically.

The benchmark table retains the previous native measurements for all variants. This bootstrap change does not constitute a new full-model benchmark run; initialization / self-test time is outside Runtime FPS. This release's board acceptance is intentionally limited to one model's end-to-end deployment.

See the [shared runtime guide](../../../runtime/rv182x-rk3588/README.md) for the complete procedure and limits.
