# rk3576_yolov8

A real-time YOLOv8 inference project based on RK3576 / RKNN / MPP / RGA / ZLMediaKit, with multi-source input and multi-stream output support.

Current features include:

- `V4L2` camera input
- `RTSP` network video input
- `localvideo` local video file loop input
- Three RTSP output streams: main stream / sub stream / AI debug stream
- `YOLOv8 RKNN` inference, bounding box overlay, class labels, timestamp and FPS OSD
- `libavformat`-preferred RTSP/local compressed stream input, with `ffmpeg pipe` as a compatibility fallback

## Feature Overview

- **Input Sources**
- `v4l2`: suitable for USB cameras and local capture
- `rtsp`: suitable for IPC / NVR / network video streams
- `localvideo`: suitable for local video playback, offline validation, and loop inference

- **Output Streams**
- Main stream: `rtsp://<board-ip>:8554/live/camera`
- Sub stream: `rtsp://<board-ip>:8554/live/camera_sub`
- AI debug stream: `rtsp://<board-ip>:8554/live/camera_ai`
- Use `--streams main,sub,ai` to enable any one, two, or all three streams

- **Resolution Strategy**
- `camera` main stream: aspect-ratio resize with padding, no stretching
- `camera_sub` sub stream: aspect-ratio resize with centered crop, no stretching
- `camera_ai` AI debug stream: aspect-ratio resize with padding, no stretching

- **AI Inference**
- Fixed input size: `640x640`
- `RGA letterbox` preprocessing is preferred
- Supports detection cache, PTS matching, and AI overlay

## Deployment Guide

### Dependencies

- RK3576 board runtime environment
- RKNN Runtime
- Rockchip MPP
- RGA / `librga`
- ZLMediaKit runtime libraries
- FFmpeg runtime libraries or command-line tools

### Build Steps

```bash
cd /home/parker/Projects/rk3576_yolov8tortsp_demo
chmod +x scripts/build-linux.sh
./scripts/build-linux.sh
```

Build artifacts:

- Executable: `install/bin/rk3576_yolov8tortsp_demo`
- Runtime library directory: `install/lib`
- Documentation directory: `install/docs`
- Model directory: `install/model`

### Pre-run Setup

```bash
cd install
export LD_LIBRARY_PATH="$(pwd)/lib:${LD_LIBRARY_PATH}"
```

### Model Placement

Place the RKNN models into:

```text
model/
  yolov8n_rk3576.rknn
  yolov8s_rk3576.rknn
  yolov8m_rk3576.rknn
```

If `--model` is not explicitly specified, the program automatically selects the first existing model in the following order:

1. `model/yolov8n_rk3576.rknn`
2. `model/yolov8s_rk3576.rknn`
3. `model/yolov8m_rk3576.rknn`

## Running the Demo

### V4L2 Camera Input

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source v4l2 \
  --device /dev/video0 \
  --model model/yolov8n_rk3576.rknn \
  --camera-width 1920 \
  --camera-height 1080 \
  --output-width 1280 \
  --output-height 720 \
  --fps 30 \
  --bitrate 4000000 \
  --perf-interval-ms 1000 \
  --rtsp-port 8554 \
  --rtsp-app live \
  --rtsp-stream camera
```

### RTSP Input

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source rtsp \
  --rtsp-input-url "rtsp://192.168.100.101:554/live" \
  --rtsp-username "admin" \
  --rtsp-password "123456" \
  --model model/yolov8n_rk3576.rknn \
  --camera-width 1920 \
  --camera-height 1080 \
  --output-width 1280 \
  --output-height 720 \
  --fps 30 \
  --bitrate 4000000 \
  --perf-interval-ms 1000 \
  --rtsp-timeout-ms 5000 \
  --rtsp-reconnect-ms 1000 \
  --rtsp-port 8554 \
  --rtsp-app live \
  --rtsp-stream camera
```

### Local Video Loop Input

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source localvideo \
  --localvideo ../video/football.mp4 \
  --model model/yolov8n_rk3576.rknn \
  --camera-width 1920 \
  --camera-height 1080 \
  --output-width 1280 \
  --output-height 720 \
  --fps 30 \
  --bitrate 4000000 \
  --perf-interval-ms 1000 \
  --rtsp-port 8554 \
  --rtsp-app live \
  --rtsp-stream camera
```

### Verbose Logging

By default, only the startup summary is printed. To enable detailed runtime logs, add:

```bash
--detail-info
```

### Dump NPU Input Images

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source localvideo \
  --localvideo ./videos/demo.mp4 \
  --model model/yolov8n_rk3576.rknn \
  --streams ai \
  --dump-ai-input-dir ./debug_ai_input \
  --dump-ai-input-every 30 \
  --detail-info
```

### Enable Output Streams as Needed

```bash
--streams ai
--streams main,ai
--streams main,sub
--streams main,sub,ai
```

Notes:

- `--streams ai`: only run AI inference and output `camera_ai`
- `--streams main,ai`: output the main stream and AI debug stream
- `--streams main,sub`: output normal dual streams only, without loading the RKNN model
- If omitted, the default is equivalent to `--streams main,sub,ai`

### Show Help

```bash
./bin/rk3576_yolov8tortsp_demo --help
```

## Interface Description

### Core Command-Line Arguments

- `--source <v4l2|rtsp|localvideo>`: input source type
- `--device <path>`: V4L2 device node
- `--rtsp-input-url <url>`: RTSP URL
- `--rtsp-username <value>`: RTSP username
- `--rtsp-password <value>`: RTSP password
- `--localvideo <path>`: local video file path
- `--model <path>`: RKNN model path
- `--camera-width <value>` / `--camera-height <value>`: input resolution
- `--output-width <value>` / `--output-height <value>`: main stream output resolution
- `--sub-width <value>` / `--sub-height <value>`: sub stream output resolution
- `--fps <value>`: target encoding frame rate
- `--bitrate <value>`: main stream target bitrate
- `--rtsp-port <value>`: local RTSP server port
- `--rtsp-app <value>`: RTSP app name
- `--rtsp-stream <value>`: main stream name
- `--sub-stream <value>`: sub stream name
- `--ai-stream <value>`: AI debug stream name
- `--streams <items>`: enabled output streams, supports comma-separated combinations of `main`, `sub`, and `ai`
- `--rtsp-timeout-ms <value>`: RTSP read timeout
- `--rtsp-reconnect-ms <value>`: RTSP reconnect interval
- `--perf-interval-ms <value>`: PERF output interval
- `--detail-info`: enable detailed logs
- `--help`: print help

### PERF Metrics

- `process`: image processing time inside a branch
- `encode`: MPP encoding time
- `push`: RTSP streaming push time
- `e2e`: total time inside a branch from processing start to stream push completion
- `fps`: actual branch throughput calculated over a wall-clock time window

## Project Structure

```text
rk3576_yolov8tortsp_demo
├── CMakeLists.txt
├── README.md
├── docs
├── include
│   ├── rk3576_demo
│   └── rk3576_yolo_demo
├── model
├── scripts
├── src
│   ├── ai          # RKNN inference and pre/post-processing
│   ├── app         # Application entry and runtime orchestration
│   ├── branch      # Three output branches, OSD, and PERF
│   ├── camera      # V4L2 capture
│   ├── codec       # MPP encode/decode
│   ├── common      # Common types, logger, thread queue
│   ├── pipeline    # V2 distribution and processing pipeline
│   ├── rga         # Image scaling and conversion
│   ├── rtsp        # Local RTSP publishing
│   ├── source      # Input source abstraction and implementations
│   ├── utils
│   └── watermark   # Timestamp / FPS / label rendering
└── third_party
```

## Secondary Development Notes

### 1. Input Source Extension

- Unified interface: `include/rk3576_yolo_demo/source/i_source.hpp`
- Factory entry: `src/source/source_factory.cpp`
- A new input source should implement:
  - `Open()`
  - `Close()`
  - `ReadPacket()`
  - `Describe()`

### 2. Branch Output Extension

- Branch implementation is located in `src/branch/branch_output.cpp`
- Three geometry strategies are currently supported:
  - `stretch`
  - `letterbox`
  - `center_crop`
- Geometric relationships are unified in `ComputeFrameTransform()` for reuse across output scaling and AI box mapping

### 3. AI Inference Extension

- RKNN entry is in `src/ai/yolov8_engine.cpp`
- Preprocessing prefers `RGA`
- Detection results are stored in the `DetectionFrame` cache, then matched and overlaid by the AI branch using `PTS`

### 4. Logging and Debugging

- By default, only the startup summary is printed
- Detailed runtime logs can be enabled with `--detail-info`
- Logger is located at `include/rk3576_yolo_demo/common/logger.hpp`

## Deployment Recommendations

- In production, make sure `/dev/rga`, video devices, and runtime libraries have proper permissions
- It is recommended to package models, runtime libraries, and config files together under `install/`
- RTSP input prefers `libavformat`, and automatically falls back to `ffmpeg pipe` on older environments
- For long-term stable operation, use `systemd` or `supervisor` for process start and restart management

## Known Limitations

- Bounding box drawing and watermark overlay on `camera_ai` still consume some CPU
- `e2e` is not the full end-to-end pipeline latency; it only represents branch-internal latency
- `localvideo` currently uses a compressed-stream read path, mainly for offline validation and loop inference

## Future Development Suggestions

1. Further reduce OSD CPU overhead on `camera_ai`, for example by introducing tile cache or a more efficient hybrid rendering path
2. Add a true full-pipeline `pipeline_e2e` metric to distinguish it from the current branch-level `e2e`
3. Improve playback pacing control in the `localvideo` scenario so it behaves more like a real-time source
4. Add more independent encoding parameter configuration options for the main and sub streams
5. If more complex input types are needed, continue extending the `source` abstraction layer
