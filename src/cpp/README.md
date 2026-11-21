# Rockchip CV API for RK3576 and RK3588

**Run computer vision on RK3576 and RK3588. Build your app with HTTP.**

Start a Docker container on your Rockchip board and send an image to
`POST /api/predict`. Each image includes the model, RKNN userspace runtime,
and a C++ engine that uses the NPU for inference and RGA for preprocessing.
Your app can call it from any machine with an HTTP client.

[Quick start](#quick-start) · [Models](#models) · [API](#api) · [Video](#video) · [Build and publish](#build-and-publish)

## Quick start

You need an **RK3576 or RK3588 board** running ARM64 Linux with Docker and
working NPU, RGA, and MPP drivers supplied by the board OS.

**1. Start YOLO26 object detection on the board**

```sh
CHIP=rk3588  # Use rk3576 on an RK3576 board.
IMAGE="ghcr.io/seeed-projects/recomputer-rk-cv/yolo26n:${CHIP}-int8"

docker run -d --name cv-api --restart unless-stopped \
  --privileged -p 8000:8000 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  "$IMAGE"
```

Pick a model and tag from the [model catalog](#models),
or [build the same image locally](#build-locally). Use a different container
name and host port, such as `-p 8001:8000`, to serve a second model.

**2. Check the model and send an image**

Run this on the board or replace `localhost` with the board's IP address.

```sh
API=http://localhost:8000

curl -fsS "$API/health"
curl -fsS "$API/api/metadata"
curl -fsS "$API/api/predict" -F "file=@image.jpg"
```

Call `/api/metadata` when your client connects to discover the model's task,
labels, and result format. See [Read model metadata](#read-model-metadata) for
the response fields and raster decoding instructions.

Use a local JPEG or PNG for `image.jpg`. Open `http://<board-ip>:8000/docs`
for interactive requests.

<details>
<summary>Example detection response</summary>

```json
{
  "success": true,
  "source": "image",
  "predictions": [
    {
      "class": "person",
      "class_id": 0,
      "confidence": 0.92,
      "box": {"x1": 100, "y1": 200, "x2": 300, "y2": 500}
    }
  ],
  "image": {"width": 1280, "height": 720}
}
```

This excerpt shows bounding boxes in original image pixels. A frame with
no detections returns an empty `predictions` array. Responses also include
the loaded model and timing fields.

</details>

### Python client

Install `requests` in your application environment with
`python -m pip install requests`. This example selects the result field from
the model metadata and prints boxes for the quick-start detection model.

```python
import requests

api = "http://localhost:8000"  # Use the board's IP when calling remotely.

metadata_response = requests.get(f"{api}/api/metadata", timeout=10)
metadata_response.raise_for_status()
metadata = metadata_response.json()

with open("image.jpg", "rb") as image:
    response = requests.post(
        f"{api}/api/predict",
        files={"file": image},
        data={"conf": 0.3},
        timeout=30,
    )

response.raise_for_status()
result = response.json()[metadata["output"]["field"]]
if metadata["model"]["task"] == "detection":
    for item in result:
        print(item["class"], item["confidence"], item["box"])
```

## Models

YOLO26 ships in sizes `n`, `s`, and `m` for RK3576 and RK3588, with
INT8 and FP16 builds for all seven tasks. Each image serves one model.
The catalog contains 84 variants across 21 image names.

All images built from the same code revision share the same ARM64 operating
system, dependencies, C++ engine, HTTP API, and browser code. Only the final
filesystem layer contains model-specific files. The common entrypoint loads
the model manifest bundled at `/opt/model/package.json`.

Image names below use the prefix `ghcr.io/seeed-projects/recomputer-rk-cv/`.

| Task | Example RK3588 image | Prediction result |
| --- | --- | --- |
| Object detection | `yolo26n:rk3588-int8` | `predictions` with boxes and scores |
| Instance segmentation | `yolo26n-seg:rk3588-int8` | `predictions[].mask` |
| Pose estimation | `yolo26n-pose:rk3588-int8` | `predictions[].keypoints` as `[x, y, confidence]` |
| Oriented boxes | `yolo26n-obb:rk3588-int8` | `predictions[].polygon` with four corners |
| Classification | `yolo26n-cls:rk3588-int8` | `classification` with ranked classes |
| Semantic segmentation | `yolo26n-sem:rk3588-int8` | `semantic` raster and class labels |
| Depth estimation | `yolo26n-depth:rk3588-int8` | `depth` raster |

Replace `rk3588` with `rk3576` for the other board, or `int8` with `fp16`
to change precision. Run `python3 tools/models.py list` for all image tags
and build IDs in the [model catalog](models/).

<details>
<summary>Raster format</summary>

Masks, semantic maps, and depth maps have `width`, `height`, `box`, and
`runs`. The box locates the raster in the source image. Runs alternate value
and count in row-major order. Semantic values are class IDs. Recover depth
samples with `offset + value * scale`. The value `65536` marks invalid depth.

</details>

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check model and runtime status |
| `GET` | `/api/models` | Identify the model loaded in this container |
| `GET` | `/api/metadata` | Discover the model's labels, input size, and prediction format |
| `POST` | `/api/predict` | Predict an image, a video frame, or the latest live frame |
| `GET`, `POST` | `/api/config` | Read or update default confidence and NMS thresholds |
| `GET` | `/api/video_feed` | Stream the configured source as annotated MJPEG |
| `GET` | `/` | Watch live object detection and adjust thresholds |
| `GET` | `/docs`, `/openapi.json` | Browse or export the API schema |

Send an image as multipart `file` or a clip as `video`. For a clip, set
`timestamp` in seconds to select one frame. It defaults to `0`.
A POST with no upload predicts the latest frame from `RK_SOURCE`.

```sh
curl -fsS "$API/api/predict" -F "video=@clip.mp4" -F "timestamp=1.5"
```

Optional multipart fields `conf` and `iou` override thresholds for one
request. Their initial defaults are `0.25` and `0.45`. To update service
defaults, POST JSON with `obj_thresh` and `nms_thresh` to `/api/config`.
NMS runs in C++ when enabled for the export. The compatibility route
`/api/models/{model_id}/predict` accepts the loaded ID, family, or a
configured compatibility alias.

### Read model metadata

```sh
curl -fsS "$API/api/metadata"
```

The JSON response includes `schema_version` (currently `1`), `model` (including
its task), `model_input` (the network width and height), `class_count`, `labels`,
and `output`. The index into `labels` is the zero-based `class_id`. The labels
reflect `RK_CLASSES` when set. If the label list is empty, use the class ID
directly. Depth has zero classes.

Read `prediction[metadata["output"]["field"]]` from the prediction response to
get the task's main result. When `output.item_field` is present, read that key
from each item in the result array. `output.point_fields` gives the order of
values in each keypoint or polygon vertex. Coordinates use source-image pixels.
Use the prediction's `image.width` and `image.height` for the frame size.

For raster results, expand each `[value, count]` pair in the flat `runs` array,
then reshape the values into `height` rows of `width` samples. The raster's
`box` locates it in the source image. Mask values are `0` or `1`, semantic
values are class IDs, and depth samples use `offset + value * scale`. Skip the
sentinel given by `output.raster.invalid_value` in the metadata before applying
that formula.

<details>
<summary>Latency and configuration</summary>

The response body reports stage timings in milliseconds, including upload
storage, decoding, context wait, DMA upload, RGA preprocessing,
`rknn_run_ms`, `output_get_ms`, and postprocessing. `nms_ms`, when present,
is part of `postprocess_ms`. The `Server-Timing` header reports request
parsing, receive wait, response encoding, and total server time before
response headers. Several timings overlap. Measure the full request on
the client to include transport and response delivery.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `RK_NMS` | Model setting | Enable or disable server NMS with `on` or `off` |
| `RK_SOURCE` | Unset | RTSP URL, mounted video file, or H.264 V4L2 device |
| `RK_LOOP_VIDEO` | `0` | Set to `1` for the browser-rendered local MP4 detection demo |
| `RK_WORKERS` | `0` | Automatic live worker count, or `1` for a single context |
| `RK_MAX_UPLOAD_MB` | `128` | Maximum upload size in MB |
| `RK_CLASSES` | Packaged labels | Override labels with one class per line |
| `RK_PACKAGE` | `/opt/model/package.json` | Model package manifest path |

Use `docker logs cv-api` to investigate startup failures. Check that the
image tag matches the board and its kernel drivers are available.
HTTP 503 on a live request means the source is unavailable or has yet to
provide a usable frame.

</details>

## Video

For continuous prediction, add `-e RK_SOURCE=rtsp://192.168.1.10/live` to
the Docker run command. For a local video, pass
`-v "$PWD/media:/media:ro" -e RK_SOURCE=/media/input.mp4`.
An H.264 V4L2 camera can use `-e RK_SOURCE=/dev/video0`.

Continuous H.264/H.265 sources use MPP hardware decoding. Once a frame is
ready, open `http://<board-ip>:8000` to watch the annotated feed and
adjust confidence and NMS IoU. Threshold changes also update the container's
API defaults. Use a detection image such as `yolo26n:rk3576-int8` for the live preview.

For smooth playback of a mounted H.264 MP4 with an object detection model,
also pass `-e RK_LOOP_VIDEO=1`. The board loops the file and publishes
timestamped detection results. The browser plays the original MP4 and draws
boxes on a canvas, so playback can run at the video's frame rate even when
inference produces fewer results. The MP4 should use landscape or portrait
dimensions encoded directly, without a rotation tag.

With a YOLO26 split scores-and-boxes model, the browser applies confidence
filtering and per-class NMS locally. The board sends every valid decoded
proposal with its highest-scoring class. It applies no confidence cutoff,
top-k limit, or NMS to this stream. The preview controls affect the current
tab. The prediction API and legacy MJPEG feed use the container's API settings.

The browser reads `GET /api/demo/candidates` as a binary stream. Each
little-endian frame contains a `uint32` payload size in bytes, `uint32` source
width, `uint32` source height, `uint32` video loop index, `float64` media
timestamp in seconds, and `uint64` frame sequence. The payload contains
six `float32` values per proposal in the order `x1, y1, x2, y2, confidence,
class_id`. Box coordinates are in source-image pixels. Use the label list
from `GET /api/metadata`.

Call `curl -fsS -X POST "$API/api/predict"` for the latest result or use
`/api/video_feed` directly for the 10 FPS MJPEG preview. Local files follow
media timestamps from container startup and retain the last result at EOF
when looping is disabled.

## Build and publish

### Build locally

Run these commands in `src/cpp` on an ARM64 Docker host. Inference requires
the matching Rockchip board.

```sh
python3 tools/models.py list

docker build --platform linux/arm64 --target model \
  --build-arg MODEL_ID=yolo26n-rk3588-int8 \
  -t yolo26n:rk3588-int8 .
```

Set `IMAGE=yolo26n:rk3588-int8` in the quick-start command to run the local
image. To build several model images from exactly the same local runtime, use
the same packaging script as CI. Leave out `--model` to build the full catalog.

```sh
docker build --platform linux/arm64 --target runtime -t rockchip-cv-runtime:local .
python3 -m pip install PyYAML
python3 tools/build_images.py --runtime rockchip-cv-runtime:local \
  --repository rockchip-cv-local --model yolo26n-rk3588-int8
```

### GitHub Actions

The [C++ Rockchip model images workflow](../../.github/workflows/cpp-model-images.yml)
uses a native ARM64 runner with a sparse checkout of `src/cpp`. Pull requests
build and check the full catalog. Changes to this directory or workflow on
`main` publish all 84 variants to GHCR. A manual run accepts one catalog ID
or the full catalog when left empty. Manual runs on `main` publish images.

The workflow saves one shared runtime per source commit under
`runtime:<commit-sha>`. Each model uses it as the base, verifies the model
package and native module imports, and checks that its layer list matches the
runtime plus exactly one model layer. Each model is published with a
`<chip>-<precision>` tag and a `<chip>-<precision>-<commit-sha>` tag.

Each manifest points to an immutable Hugging Face revision. The build
downloads the RKNN model and its metadata, verifies their SHA-256 checksums,
and checks the metadata against the task, chip, class list, and tensor
contract. The repository stores the manifests without the weights.

## Credits

This implementation builds on [rockchip-cv-api](https://github.com/Hanzo-Huang/rockchip-cv-api).
The C++ inference engine draws from [rockchip-yolo-cpp](https://github.com/Hanzo-Huang/rockchip-yolo-cpp).
The HTTP interface follows [Seeed's Rockchip CV API](https://github.com/Seeed-Projects/reComputer-RK-CV/tree/main/src/rk3588_yolo11).
The model packages use weights and conversion metadata from
[HanzoHuang/YOLO26-RKNN2](https://huggingface.co/HanzoHuang/YOLO26-RKNN2).
Model manifests and metadata record provenance, license information, and the
SHA-256 checksums used during packaging.
