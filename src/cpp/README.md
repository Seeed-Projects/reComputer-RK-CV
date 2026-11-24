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

```bash
docker run -d --name cv-api --restart unless-stopped \
  --privileged -p 8000:8000 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/yolo26n:rk3588-int8
```

On an RK3576 board, use the `rk3576-int8` tag.
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

The catalog follows the model exports in [Rockchip CV API](https://github.com/Hanzo-Huang/rockchip-cv-api).
It contains 316 INT8 and FP16 variants across 10 families for RK3576 and RK3588.
Each image serves one model and includes demo media for its task.

| Family | Sizes | Tasks |
| --- | --- | --- |
| YOLO26 | `n`, `s`, `m` | Detection, segmentation, pose, oriented boxes, classification, semantic segmentation, depth |
| YOLO11, YOLOv8 | `n`, `s`, `m` | Detection, segmentation, pose, oriented boxes, classification |
| YOLOv5 | `n`, `s`, `m`, plus `s-relu` for detection | Detection, segmentation, classification |
| YOLOv6 | `n`, `s`, `m` | Detection |
| YOLOv7 | `base`, `tiny` | Detection |
| YOLOv10 | `n`, `s`, `m`, `b`, `l`, `x` | Detection |
| YOLOX | `nano`, `s`, `m` | Detection |
| PP-YOLOE | `s`, `m` | Detection |
| YOLO-World | `v2s`, `v2m` | Detection with the model zoo's fixed COCO vocabulary |

All images built from the same code revision share the same ARM64 operating
system, dependencies, C++ engine, HTTP API, and browser code. The final filesystem layer contains model-specific files. A separate layer contains
the selected demo media and its attribution. The common entrypoint loads
the model manifest bundled at `/opt/model/package.json`.

Image names below use the prefix `ghcr.io/seeed-projects/recomputer-rk-cv/`.

| Task | Example RK3588 image | Prediction result |
| --- | --- | --- |
| Object detection | `yolo26n:rk3588-int8` | `predictions` with boxes and scores |
| Instance segmentation | `yolo26n-seg:rk3588-int8` | `predictions[].mask` |
| Pose estimation | `yolo26n-pose:rk3588-int8` | `predictions[].keypoints` as `[x, y, confidence]` |
| Oriented boxes | `yolo26n-obb:rk3588-int8` | `predictions[].polygon` with four corners |
| Classification | `yolo26n-cls:rk3588-int8` | `classification` with ranked class IDs, names, and scores |
| Semantic segmentation | `yolo26n-sem:rk3588-int8` | `semantic` raster and class labels |
| Depth estimation | `yolo26n-depth:rk3588-int8` | `depth` raster |

Replace `rk3588` with `rk3576` for the other board, or `int8` with `fp16`
to change precision. For example, use `yolo11n-seg:rk3588-int8` for YOLO11
segmentation. Run `python3 tools/models.py list` for all image tags
and build IDs in the [model catalog](models/).

<details>
<summary>Raster format</summary>

Masks, semantic maps, and depth maps have `width`, `height`, `box`, and
`runs`. The box locates the raster in the source image. Runs alternate value
and count in row-major order. Semantic values are class IDs. Recover depth
samples with `offset + value * scale`. The value `65536` marks invalid depth.
The YOLO26 depth packages use RGA to resize each frame to the model's full
input size. The live preview colorizes inverse depth, with warmer colors for
nearer regions and a relative scale within each frame.

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
| `RK_SOURCE` | Bundled demo video | Override with an RTSP URL, mounted video file, or H.264 V4L2 device. Set to an empty string to disable live preview. |
| `RK_LOOP_VIDEO` | `1` for the bundled demo, otherwise `0` | Set to `1` for looping, browser-rendered playback of a local MP4 |
| `RK_WORKERS` | `0` | Use all NPU cores with dedicated workers, two on RK3576 and three on RK3588. Set to `1` for one context. |
| `RK_MAX_UPLOAD_MB` | `128` | Maximum upload size in MB |
| `RK_CLASSES` | Packaged labels | Override labels with one class per line |
| `RK_PACKAGE` | `/opt/model/package.json` | Model package manifest path |

Use `docker logs cv-api` to investigate startup failures. Check that the
image tag matches the board and its kernel drivers are available.
HTTP 503 on a live request means the source is unavailable or has yet to
provide a usable frame.

</details>

## Video

Open `http://<board-ip>:8000` to play the bundled clip and view the model's
output. The first live-stream subscriber starts the decoder and NPU workers.
The last disconnect stops them. Reading the page or checking `/health` does
not start inference. Image or video-frame uploads still run on demand.
`/health` reports whether the pipeline is active, the subscriber count, and
the worker count and core masks when running.

| Tasks | Bundled media |
| --- | --- |
| Detection | [Street traffic from the YOLO11 showcase, in the existing 1080p YOLO26 example](../rv182x_rk3588_yolo26/video/test.mp4) |
| Depth | [Provided depth demo clip](demo/depth.txt). Warmer colors show nearer regions. Adjust overlay opacity to compare with the video. |
| Pose | [Single-person squat demonstration](demo/pose.txt). The browser smooths matched joints across consecutive frames. |
| Instance segmentation | [YOLOv8 pose showcase](../rk3576_yolov8_pose/video/test.mp4) |
| Oriented boxes | [Official Ultralytics aerial ship photo](demo/ships.txt). Select **Detect ships** to draw rotated boxes on the image. The prediction API returns all model classes. |
| Semantic segmentation | [Street-level urban footage](demo/semantic.txt). Each color shows which pixels belong to a class like road, sidewalk, or vegetation. |
| Classification | [Example photos](demo/photos.txt) and an upload form. The page runs a prediction for each photo the viewer submits. |

Use **Input source** to select the built-in example, a capture device exposed to
the container, or a local photo or video. A board camera opens only when you
start its preview. The browser decodes uploaded videos locally and posts sampled
JPEG frames to the preview endpoint. Uploads and camera streams work across all
seven tasks.
Camera and upload previews use one inference worker per NPU core by default,
with two on RK3576 and three on RK3588. They keep only one frame in flight per
worker and present results in frame order. Set `RK_WORKERS=1` to use a single
context. The preview displays each sampled upload frame with its own result. Built-in
videos draw results only on matching video timestamps. Closing the last live
preview stops its inference. On RK3588, select `rk_hdmirx` to use HDMI IN.

The browser plays the H.264 MP4 and draws model results on a canvas. For a
custom clip, pass `-v "$PWD/input.mp4:/media/input.mp4:ro"` with
`-e RK_SOURCE=/media/input.mp4 -e RK_LOOP_VIDEO=1`. Use a browser-compatible
MP4 without a rotation tag. RTSP and H.264 V4L2 sources still use the MJPEG
preview with the source selected through `RK_SOURCE`.

With a YOLO26 split scores-and-boxes model, the browser applies confidence
filtering and per-class NMS for the built-in clip, board camera, and uploaded
video. The board sends every valid decoded
proposal with its highest-scoring class. It applies no confidence cutoff,
top-k limit, or NMS to this stream. The preview controls affect the current
tab. The prediction API and legacy MJPEG feed use the container's API settings.

For the built-in clip, the browser reads `GET /api/demo/candidates` as a binary stream. Each
little-endian frame contains a `uint32` payload size in bytes, `uint32` source
width, `uint32` source height, `uint32` video loop index, `float64` media
timestamp in seconds, and `uint64` frame sequence. The payload contains
six `float32` values per proposal in the order `x1, y1, x2, y2, confidence,
class_id`. Box coordinates are in source-image pixels. Use the label list
from `GET /api/metadata`.

Call `curl -fsS -X POST "$API/api/predict"` for the latest result while a
preview stream is subscribed. `/api/video_feed` serves the 10 FPS MJPEG
preview and holds a live-source subscription.

## Build and publish

### Build locally

Run these commands in `src/cpp` on an ARM64 Docker host. Inference requires
the matching Rockchip board. Use `--family yolo11` in place of `--model` to
build a family, or omit both filters to build the full catalog.

```sh
python3 tools/models.py list
docker build --platform linux/arm64 --target runtime -t rockchip-cv-runtime:local .
python3 -m pip install PyYAML
python3 tools/build_images.py --runtime rockchip-cv-runtime:local \
  --repository rockchip-cv-local --model yolo26n-rk3588-int8
```

Replace the quick-start image reference with `rockchip-cv-local/yolo26n:rk3588-int8`.
In a sparse checkout, include the two existing project clip paths in the table
above. Use `--size n --precision int8` in place of `--model` to check the
n-size INT8 variants on both chips.

### GitHub Actions

The [C++ Rockchip model images workflow](../../.github/workflows/cpp-model-images.yml)
uses native ARM64 runners with sparse checkouts. A shared runtime job builds
the base image, and per-family jobs add the demo media and model packages.
Pull requests check n-size INT8 variants on both chips. Changes to the code,
packaged videos, or workflow on `main` publish all 316 variants to GHCR.
A manual run accepts one catalog ID
or the full catalog when left empty. Manual runs on `main` publish images.

The workflow saves one shared runtime per source commit under
`runtime:<commit-sha>`. Each model uses it as the base, verifies the model
package and native module imports, and checks that its layer list matches the
runtime plus one independent demo media layer and one model layer. The workflow
builds each media layer once from a deterministic archive and reuses it for all
images with the same demo assets.
BuildKit's `ADD --link` keeps the media layer independent so code-only
rebuilds can reuse it. Each model is published with a
`<chip>-<precision>` tag and a `<chip>-<precision>-<commit-sha>` tag.

Each manifest points to an immutable Hugging Face revision. The build
downloads the RKNN model and its metadata, verifies their SHA-256 checksums,
and checks the metadata against the task, chip, class list, and tensor
contract. YOLO-World also downloads and verifies the fixed text embedding
tensor. The repository stores the manifests without the weights.

## Credits

This implementation builds on [rockchip-cv-api](https://github.com/Hanzo-Huang/rockchip-cv-api).
The C++ inference engine draws from [rockchip-yolo-cpp](https://github.com/Hanzo-Huang/rockchip-yolo-cpp).
The HTTP interface follows [Seeed's Rockchip CV API](https://github.com/Seeed-Projects/reComputer-RK-CV/tree/main/src/rk3588_yolo11).
The model packages use weights and conversion metadata from
[HanzoHuang/YOLO26-RKNN2](https://huggingface.co/HanzoHuang/YOLO26-RKNN2).
Model manifests and metadata record provenance, license information, and the
SHA-256 checksums used during packaging.

Demo media sources, attribution, and applicable licenses are recorded in the
image at `/opt/demo/ATTRIBUTION.txt`.
