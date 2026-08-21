# RK3588 PPOCR End-to-End OCR Deployment

[English] | [中文](./README_zh.md)

This project combines PPOCR-Det and PPOCR-Rec into one RK3588 service. A complete still image is accepted from the Web page, REST API, or `--image` command-line option. The service detects text polygons, perspective-rectifies each region, recognizes every crop, and returns the text together with coordinates and confidence.

## Features

- Runs `ppocr_det.rknn` and `ppocr_rec.rknn` with RKNNLite on the RK3588 NPU.
- Detects multiple text regions in documents and natural-scene images.
- Sorts regions into reading order and performs four-point perspective correction.
- Displays the annotated image, recognized text, text crops, confidence, and latency in the Web UI.
- Provides a standard image prediction API and OpenAPI documentation.
- Supports local-image startup and one-shot command-line inference.
- Does not include video upload, video analysis, or local-camera code.

## Processing pipeline

```text
still image -> PPOCR-Det -> sorted quadrilaterals -> perspective crops
            -> PPOCR-Rec -> text, confidence, coordinates -> annotated image
```

## Directory structure

- `model/ppocr_det.rknn`: RK3588 text detection model.
- `model/ppocr_rec.rknn`: RK3588 text recognition model.
- `model/ppocr_keys_v1.txt`: CTC character dictionary.
- `model/simfang.ttf`: Unicode annotation font.
- `model/test.jpg`: startup warm-up image.
- `task_runtime.py`: detection, sorting, perspective correction, recognition, and annotation.
- `web_service.py`: Web page, REST API, and command-line image entry point.
- `rknn_runtime.py`: thread-safe RKNNLite wrapper.
- `utils/`: PPOCR preprocessing and postprocessing utilities.

## Docker quick start

```bash
sudo docker run --rm --name rk3588-ppocr \
  --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-ppocr:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --host 0.0.0.0 --port 8000
```

Open `http://<board-ip>:8000`. OpenAPI is available at `http://<board-ip>:8000/docs`.

Build from the repository root:

```bash
docker build -f docker/rk3588/ppocr.dockerfile \
  -t rk3588-ppocr:local src/rk3588_ppocr
```

## Web interaction

1. Upload a JPG, PNG, BMP, or WEBP image containing one or more text regions.
2. Adjust detection, box, expansion, and recognition thresholds if needed.
3. Click **Analyze image**.
4. Inspect the polygons and labels on the annotated image.
5. Review the ordered crop list, text, confidence, coordinates, and stage latency.

The page only accepts still images. Video and camera controls have intentionally been removed.

## Analyze a local image from the command line

Preload an image and keep the Web server running:

```bash
python web_service.py --platform rk3588 --model_dir model \
  --image /data/document.jpg --host 0.0.0.0 --port 8000
```

Run once, print JSON, save the annotated result, and exit:

```bash
python web_service.py --platform rk3588 --model_dir model \
  --image /data/document.jpg --output /data/document_result.jpg --no_server
```

Mount the image directory when running this mode in Docker:

```bash
sudo docker run --rm --privileged \
  -e RKNN_LOG_LEVEL=0 \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  -v "$PWD/data:/data" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-ppocr:latest \
  python web_service.py --platform rk3588 --model_dir /app/model \
    --image /data/document.jpg --output /data/document_result.jpg --no_server
```

## Command-line parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--platform` | Yes | — | RKNN target; use `rk3588`. |
| `--model_dir` | No | `model` | Directory containing both models, dictionary, font, and sample. |
| `--image` | No | — | Local still image analyzed before the server starts. |
| `--output` | No | — | JPG/PNG path for the annotated `--image` result. |
| `--no_server` | No | Off | Exit after processing `--image`. |
| `--host` | No | `0.0.0.0` | FastAPI listen address. |
| `--port` | No | `8000` | FastAPI listen port. |

## REST API

**Endpoint:** `POST /api/models/ppocr/predict`
**Content type:** `multipart/form-data`

| Field | Required | Default | Description |
| --- | --- | --- | --- |
| `file` | Yes | — | JPG, PNG, BMP, or WEBP image. Use `@` before a curl file path. |
| `det_threshold` | No | `0.3` | Pixel-map detection threshold, from 0 to 1. |
| `box_threshold` | No | `0.6` | Minimum DB text-box score, from 0 to 1. |
| `unclip_ratio` | No | `1.5` | Polygon expansion ratio, from 0.1 to 5. |
| `drop_score` | No | `0.5` | Minimum recognition confidence included in combined text. |
| `max_results` | No | `100` | Maximum detected regions sent to Rec, from 1 to 1000. |
| `include_crops` | No | `false` | Include base64 JPEG crop thumbnails in each result line. |

```bash
curl -X POST http://127.0.0.1:8000/api/models/ppocr/predict \
  -F "file=@/data/document.jpg" \
  -F "drop_score=0.5" \
  -F "include_crops=false"
```

Important response fields:

```json
{
  "success": true,
  "model": "ppocr",
  "inference_time": 0.1532,
  "result": {
    "text": "Recognized first line\nRecognized second line",
    "lines": [
      {
        "index": 1,
        "text": "Recognized first line",
        "confidence": 0.9621,
        "accepted": true,
        "box": [[42, 31], [286, 29], [287, 72], [43, 74]],
        "crop_size": {"width": 245, "height": 43}
      }
    ],
    "count": 2,
    "detected_count": 2,
    "processed_count": 2,
    "timing_ms": {"detection": 54.2, "recognition": 71.4, "total": 129.8}
  }
}
```

Other endpoints:

- `GET /api/health`: model and platform status.
- `GET /api/config`, `POST /api/config`: read or update default thresholds.
- `GET /api/results/latest`: latest structured OCR result.
- `GET /api/results/latest/image`: latest annotated JPEG result.
- `GET /docs`: interactive OpenAPI documentation.

## Scope and limitations

- The recognizer is optimized for perspective-corrected single-line crops; detection quality directly affects recognition.
- Very small, blurred, curved, vertical, low-contrast, or heavily rotated text can be missed or decoded incorrectly.
- Regions are sorted top-to-bottom and left-to-right with a simple line heuristic; complex multi-column documents may need a layout model.
- `include_crops=true` increases API response size and is mainly intended for the Web result panel.

Model conversion sources are under `rknn_model_zoo/examples/PPOCR/PPOCR-Det` and `PPOCR/PPOCR-Rec`.
