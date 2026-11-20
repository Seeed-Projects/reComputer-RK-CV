# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| yolov8n | 28.98 | 12.57 | 40.53 | [JSON](validation/benchmark-yolov8n.json) |
| yolov8s | 21.46 | 10.64 | 53.15 | [JSON](validation/benchmark-yolov8s.json) |
| yolov8m | 11.27 | 7.05 | 98.93 | [JSON](validation/benchmark-yolov8m.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-yolov8:local`

Image ID: `sha256:8157025bae6c0607ffeee95c5e91914aa489481a8d68b2e3cf6f22aac588ebff`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
