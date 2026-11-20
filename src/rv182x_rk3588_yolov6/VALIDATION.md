# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| yolov6n | 35.54 | 14.52 | 31.80 | [JSON](validation/benchmark-yolov6n.json) |
| yolov6s | 18.99 | 10.43 | 60.22 | [JSON](validation/benchmark-yolov6s.json) |
| yolov6m | 10.64 | 6.37 | 103.70 | [JSON](validation/benchmark-yolov6m.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-yolov6:local`

Image ID: `sha256:4b05a75fa37424517e4aca63ca49338c285eb61124256c47c621f41d30f5ffe5`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
