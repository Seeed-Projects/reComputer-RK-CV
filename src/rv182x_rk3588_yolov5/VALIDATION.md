# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| yolov5n | 20.93 | 10.40 | 59.38 | [JSON](validation/benchmark-yolov5n.json) |
| yolov5s | 17.89 | 9.67 | 66.11 | [JSON](validation/benchmark-yolov5s.json) |
| yolov5m | 12.21 | 7.55 | 92.51 | [JSON](validation/benchmark-yolov5m.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-yolov5:local`

Image ID: `sha256:089672e314afe0131805230c3e4233a95aa2a9bfd856c4f10c1e94392779c543`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
