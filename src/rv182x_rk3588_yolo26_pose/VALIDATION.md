# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| yolo26n-pose | 33.46 | 14.70 | 34.07 | [JSON](validation/benchmark-yolo26n-pose.json) |
| yolo26s-pose | 21.23 | 11.05 | 51.30 | [JSON](validation/benchmark-yolo26s-pose.json) |
| yolo26m-pose | 11.95 | 7.85 | 88.25 | [JSON](validation/benchmark-yolo26m-pose.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-yolo26-pose:local`

Image ID: `sha256:7ee39f0aab44af51dd448fc169049a5a14281b7fd906d91ed62ef61e2ac7c899`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
