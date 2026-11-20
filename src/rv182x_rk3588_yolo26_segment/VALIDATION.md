# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| yolo26n-seg | 17.23 | 3.93 | 67.44 | [JSON](validation/benchmark-yolo26n-seg.json) |
| yolo26s-seg | 14.29 | 2.58 | 82.22 | [JSON](validation/benchmark-yolo26s-seg.json) |
| yolo26m-seg | 6.94 | 2.19 | 164.00 | [JSON](validation/benchmark-yolo26m-seg.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-yolo26-segment:local`

Image ID: `sha256:8a33d8e78718eaad651c4d7fa935573f35407f5c67469882db46309be748c8aa`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
