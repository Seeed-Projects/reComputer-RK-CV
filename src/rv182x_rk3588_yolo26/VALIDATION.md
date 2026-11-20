# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| yolo26n | 33.59 | 12.87 | 36.80 | [JSON](validation/benchmark-yolo26n.json) |
| yolo26s | 23.02 | 10.50 | 50.16 | [JSON](validation/benchmark-yolo26s.json) |
| yolo26m | 12.50 | 7.41 | 86.17 | [JSON](validation/benchmark-yolo26m.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-yolo26:local`

Image ID: `sha256:8eba91e593cd3e66ffc9fc90211745d6eddfe41b83441fa0b8bf666db2b61e58`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
