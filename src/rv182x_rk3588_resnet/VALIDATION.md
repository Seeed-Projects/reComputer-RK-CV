# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| resnet50-v2-7 | 89.10 | 26.31 | 11.51 | [JSON](validation/benchmark-resnet50-v2-7.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-resnet:local`

Image ID: `sha256:f36b388d18a271c78414911f1aa5222b7fe31b0d4cbc063f98909a9ac50b4ef7`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
