# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| mobilenetv2-12 | 196.95 | 37.42 | 5.63 | [JSON](validation/benchmark-mobilenetv2-12.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.

Tested image: `rv182x-rk3588-mobilenetv2:local`

Image ID: `sha256:3b265018723d567e7087eb11f58f688ddc0d39215ae79f0091aba552a33027ec`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
