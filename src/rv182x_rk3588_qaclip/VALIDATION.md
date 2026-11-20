# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| qaclip-vit-l14 | 6.45 | 5.26 | 155.36 | [JSON](validation/benchmark-qaclip-vit-l14.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.
Text vectors are cached on the host; the text context is released before loading vision. Prompt changes are tested. FPS measures cached-text frame processing; text setup time is recorded separately.

Tested image: `rv182x-rk3588-qaclip:local`

Image ID: `sha256:e90cd0464bdeb48e1b33043c0029e19ff85ba8cc7156f196e7d3ba4c9ec24ced`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
