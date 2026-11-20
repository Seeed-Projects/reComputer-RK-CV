# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| da3-base | 2.92 | 2.56 | 369.29 | [JSON](validation/benchmark-da3-base.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
API: health, MJPEG, JPEG snapshot, identical results for eight concurrent uploaded-image requests, invalid-input rejection, three-frame video upload/analyze/download and decoding of all three output frames.
Input preprocessing matches the official C++ direct resize to 280x280. Global/Head calibration was regenerated for this preprocessing. These results replace the earlier letterbox prototype. Single-view relative depth has limited detail; visible INT8 banding remains. No dataset accuracy certification or metric-distance claim is made.

[FP32 ONNX reference for frame 34](validation/fp32-reference-frame34.jpg) · [INT8 native preview](validation/benchmark-da3-base.jpg).

Tested image: `rv182x-rk3588-depth-anything-v3:local`

Image ID: `sha256:14d215c6d205dd7d1e9531a97b77c4724c1bca1441021b1f8692ffcb2d13d037`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
