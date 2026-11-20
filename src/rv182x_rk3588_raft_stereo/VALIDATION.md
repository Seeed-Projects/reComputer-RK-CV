# Native RK3588 + RK1820 validation · 2026-11-20

All listed variants passed native Docker benchmarks. RKNN3 Toolkit/Lite 1.1.0, rk1820, W8A8, core mask 0x1; five warmup calls and 30 measured frames/pairs.

| Variant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Evidence |
|---|---:|---:|---:|---|
| raftstereo-eth3d-160x240 | 4.75 | 1.84 | 211.66 | [JSON](validation/benchmark-raftstereo-eth3d-160x240.json) |
| raftstereo-middlebury-160x240 | 4.76 | 1.87 | 210.81 | [JSON](validation/benchmark-raftstereo-middlebury-160x240.json) |
| raftstereo-sceneflow-160x240 | 4.77 | 1.86 | 210.28 | [JSON](validation/benchmark-raftstereo-sceneflow-160x240.json) |
| raftstereo-realtime-160x240 | 16.48 | 2.56 | 61.58 | [JSON](validation/benchmark-raftstereo-realtime-160x240.json) |

API regression: [JSON](validation/api.json), [log](validation/api.log).
Paired official images, one exported iteration. API: health, paired upload, sample JPEG and missing-input rejection. No monocular video is treated as stereo.

Tested image: `rv182x-rk3588-raft-stereo:local`

Image ID: `sha256:f3a5ed8e5b249dd5abcf0e48752efa431632c40917b479f96d45668527c0c18b`

Conversion records are in conversion/*.json; runtime artifact checksums are in SHA256SUMS. This is functional/performance validation, not a COCO/ImageNet/stereo/depth accuracy evaluation.
