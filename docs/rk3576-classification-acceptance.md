# RK3576 classification acceptance report

Date: 2026-07-31

## Device

- Board compatibility: `seeed,recomputer rk3576 devkit`, `rockchip,rk3576`
- Architecture: AArch64
- Kernel: `6.1.115-vendor-seeed-rk3576`
- Docker: `20.10.24+dfsg1`
- RKNN Toolkit Lite2: `2.3.2`
- Python image: `python:3.11-slim`

The host had Docker AppArmor support enabled but did not have
`apparmor_parser`. The Debian `apparmor` package was installed before building;
no other host packages were upgraded.

## Results

| Check | RK3576 MobileNet | RK3576 ResNet50V2 |
| :--- | :--- | :--- |
| Static project validator | PASS | PASS |
| Native arm64 Docker build | PASS | PASS |
| RKNN model load | PASS | PASS |
| RK3576 two-core initialization | PASS | PASS |
| Real NPU image inference | PASS | PASS |
| `GET /api/health` | PASS | PASS |
| Browser page | HTTP 200 | HTTP 200 |
| OpenAPI | 10 paths | 10 paths |
| Async image analysis | PASS | PASS |
| Result download | HTTP 200 | HTTP 200 |
| Docker health check | healthy | healthy |
| LAN access from development host | PASS | Not separately repeated |

MobileNet test image result:

```text
Top-1: n03017168 chime, bell, gong
Confidence: 0.9873423576
```

ResNet50V2 test image result:

```text
Top-1: n02086240 Shih-Tzu
Confidence: 0.7921831608
```

Built image sizes on the acceptance device:

```text
rkcv-acceptance-mobilenet:rk3576    590 MB
rkcv-acceptance-resnet50v2:rk3576  613 MB
```

The RKNN runtime prints `Query dynamic range failed` for these static-shape
models and explicitly states that the warning can be ignored. It did not affect
model load or inference.

## Device state after acceptance

- Temporary running containers were stopped and removed.
- The two acceptance images remain available for retesting.
- Acceptance files remain under the dedicated `rkcv-acceptance-20260731`
  directory.
- No service is left listening on port 8000.
