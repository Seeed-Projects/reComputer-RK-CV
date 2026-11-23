# Shared-runtime board acceptance — 2026-11-23

Accepted model image: **YOLOv5n only**, built natively from this change on RK3588 + RK1820 M.2, kernel `6.1.172-vendor-seeed-rk3588`, SDK / Lite 1.1.0. Candidate image IDs and original API responses are retained below. The GHCR workflow builds the same runtime/application source, with normal CI dependency installation and OCI labels; native candidate image IDs are not claimed as GHCR digests.

## Verified

- Before startup: PCI endpoint present, driver not loaded, proxy port closed, host service stopped.
- Container loaded the verified kernel module and created its device node without a full host `/dev` mount.
- Container started its own proxy, obtained Online / OK, passed a real MobileNetV2 W8A8 bundled-sample inference, then started YOLOv5n.
- Health endpoint and Docker health check passed.
- Browser upload mode `--camera_id -1`, with **no workspace mount**: traffic-frame prediction returned detections; upload, offline analysis (100%) and download succeeded.
- Downloaded output was actually decoded: **3 frames, 1920×1080**. This is a functional test, not a new FPS benchmark.
- A second managed container was rejected with `DEVICE_BUSY` before inference.
- Stopping the test container stopped its owned proxy. The previously enabled host service was restored; restarting the same image reused that external proxy and passed health again.
- No firmware write occurred. Recovery was disabled. No reboot, unknown-kernel source build or real automatic firmware recovery was exercised by this acceptance.

The bundled `rknn-smi` prints device index and status in one cell and can return nonzero (observed 251) even for an Online / OK table. The parser requires an explicit recognizable table; Online health errors and unknown/failed queries never authorize flashing. A real inference self-test independently verifies readiness. Lite 1.1.0 also needs standard `/usr/lib` discovery paths; the base provides links to the verified SDK libraries.

## Evidence

- [Conditions and native image IDs](validation/2026-11-23/acceptance-summary.json)
- [Structured initialization events](validation/2026-11-23/acceptance-events.json)
- [Health, traffic prediction, upload/analyze/download](validation/2026-11-23/acceptance-api.json)
- [Lock, proxy cleanup and host-service restoration](validation/2026-11-23/acceptance-lifecycle.json)

Bootstrap policy tests: **14 passed**, including corrupted assets / driver rejection, ABI mismatch, actual SDK table format, nonstandard exit handling, unknown status refusal, explicit M.2 confirmation, shared-lock requirement, external-proxy protection, one-attempt recovery and no flash on Online inference failure. Firmware calls in these policy tests are mocked.

Per-model benchmark tables retain the prior 2026-11-20 native measurements; all 26 variants were not benchmarked again for this bootstrap change.
