# Shared RK1820 runtime for RK3588

所有 `rv182x-rk3588-*` CV 模型镜像共享本目录构建的 ARM64 基础层：

`ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0`

普通用户只拉取和运行模型镜像，Docker 自动获取其基础层，无需额外启动一个“检查容器”。模型命令和尺寸仍由各项目 README 指定。

## Supported hardware / compatibility

- RK3588 Linux ARM64 host + **RK1820 M.2 PCIe** accelerator, PCI ID `1d87:182a`.
- SDK, Lite wheel, libraries and M.2 firmware pinned to **1.1.0**. Do not replace individual components with 1.2 assets.
- One card per host. Concurrent managed model containers are excluded by the shared lock. Stop legacy unmanaged inference users before starting a model.
- Bundled driver profile: `6.1.172-vendor-seeed-rk3588`, validated on reComputer RK3588 Armbian 26.8.7. Other kernels attempt a source build with matching vendor headers/toolchain; compatibility is not guaranteed merely by being RK3588.
- Firmware is specific to `ext-m2`; a matching PCI ID does not establish the board type. Other board types are not authorized for recovery by this image.

## One-command startup

Example: YOLOv5n in browser upload mode (no camera or workspace directory required).

```bash
sudo docker run --rm --name rv182x-rk3588-yolov5 \
  --privileged --net=host --ipc=host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolov5:latest \
  python3 web_detection.py --variant yolov5n \
    --model_path model/yolov5n.rknn --weight_path model/yolov5n.weight \
    --camera_id -1
```

Open `http://<BOARD_IP>:8000`. Replace `--camera_id -1` with `--video video/test.mp4` for the bundled traffic video, or `--camera_id 0` plus `--device /dev/video0:/dev/video0` for a supported camera. RAFT-Stereo requires a rectified left/right pair instead; see its model guide.

The entrypoint verifies asset hashes, discovers the card, acquires its lock, loads the matching driver (or builds it), creates the container device node, reuses or starts the proxy, waits for **Online / OK**, and runs a real MobileNetV2 W8A8 sample inference before launching the application. Structured logs use `component=rk182x-init`; successful startup ends with `SELF_TEST_PASSED` and `MODEL_START_READY`.

`--privileged` allows loading a module into the **host kernel**. `/lib/modules` and `/usr/src` remain read-only. Containers do not install host services or write the host root filesystem. `/run/lock` provides a host-shared lock for the entire model lifetime. A proxy started by the container ends with it; an existing external proxy is reused and not killed. The kernel module remains loaded after exit.

No full host `/dev` mount is necessary: the entrypoint creates the matching device node from sysfs. Workspace persistence is optional: add `-v "$PWD/workspace:/app/workspace"` if uploads/results must survive container removal. Otherwise download results before exiting. This does not affect browser upload/analysis support.

## Read-only diagnosis

```bash
sudo docker run --rm --privileged --net=host --ipc=host \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0 doctor
```

`doctor` reports PCI detection, kernel, registered driver, proxy and card status; it does not load a driver, start a proxy, run inference or flash firmware. Exit 0 means Online with a reachable proxy, not proof of model inference. Normal model startup supplies that additional self-test.

| Error code | Action |
|---|---|
| `PCI_DEVICE_NOT_FOUND` | Check card installation, power, PCIe configuration and host enumeration. Software cannot repair an absent PCI endpoint. |
| `KERNEL_HEADERS_MISSING` / `DRIVER_BUILD_FAILED` / `DRIVER_LOAD_FAILED` | Install matching vendor headers, provide both read-only mounts, inspect the compiler/ABI details. Never force-load a module from another kernel. |
| `DEVICE_PERMISSION_DENIED` | Use the documented privileged configuration. |
| `DEVICE_BUSY` | Stop the other managed model container before switching. |
| `PROXY_START_FAILED` | Inspect logs and SDK compatibility; check port 18898. |
| `CARD_OFFLINE` | Default startup refuses to flash. Check power/PCIe and use the explicit maintenance procedure only when appropriate. |
| `CARD_STATUS_QUERY_FAILED` / `CARD_STATUS_UNKNOWN` / `CARD_HEALTH_ERROR` | Inspect SMI output, SDK compatibility and card health. These errors never authorize firmware recovery. |
| `SELF_TEST_FAILED` | Card was Online but inference failed. Inspect matching runtime/firmware and other card users; this error never triggers flashing. |
| `ASSET_CHECKSUM_FAILED` | Replace the image/assets with a verified published build. |

## Explicit M.2 firmware maintenance

The image **contains** the matched firmware but default startup **does not write it**. It cannot guarantee every Docker command succeeds on arbitrary hardware/kernel/power conditions.

After confirming the physical RK1820 M.2 board and stopping every inference user and external proxy, add these options **before the image name** to the normal model command:

```bash
-e RK182X_ALLOW_FIRMWARE_RECOVERY=1 \
-e RK182X_BOARD=ext-m2 \
```

Keep the `/run/lock` mount and 300-second stop timeout. Only an Offline card with a container-owned proxy can enter recovery. The proxy is stopped, firmware hashes are checked, and the SDK upgrade tool is invoked once with the selected PCI address and reset option. The proxy is then restarted and both Online/OK and inference must pass. There is no retry loop and Online self-test/model failures never cause flashing. SIGTERM/interrupt requests during the write are deferred until it finishes; power loss, SIGKILL and host failure cannot be made safe by a container. Remove the recovery options for routine operation.

An existing host proxy/service must be stopped by its administrator before recovery; the container intentionally does not stop it. The image does not create or change systemd services. An already installed compatible host service can coexist with ordinary inference through proxy reuse.

## Build / publish

From the repository root:

```bash
docker build --platform linux/arm64 \
  -f docker/rv182x-rk3588/runtime.dockerfile \
  -t ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0 \
  runtime/rv182x-rk3588
docker build --platform linux/arm64 \
  --build-arg RK182X_BASE_IMAGE=ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0 \
  -f docker/rv182x-rk3588/yolov5.dockerfile \
  -t rv182x-rk3588-yolov5:local src/rv182x_rk3588_yolov5
```

Optional `APT_MIRROR` / `PIP_INDEX_URL` arguments apply to the **base build**. Models inherit dependencies; changing these arguments on a model build does not reinstall packages. QAClip retains an asset preparation stage to reconstruct and verify its chunked weight without copying both chunks and full weight into the final layer.

`.github/workflows/docker-build-rv182x.yml` runs policy tests, builds/publishes the base, then builds all 11 models against that **immutable base digest**. Every image gets `latest` and commit tags; the base also gets SDK tag `1.1.0`. Model OCI labels record source revision and base digest. Manual dispatch can select one model or all; the base is always resolved first. Pull requests run tests and build the base without publishing; model publication is restricted to push/manual runs.

Bundled assets are covered by `ASSET_SHA256SUMS`, driver provenance by `drivers/manifest.json`, and upstream notices by `NOTICE.md`. Policy tests:

```bash
python3 -m unittest discover -s runtime/rv182x-rk3588 -p test_runtime.py -v
```

## Validation scope

See [board acceptance](VALIDATION.md) for this shared-runtime release. Only one model is required for board acceptance. Existing per-model benchmark tables retain their previous measurements and are not represented as new benchmarks of all variants. Automatic firmware recovery is an explicit maintenance path; a normal successful boot does not exercise a flash.
