# RAFT-Stereo · RK1820 / RK3588

[English](README.md) | 中文

RK3588 通过 PCIe 连接 RK1820 M.2，使用 RKNN3 进行双目视差估计。独立 Docker 项目，提供同风格 Web 界面、成对图片上传、视差可视化与 JSON API。

## 部署状态与核心功能

- 已完成本地整合与实机验证，ARM64 镜像由 GitHub Actions 发布到 GHCR；发布状态见仓库 Actions。
- RKNN3 Toolkit / Lite / firmware **1.1.0**, target `rk1820`, core mask `0x1`.
- **INT8 W8A8**, `do_quantization=True`, 无手工 W16A16 子图.
- 包含全部四个官方变体：eth3d、middlebury、sceneflow、realtime。
- 每次调用输入两张 160×240 RGB 图片；等比例缩放并在右下方补黑。
- 当前导出固定 1 次 GRU 迭代，与官方导出脚本默认值一致。此速度不能与 32 次迭代配置直接比较；更多迭代通常会改变速度和效果，须重新导出和转换。

输入必须是同一时刻、同一场景、已校正的真实左右目图片。单路车流视频不能构成双目输入。本项目使用官方 `xt_left.jpg` / `xt_right.jpg` 验证；用户提供的视频保存在 `video/test.mp4`，不作为双目验证输入。若要验证道路双目视频，需要同步、校正的左右目视频。当前服务提供成对图片接口。视差没有结合基线和焦距转换为米制距离。

## 目录结构

```text
rv182x_rk3588_raft_stereo/
├── web_detection.py       # Paired-input FastAPI service
├── special_engine.py      # RKNN3 inference + disparity visualization
├── index.html             # Left/right upload + preview
├── project.json           # Four variants and iteration count
├── model/                 # .rknn/.weight pairs + xt_left/right.jpg
├── lib/
├── rknn3-toolkit-lite-packages/
├── host/                  # SDK driver / proxy / M.2 firmware
├── scripts/setup_host.sh
├── conversion/
├── benchmark.py
├── verify_stereo.py
├── validation/
└── compose.yaml
```

## 可选：手动准备主机（旧版镜像 / 宿主机原生运行）

```bash
cd src/rv182x_rk3588_raft_stereo
sudo apt-get install -y build-essential linux-headers-$(uname -r)
sudo bash scripts/setup_host.sh
```

普通启动无需刷写固件。仅在明确需要更新匹配的 M.2 固件时使用 `--flash-m2`。

## 使用 GHCR 镜像快速部署

前提：Linux ARM64 RK3588、Docker、正确安装的 RK1820 M.2 PCIe 卡。镜像现在继承公共运行时层，自动检查/准备驱动和 proxy、确认 Online，并完成真实 W8A8 自检后启动模型。已验证内核为 `6.1.172-vendor-seeed-rk3588`；其它内核需匹配的 headers 和可用构建环境。

#### 步骤 A：拉取镜像

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-raft-stereo:latest
```

#### 步骤 B：一键运行

先停止占用 8000 端口或正在使用加速卡的演示容器。运行命令不依赖宿主机项目目录。workspace 挂载可选；容器退出前可从 Web 下载结果。

```bash
sudo docker run --rm --name rv182x-rk3588-raft-stereo \
  --privileged --net=host --ipc=host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-raft-stereo:latest \
  python3 web_detection.py --variant raftstereo-eth3d-160x240
```

访问方式：`http://<开发板IP>:8000`；接口文档：`http://<开发板IP>:8000/docs`。前台运行时按 Ctrl+C 退出。后台运行可在 `run` 后增加 `-d`。

型号可选：`raftstereo-eth3d-160x240`, `raftstereo-middlebury-160x240`, `raftstereo-sceneflow-160x240`, `raftstereo-realtime-160x240`。修改 `--variant` 后重新创建容器；对于单文件模型，同步修改配套的 `--model_path` / `--weight_path`。

默认样例直接使用镜像内素材，无需映射摄像头或 DRM 节点。`--ipc=host` 用于 RKNN3 proxy / DMA 共享内存；推理在 RK1820 卡上执行。页面自动显示内置左右目样例；上传输入必须是同步、校正的同尺寸左右目图片。当前不接收单目视频，也未实现双目摄像头实时采集。

四个型号是可独立选择的完整模型。当前输出视差，未换算米制距离；固定一次迭代。

## 构建镜像

```bash
# Repository root
docker build --network host --platform linux/arm64 \
  -f docker/rv182x-rk3588/raft-stereo.dockerfile \
  -t rv182x-rk3588-raft-stereo:local src/rv182x_rk3588_raft_stereo
```

国内网络可追加 `--build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

## 一键运行与选择型号

```bash
docker run -d --name rv182x-rk3588-raft-stereo \
  --privileged --network host --ipc host --restart unless-stopped --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e MODEL_VARIANT=raftstereo-eth3d-160x240 \
  rv182x-rk3588-raft-stereo:local
```

打开 **http://192.168.100.145:8000** (或设备 IP).

可选型号： `raftstereo-eth3d-160x240`, `raftstereo-middlebury-160x240`, `raftstereo-sceneflow-160x240`, `raftstereo-realtime-160x240`.

```bash
cd src/rv182x_rk3588_raft_stereo
MODEL_VARIANT=raftstereo-realtime-160x240 docker compose up -d --build
docker compose logs -f
docker compose down
```

通过环境变量选择后需重新创建容器。多个演示共用 8000 端口，切换前先停止当前容器。特权、host 网络与 IPC 用于 PCIe DMA、宿主机 RKNN3 proxy 和共享内存。

## API

```bash
curl http://192.168.100.145:8000/api/health
curl -X POST http://192.168.100.145:8000/api/models/raft-stereo/predict \
  -F left=@left.jpg -F right=@right.jpg
curl http://192.168.100.145:8000/api/sample -o disparity.jpg
```

| Endpoint | 说明 |
|---|---|
| `GET /api/health` | 型号、迭代次数与调用计数 |
| `POST /api/predict` | 与模型专属接口相同；multipart left/right |
| `POST /api/models/raft-stereo/predict` | 成对图片推理；左右图尺寸须相同 |
| `GET /api/sample` | 运行官方样例，返回 JPEG 视差预览 |
| `GET /docs` | OpenAPI |

预测返回 `predictions[0].visualization_png`（Base64 PNG）、视差 minimum/maximum、valid_fraction，以及 `timing.runtime_ms` / `total_ms`。缺少任一输入返回 422；无效媒体或尺寸不匹配返回 400。

## 实机 Benchmark

输入为官方左右目图片对，两个输入均为 NHWC `1×160×240×3`，core mask `0x1`，每次调用 1 次迭代；其他推理演示停止。每帧表示完成一对左右目图片的视差推理。

| Variant | Quant | Runtime FPS | Pipeline FPS | Runtime p95 ms | Warmup / pairs | Evidence |
|---|---|---:|---:|---:|---:|---|
| raftstereo-eth3d-160x240 | W8A8 | 4.75 | 1.84 | 211.66 | 5/30 | [JSON](validation/benchmark-raftstereo-eth3d-160x240.json) |
| raftstereo-middlebury-160x240 | W8A8 | 4.76 | 1.87 | 210.81 | 5/30 | [JSON](validation/benchmark-raftstereo-middlebury-160x240.json) |
| raftstereo-sceneflow-160x240 | W8A8 | 4.77 | 1.86 | 210.28 | 5/30 | [JSON](validation/benchmark-raftstereo-sceneflow-160x240.json) |
| raftstereo-realtime-160x240 | W8A8 | 16.48 | 2.56 | 61.58 | 5/30 | [JSON](validation/benchmark-raftstereo-realtime-160x240.json) |

Runtime FPS 包含同步 Lite 调用、PCIe 传输与同步；Pipeline FPS 还包含两张 JPEG 的读取/解码、缩放补边、视差着色及最大 1280 宽 JPEG 编码。不包含 HTTP 和网络。FPS 使用完成的对数除以对应累计实测耗时。没有成功记录的型号显示 `—`。

```bash
mkdir -p validation
docker run --rm --privileged --network host --ipc host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v "$PWD/validation:/app/validation" rv182x-rk3588-raft-stereo:local \
  python benchmark.py --variant raftstereo-eth3d-160x240 \
  --warmup 5 --frames 30 --output validation/benchmark-raftstereo-eth3d-160x240.json
```

## 模型转换与来源

参考本项目 `conversion/README.md`，使用 x86 RKNN3 Toolkit 1.1.0 转换。导出使用固定 `160×240` 和 `iters=1`；所有四个 checkpoint 均按 W8A8 转换。原官方转换脚本中的 i8 选项实际配置 W8A16，本项目没有沿用此混合精度配置。

- [RKNN3 official RAFT-Stereo example](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/RAFT-Stereo)
- 每个型号的转换参数和 ONNX 校验值： `conversion/*.json`.
- 模型和同名 weight 必须配套。校验运行文件： `sha256sum -c SHA256SUMS`.

## 排错

先检查 `docker logs`、宿主机驱动、加速卡 Online 状态和 RKNN3 proxy。左右图必须同尺寸且经过双目标定校正；色彩图只反映视差，不能代替有真值的误差评估。overlayroot 系统应将 Docker/containerd 数据目录放在可写 ext4 分区。

## 公共基础镜像与自动初始化

所有模型层继承 `ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0`。CI 先发布基础镜像，再按其不可变 digest 构建模型镜像。固件、SDK 工具、驱动、自检模型均在基础层，无需临时下载。

默认流程：校验资源 → 发现单张 RK1820 → 加载/按需编译驱动 → 复用/启动 proxy → 等待 Online / OK → MobileNetV2 W8A8 样例自检 → 启动本模型。只读诊断入口为镜像末尾的 `doctor`。

- `/run/lock` 共享挂载用于跨容器互斥；切换模型前停止旧容器。此版本支持单主机单 RK1820 卡。
- `/lib/modules` 和 `/usr/src` 只读挂载提供其它内核的编译信息。已附带本次板卡内核的已验证驱动；缺 headers、ABI 不匹配或构建失败时会给出明确错误，不强制加载模块。
- 内核驱动作用于宿主机；运行命令需要 `--privileged`。驱动加载后会在容器创建对应设备节点，不必把整个宿主机 `/dev` 挂载进来。
- 入口只清理自己启动的 proxy；复用的外部 proxy 不会被终止。容器内维护 proxy 时，容器退出会一并结束它。
- 默认不刷写固件。常规 Offline、驱动错误和模型自检失败不会被当作可以任意刷卡的理由。
- 上传、离线分析和下载不依赖 workspace 挂载；如需保留容器退出后的文件，可增加 `-v "$PWD/workspace:/app/workspace"`。

只有确认板型、停止全部推理和外部 proxy 后，才可在启动命令中显式增加 `-e RK182X_ALLOW_FIRMWARE_RECOVERY=1 -e RK182X_BOARD=ext-m2`。该设置只在卡 Offline 时尝试一次匹配固件恢复，正常卡直接跳过；需要共享锁挂载。不要在恢复过程中断电或强制终止容器，停止宽限期应保留为 300 秒。

公共运行时本地构建（仓库根目录）：

```bash
docker build --platform linux/arm64 -f docker/rv182x-rk3588/runtime.dockerfile \
  -t ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0 \
  runtime/rv182x-rk3588
```

国内镜像源参数 `APT_MIRROR` / `PIP_INDEX_URL` 作用于基础镜像；模型构建通过 `--build-arg RK182X_BASE_IMAGE=<基础镜像引用>` 选择基础层。普通部署只需拉取模型镜像，Docker 自动取得基础层。

本页 benchmark 是之前完成的全部型号原始实测，未因本次启动器改造重测全部模型；启动器初始化和自检时间不属于表内 Runtime FPS。当前版本仅按本次验收要求选择一个模型进行板卡端到端复验。

完整流程与限制见仓库 [公共运行时说明](../../../runtime/rv182x-rk3588/README.md)。
