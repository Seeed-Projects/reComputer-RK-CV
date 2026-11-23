# YOLO26 Segment · RK1820 / RK3588

[English](README.md) | 中文

在 RK3588 主机上，通过 PCIe 连接 RK1820 M.2 加速卡运行 **YOLO26 Segment RKNN3**。独立 Docker 项目，提供 Web 预览、图片/视频帧推理、视频上传分析和结果下载。

## 部署状态

本项目已通过 RK3588 + RK1820 M.2 实机验证。GitHub Actions 发布 ARM64 镜像到 GHCR；下方为远程镜像部署和本地构建两种方式。镜像发布状态见仓库 Actions。

- 主机: RK3588, Linux aarch64, Docker / Compose.
- 加速卡: RK1820 M.2, PCIe, firmware / RKNN3 Lite **1.1.0**.
- 量化: **INT8 W8A8**, `do_quantization=True`, no manually assigned W16A16 subgraphs.
- 模型尺寸: `yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`.
- 示例视频: `video/test.mp4`, supplied `30_1080p.mp4` (traffic).

## 目录结构

```text
rv182x_rk3588_yolo26_segment/
├── web_detection.py        # FastAPI + serialized preview/offline workers
├── engine.py               # RKNN3 inference + task decoding/rendering
├── index.html              # Live preview + local video analysis
├── project.json            # Variants, task and artifact metadata
├── model/                  # Paired .rknn/.weight files and labels
├── video/test.mp4
├── lib/                    # Matched ARM64 runtime libraries
├── rknn3-toolkit-lite-packages/
├── host/                   # Driver / proxy / matching M.2 firmware
├── scripts/setup_host.sh
├── conversion/             # Conversion config and provenance
├── benchmark.py
├── validation/             # Measured JSON + preview images
└── compose.yaml
```

## 可选：手动准备主机（旧版镜像 / 宿主机原生运行）

```bash
cd src/rv182x_rk3588_yolo26_segment
sudo apt-get install -y build-essential linux-headers-$(uname -r)
sudo bash scripts/setup_host.sh
```

仅在明确需要更新匹配的 M.2 固件时使用 `--flash-m2`；普通启动无需刷写。服务通过 RKNN3 proxy 访问 PCIe 卡。新版镜像启动入口可以自行加载驱动、复用已有 proxy 或管理自己的 proxy；无需预先安装 systemd 服务。

## 使用 GHCR 镜像快速部署

前提：Linux ARM64 RK3588、Docker、正确安装的 RK1820 M.2 PCIe 卡。镜像现在继承公共运行时层，自动检查/准备驱动和 proxy、确认 Online，并完成真实 W8A8 自检后启动模型。已验证内核为 `6.1.172-vendor-seeed-rk3588`；其它内核需匹配的 headers 和可用构建环境。

#### 步骤 A：拉取镜像

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolo26-segment:latest
```

#### 步骤 B：一键运行

先停止占用 8000 端口或正在使用加速卡的演示容器。运行命令不依赖宿主机项目目录。workspace 挂载可选；容器退出前可从 Web 下载结果。

```bash
sudo docker run --rm --name rv182x-rk3588-yolo26-segment \
  --privileged --net=host --ipc=host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolo26-segment:latest \
  python3 web_detection.py --variant yolo26n-seg --model_path model/yolo26n-seg.rknn --weight_path model/yolo26n-seg.weight --video video/test.mp4
```

访问方式：`http://<开发板IP>:8000`；接口文档：`http://<开发板IP>:8000/docs`。前台运行时按 Ctrl+C 退出。后台运行可在 `run` 后增加 `-d`。

型号可选：`yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`。修改 `--variant` 后重新创建容器；对于单文件模型，同步修改配套的 `--model_path` / `--weight_path`。

默认样例直接使用镜像内素材，无需映射摄像头或 DRM 节点。`--ipc=host` 用于 RKNN3 proxy / DMA 共享内存；推理在 RK1820 卡上执行。如需使用摄像头，可增加对应的 `--device /dev/video0:/dev/video0`，并将 `--video video/test.mp4` 替换为 `--camera_id 0`。

#### 自定义显示类别名称

默认使用COCO 80 类标签。`class_config.txt` 每行一个名称，必须保留原模型的类别顺序与数量。仅修改文字不会增加或改变模型的识别类别；新类别需要训练并转换对应模型。

```bash
sudo docker run --rm --name rv182x-rk3588-yolo26-segment \
  --privileged --net=host --ipc=host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  -v "$(pwd)/class_config.txt:/app/class_config.txt:ro" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-yolo26-segment:latest \
  python3 web_detection.py --variant yolo26n-seg --model_path model/yolo26n-seg.rknn --weight_path model/yolo26n-seg.weight --video video/test.mp4 --class_path /app/class_config.txt
```

## 构建镜像

从仓库根目录执行：

```bash
docker build --network host --platform linux/arm64 \
  -f docker/rv182x-rk3588/yolo26-segment.dockerfile \
  -t rv182x-rk3588-yolo26-segment:local src/rv182x_rk3588_yolo26_segment
```

依赖已由公共基础层提供。如需修改 PyPI / APT 源，请先按下方公共基础镜像说明重建基础层，再构建本模型。

## 一键运行

```bash
docker run -d --name rv182x-rk3588-yolo26-segment \
  --privileged --network host --ipc host --restart unless-stopped --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -e MODEL_VARIANT=yolo26n-seg \
  rv182x-rk3588-yolo26-segment:local
```

打开 **http://192.168.100.145:8000** (或设备 IP).

通过 `MODEL_VARIANT` 选择上面列出的尺寸，重新创建容器后生效。多个演示容器不要同时占用 8000 端口；切换项目前先停止当前演示。

```bash
cd src/rv182x_rk3588_yolo26_segment
MODEL_VARIANT=yolo26m-seg docker compose up -d --build
docker compose logs -f
docker compose down
```

`--privileged`、host 网络和 host IPC 用于 RKNN3 proxy、PCIe DMA 与共享内存。首次服务启动会加载模型，查看健康接口确认就绪。

## 类别名称与自定义模型

官方模型类别数为 80。可替换 `model/coco_80_labels_list.txt` 中的显示名称，每行一个名称并保留原顺序与行数；重命名不改变模型识别能力。通过 `--class_path` 指定自定义文件。同结构 RKNN 模型需要同名配套 `.rknn` / `.weight`；可通过 `--model_path` / `--weight_path` 指定。

## API 使用

交互式接口说明： `/docs`.

```bash
curl http://192.168.100.145:8000/api/health
curl -X POST http://192.168.100.145:8000/api/models/yolo26-segment/predict \
  -F file=@sample.jpg -F conf=0.25 -F iou=0.45
curl -X POST http://192.168.100.145:8000/api/predict \
  -F video=@sample.mp4 -F timestamp=1.0
curl -X POST http://192.168.100.145:8000/api/config \
  -H 'Content-Type: application/json' -d '{"obj_thresh":0.25,"nms_thresh":0.45}'
```

| Endpoint | 说明 |
|---|---|
| `GET /api/health` | 模型、运行库、调用计数与状态 |
| `GET/POST /api/config` | 检测阈值，范围 0–1 |
| `POST /api/predict` | 图片、指定时间的视频帧或当前预览帧；JSON 结果 |
| `GET /api/video_feed` | MJPEG preview |
| `GET /api/snapshot` | JPEG snapshot |
| `POST /api/video/upload` | multipart `file` |
| `GET /api/video/list` | 输入/输出文件列表 |
| `POST /api/video/analyze` | form `filename` |
| `GET /api/video/status` | 进度和错误 |
| `GET /api/video/download/{filename}` | 下载完成的视频 |

`file` 与 `video` 二选一；未传文件时使用预览帧。坐标对应原始图像。分类返回 Top-5；分割额外返回 `segments` 多边形；姿态额外返回 17 个 `keypoints`（x、y、置信度）。分类不使用检测阈值。参数错误返回 422，非法媒体 400，任务冲突 409，未就绪 503。

## 实机 Benchmark

仅填写这台 RK3588 + RK1820 M.2 上的成功测量。`—` 表示尚无通过的实测记录。

| Variant | Quant | NHWC input | Runtime FPS | Pipeline FPS | Runtime p95 ms | Warmup / frames | Evidence / status |
|---|---|---|---:|---:|---:|---:|---|
| yolo26n-seg | W8A8 | 1×640×640×3 | 17.23 | 3.93 | 67.44 | 5/30 | [JSON](validation/benchmark-yolo26n-seg.json) |
| yolo26s-seg | W8A8 | 1×640×640×3 | 14.29 | 2.58 | 82.22 | 5/30 | [JSON](validation/benchmark-yolo26s-seg.json) |
| yolo26m-seg | W8A8 | 1×640×640×3 | 6.94 | 2.19 | 164.00 | 5/30 | [JSON](validation/benchmark-yolo26m-seg.json) |

测试条件：RKNN3 Lite / 固件 1.1.0，core mask `0x1`，单进程串行推理，其他演示容器停止。Runtime FPS = 实际完成帧数 / 同步 Lite 调用累计耗时，包含 PCIe 传输与同步，不是纯 NPU 计算速度。Pipeline FPS = 实际完成帧数 / 整段测量耗时，包含视频解码、预处理、推理、任务后处理、绘制、最大 1280 宽 JPEG 编码；不包含 HTTP、网络或按视频原始帧率限速。Web UI 的实时播放速度受源视频 FPS 与其他处理开销影响。

```bash
mkdir -p validation
docker run --rm --privileged --network host --ipc host --stop-timeout 300 \
  -v /run/lock:/run/lock \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v "$PWD/validation:/app/validation" rv182x-rk3588-yolo26-segment:local \
  python benchmark.py --variant yolo26n-seg --warmup 5 --frames 30 \
  --output validation/benchmark-yolo26n-seg.json
```

速度验证不等于数据集精度认证。车流视频适合检测与车辆分割；ImageNet 分类类别不等于道路业务标签，姿态展示还需包含清晰近景人物的素材。

## 模型来源与转换

- [RKNN3 official example](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/yolo26_segment)
- 逐尺寸转换记录： `conversion/*.json`.

可复现转换命令与校准数据说明见 [conversion/README.md](conversion/README.md)。
- 模型与 weight 必须同名配套。不要混用 RKNN2/RKNN3 模型或不匹配的 1.2 runtime 与 1.1 固件。
- 标签文件： `model/coco_80_labels_list.txt`.
- 校验： `sha256sum -c SHA256SUMS`.

## 排错

健康检查失败时先查看 `docker logs`，确认宿主机驱动、卡 Online 状态和 proxy，再检查模型/weight 是否配套、选定尺寸是否已经生成。镜像加载失败且宿主机使用 overlayroot 时，Docker 和 containerd 的数据目录需落在支持 overlay2 的可写 ext4 分区。

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
