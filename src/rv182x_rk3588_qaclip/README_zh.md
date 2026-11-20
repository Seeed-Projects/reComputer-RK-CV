# QA-CLIP ViT-L/14 · RK1820 / RK3588

[English](README.md) | 中文

在 RK3588 主机上，通过 PCIe 连接 RK1820 M.2 加速卡运行 **QA-CLIP ViT-L/14 RKNN3**。独立 Docker 项目，提供 Web 预览、图片/视频帧推理、视频上传分析和结果下载。

## 部署状态

本项目已通过 RK3588 + RK1820 M.2 实机验证。GitHub Actions 发布 ARM64 镜像到 GHCR；下方为远程镜像部署和本地构建两种方式。镜像发布状态见仓库 Actions。

- 主机: RK3588, Linux aarch64, Docker / Compose.
- 加速卡: RK1820 M.2, PCIe, firmware / RKNN3 Lite **1.1.0**.
- 量化: **INT8 W8A8**, `do_quantization=True`, no manually assigned W16A16 subgraphs.
- 模型尺寸: `qaclip-vit-l14`.
- 示例视频: `video/test.mp4`, supplied `30_1080p.mp4` (traffic).

## 目录结构

```text
rv182x_rk3588_qaclip/
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

## 主机准备（在 RK3588 上）

```bash
cd src/rv182x_rk3588_qaclip
sudo apt-get install -y build-essential linux-headers-$(uname -r)
sudo bash scripts/setup_host.sh
```

仅在明确需要更新匹配的 M.2 固件时使用 `--flash-m2`；普通启动无需刷写。服务通过本机 RKNN3 proxy 访问 PCIe 卡，不是在 RK3588 内置 NPU 上运行。

## 使用 GHCR 镜像快速部署

前提：RK3588 已安装 Docker，RK1820 M.2 PCIe 卡 Online，且已按下方“主机准备”安装匹配 SDK 1.1.0 的驱动和 RKNN3 proxy。镜像面向 Linux ARM64。

#### 步骤 A：拉取镜像

```bash
sudo docker pull ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-qaclip:latest
```

#### 步骤 B：一键运行

先停止占用 8000 端口或正在使用加速卡的演示容器。在可写目录执行；`workspace` 用于保存上传文件和分析结果。

```bash
sudo docker run --rm --name rv182x-rk3588-qaclip \
  --privileged --net=host --ipc=host \
  -e PYTHONUNBUFFERED=1 \
  -e RKNN_LOG_LEVEL=0 \
  -v "$(pwd)/workspace:/app/workspace" \
  ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-qaclip:latest \
  python3 web_detection.py --variant qaclip-vit-l14 --video video/test.mp4
```

访问方式：`http://<开发板IP>:8000`；接口文档：`http://<开发板IP>:8000/docs`。前台运行时按 Ctrl+C 退出。后台运行可在 `run` 后增加 `-d`。

型号可选：`qaclip-vit-l14`。修改 `--variant` 后重新创建容器；对于单文件模型，同步修改配套的 `--model_path` / `--weight_path`。

默认样例直接使用镜像内素材，无需映射摄像头或 DRM 节点。`--ipc=host` 用于 RKNN3 proxy / DMA 共享内存；推理在 RK1820 卡上执行。如需使用摄像头，可增加对应的 `--device /dev/video0:/dev/video0`，并将 `--video video/test.mp4` 替换为 `--camera_id 0`。

QAClip 的两个 RKNN 编码器由程序自动加载；候选文本通过 `POST /api/prompts` 更新，无需修改类别文件。

## 构建镜像

从仓库根目录执行：

```bash
docker build --network host --platform linux/arm64 \
  -f docker/rv182x-rk3588/qaclip.dockerfile \
  -t rv182x-rk3588-qaclip:local src/rv182x_rk3588_qaclip
```

如果 PyPI 下载较慢，可添加 `--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn`。

## 一键运行

```bash
docker run -d --name rv182x-rk3588-qaclip \
  --privileged --network host --ipc host --restart unless-stopped \
  -e MODEL_VARIANT=qaclip-vit-l14 \
  -v "$PWD/workspace:/app/workspace" \
  rv182x-rk3588-qaclip:local
```

打开 **http://192.168.100.145:8000** (或设备 IP).

通过 `MODEL_VARIANT` 选择上面列出的尺寸，重新创建容器后生效。多个演示容器不要同时占用 8000 端口；切换项目前先停止当前演示。

```bash
cd src/rv182x_rk3588_qaclip
MODEL_VARIANT=qaclip-vit-l14 docker compose up -d --build
docker compose logs -f
docker compose down
```

`--privileged`、host 网络和 host IPC 用于 RKNN3 proxy、PCIe DMA 与共享内存。首次服务启动会加载模型，查看健康接口确认就绪。

## 图文匹配与候选文本

本项目的文本和视觉两个 RKNN 编码器均采用 W8A8；外置词嵌入表在 CPU 上以 FP32 使用。文本长度固定为 128 tokens（含特殊标记），超长候选文本返回 422。候选文本先编码并缓存，逐帧运行视觉编码器并计算相似度；更新候选文本会重新编码。Benchmark 的 FPS 对应固定候选文本缓存后的逐帧处理，文本初始化耗时单独保存在 JSON 的 `text_setup_ms` 中。预览中的 prompt # 与界面/API 的候选文本编号一致，可点击 Score current frame 查看中文类别和分数。

```bash
curl -X POST http://192.168.100.145:8000/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"prompts":["一辆汽车","一辆卡车","一个行人","一条公路"]}'
```

## API 使用

交互式接口说明： `/docs`.

```bash
curl http://192.168.100.145:8000/api/health
curl -X POST http://192.168.100.145:8000/api/models/qaclip/predict \
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
| qaclip-vit-l14 | W8A8 | 1×224×224×3 | 6.45 | 5.26 | 155.36 | 5/30 | [JSON](validation/benchmark-qaclip-vit-l14.json) |

测试条件：RKNN3 Lite / 固件 1.1.0，core mask `0x1`，单进程串行推理，其他演示容器停止。Runtime FPS = 实际完成帧数 / 同步 Lite 调用累计耗时，包含 PCIe 传输与同步，不是纯 NPU 计算速度。Pipeline FPS = 实际完成帧数 / 整段测量耗时，包含视频解码、预处理、推理、任务后处理、绘制、最大 1280 宽 JPEG 编码；不包含 HTTP、网络或按视频原始帧率限速。Web UI 的实时播放速度受源视频 FPS 与其他处理开销影响。

```bash
mkdir -p validation
docker run --rm --privileged --network host --ipc host \
  -v "$PWD/validation:/app/validation" rv182x-rk3588-qaclip:local \
  python benchmark.py --variant qaclip-vit-l14 --warmup 5 --frames 30 \
  --output validation/benchmark-qaclip-vit-l14.json
```

速度验证不等于数据集精度认证。车流视频适合检测与车辆分割；ImageNet 分类类别不等于道路业务标签，姿态展示还需包含清晰近景人物的素材。

## 模型来源与转换

- [RKNN3 official example](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/QAClip)
- 逐尺寸转换记录： `conversion/*.json`.

可复现转换命令与校准数据说明见 [conversion/README.md](conversion/README.md)。
- 模型与 weight 必须同名配套。不要混用 RKNN2/RKNN3 模型或不匹配的 1.2 runtime 与 1.1 固件。
- 标签文件： `model/labels.txt`.
- 校验： `sha256sum -c SHA256SUMS`.

## 排错

健康检查失败时先查看 `docker logs`，确认宿主机驱动、卡 Online 状态和 proxy，再检查模型/weight 是否配套、选定尺寸是否已经生成。镜像加载失败且宿主机使用 overlayroot 时，Docker 和 containerd 的数据目录需落在支持 overlay2 的可写 ext4 分区。

## 大权重分块说明

297 MiB 的视觉权重以四个小于 100 MiB 的 `.weight.part-*` 文件保存在普通 Git 中，避免 Git LFS 存储连接和额度问题。Docker 构建自动合并，并按 `model/qaclip-vision.parts.json` 与 `SHA256SUMS` 校验；运行镜像仅保留合并后的标准 `.weight`。源码直接运行或检查全量校验和前，先执行：

```bash
python3 scripts/prepare_assets.py
sha256sum -c SHA256SUMS
```

分块仅改变文件存储方式，合并后的权重与实测 W8A8 模型 SHA256 完全一致。
