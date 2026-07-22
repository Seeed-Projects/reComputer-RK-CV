# rk3576_yolov8

基于 RK3576 / RKNN / MPP / RGA / ZLMediaKit 的多源输入、多码流输出、YOLOv8 实时推理工程。

当前工程支持：

- `V4L2` 摄像头输入
- `RTSP` 网络视频输入
- `localvideo` 本地视频文件循环输入
- 主码流 / 副码流 / AI 调试流三路 RTSP 输出
- `YOLOv8 RKNN` 推理、检测框叠加、类别标签、时间与 FPS OSD
- `libavformat` 优先的 RTSP/本地压缩流输入，`ffmpeg pipe` 作为兼容兜底

## 功能概览

- **输入源**
- `v4l2`：适合 USB 摄像头、本地采集
- `rtsp`：适合 IPC/NVR/网络视频流
- `localvideo`：适合本地视频回放、离线验证、循环推理

- **输出流**
- 主码流：`rtsp://<board-ip>:8554/live/camera`
- 副码流：`rtsp://<board-ip>:8554/live/camera_sub`
- AI 调试流：`rtsp://<board-ip>:8554/live/camera_ai`
- 可通过 `--streams main,sub,ai` 选择任意一路、两路或三路同时输出

- **分辨率策略**
- `camera` 主码流：等比例缩放 + 填充，不做拉伸
- `camera_sub` 副码流：等比例缩放 + 居中裁剪，不做拉伸
- `camera_ai` AI 调试流：等比例缩放 + 填充，不做拉伸

- **AI 推理**
- 固定输入 `640x640`
- 优先使用 `RGA letterbox` 预处理
- 支持 detection cache、PTS 匹配、AI overlay

## 部署说明

### 依赖环境

- RK3576 板卡运行环境
- RKNN Runtime
- Rockchip MPP
- RGA / `librga`
- ZLMediaKit 运行库
- FFmpeg 运行库或命令行工具

### 构建步骤

```bash
cd /home/parker/Projects/rk3576_yolov8tortsp_demo
chmod +x scripts/build-linux.sh
./scripts/build-linux.sh
```

构建产物：

- 可执行文件：`install/bin/rk3576_yolov8tortsp_demo`
- 运行库目录：`install/lib`
- 文档目录：`install/docs`
- 模型目录：`install/model`

### 运行前准备

```bash
cd install
export LD_LIBRARY_PATH="$(pwd)/lib:${LD_LIBRARY_PATH}"
```

### 模型放置

将 RKNN 模型放入：

```text
model/
  yolov8n_rk3576.rknn
  yolov8s_rk3576.rknn
  yolov8m_rk3576.rknn
```

如果未显式传 `--model`，程序会自动按如下顺序选择存在的模型：

1. `model/yolov8n_rk3576.rknn`
2. `model/yolov8s_rk3576.rknn`
3. `model/yolov8m_rk3576.rknn`

## 运行方式

### V4L2 摄像头输入

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source v4l2 \
  --device /dev/video0 \
  --model model/yolov8n_rk3576.rknn \
  --camera-width 1920 \
  --camera-height 1080 \
  --output-width 1280 \
  --output-height 720 \
  --fps 30 \
  --bitrate 4000000 \
  --perf-interval-ms 1000 \
  --rtsp-port 8554 \
  --rtsp-app live \
  --rtsp-stream camera
```

### RTSP 输入

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source rtsp \
  --rtsp-input-url "rtsp://192.168.100.101:554/live" \
  --rtsp-username "admin" \
  --rtsp-password "123456" \
  --model model/yolov8n_rk3576.rknn \
  --camera-width 1920 \
  --camera-height 1080 \
  --output-width 1280 \
  --output-height 720 \
  --fps 30 \
  --bitrate 4000000 \
  --perf-interval-ms 1000 \
  --rtsp-timeout-ms 5000 \
  --rtsp-reconnect-ms 1000 \
  --rtsp-port 8554 \
  --rtsp-app live \
  --rtsp-stream camera
```

### 本地视频循环输入

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source localvideo \
  --localvideo ../video/football.mp4 \
  --model model/yolov8n_rk3576.rknn \
  --camera-width 1920 \
  --camera-height 1080 \
  --output-width 1280 \
  --output-height 720 \
  --fps 30 \
  --bitrate 4000000 \
  --perf-interval-ms 1000 \
  --rtsp-port 8554 \
  --rtsp-app live \
  --rtsp-stream camera
```

### 详细日志

默认仅打印启动摘要；如需打印详细运行日志，增加：

```bash
--detail-info
```

### 导出 NPU 输入图

```bash
./bin/rk3576_yolov8tortsp_demo \
  --source localvideo \
  --localvideo ./videos/demo.mp4 \
  --model model/yolov8n_rk3576.rknn \
  --streams ai \
  --dump-ai-input-dir ./debug_ai_input \
  --dump-ai-input-every 30 \
  --detail-info
```

### 按需启用输出流

```bash
--streams ai
--streams main,ai
--streams main,sub
--streams main,sub,ai
```

说明：

- `--streams ai`：仅进行 AI 推理并输出 `camera_ai`
- `--streams main,ai`：输出主码流和 AI 调试流
- `--streams main,sub`：只做普通双码流，不加载 RKNN 模型
- 未指定时默认等价于 `--streams main,sub,ai`

### 查看帮助

```bash
./bin/rk3576_yolov8tortsp_demo --help
```

## 接口说明

### 核心命令行参数

- `--source <v4l2|rtsp|localvideo>`：输入源类型
- `--device <path>`：V4L2 设备节点
- `--rtsp-input-url <url>`：RTSP 地址
- `--rtsp-username <value>`：RTSP 用户名
- `--rtsp-password <value>`：RTSP 密码
- `--localvideo <path>`：本地视频文件路径
- `--model <path>`：RKNN 模型路径
- `--camera-width <value>` / `--camera-height <value>`：输入尺寸
- `--output-width <value>` / `--output-height <value>`：主码流输出尺寸
- `--sub-width <value>` / `--sub-height <value>`：副码流输出尺寸
- `--fps <value>`：编码目标帧率
- `--bitrate <value>`：主码流目标码率
- `--rtsp-port <value>`：本地 RTSP 服务端口
- `--rtsp-app <value>`：RTSP app 名称
- `--rtsp-stream <value>`：主码流名称
- `--sub-stream <value>`：副码流名称
- `--ai-stream <value>`：AI 调试流名称
- `--streams <items>`：启用哪些输出流，支持 `main`、`sub`、`ai` 的逗号组合
- `--rtsp-timeout-ms <value>`：RTSP 读超时
- `--rtsp-reconnect-ms <value>`：RTSP 重连间隔
- `--perf-interval-ms <value>`：PERF 输出周期
- `--detail-info`：打开详细日志
- `--help`：打印帮助

### PERF 含义

- `process`：分支内图像处理耗时
- `encode`：MPP 编码耗时
- `push`：RTSP 推流耗时
- `e2e`：分支内部从开始处理到推流完成的总耗时
- `fps`：按 wall-clock 统计窗口计算的真实分支吞吐率

## 项目结构简介
。，。
```text
rk3576_yolov8tortsp_demo
├── CMakeLists.txt
├── README.md
├── docs
├── include
│   ├── rk3576_demo
│   └── rk3576_yolo_demo
├── model
├── scripts
├── src
│   ├── ai          # RKNN 推理与前后处理
│   ├── app         # 应用入口与运行编排
│   ├── branch      # 三路分支输出、OSD、PERF
│   ├── camera      # V4L2 采集
│   ├── codec       # MPP 编解码
│   ├── common      # 通用类型、日志、线程队列
│   ├── pipeline    # V2 分发与处理管线
│   ├── rga         # 图像缩放与转换
│   ├── rtsp        # 本地 RTSP 发布
│   ├── source      # 输入源抽象与实现
│   ├── utils
│   └── watermark   # 时间/FPS/标签渲染
└── third_party
```

## 二次开发说明

### 1. 输入源扩展

- 统一接口在 `include/rk3576_yolo_demo/source/i_source.hpp`
- 工厂入口在 `src/source/source_factory.cpp`
- 新输入源应实现：
  - `Open()`
  - `Close()`
  - `ReadPacket()`
  - `Describe()`

### 2. 分支输出扩展

- 分支实现位于 `src/branch/branch_output.cpp`
- 当前支持三种几何策略：
  - `stretch`
  - `letterbox`
  - `center_crop`
- 几何关系统一在 `ComputeFrameTransform()` 中计算，便于输出缩放与 AI 框映射复用

### 3. AI 推理扩展

- RKNN 入口在 `src/ai/yolov8_engine.cpp`
- 预处理优先走 `RGA`
- 检测结果会进入 `DetectionFrame` 缓存，再由 AI 分支按 `PTS` 匹配叠加

### 4. 日志与调试

- 默认只输出启动摘要
- 详细运行日志通过 `--detail-info` 打开
- Logger 在 `include/rk3576_yolo_demo/common/logger.hpp`

## 项目部署建议

- 正式部署时为 `/dev/rga`、视频设备和相关运行库配置好权限
- 建议将模型、运行库和配置文件跟随 `install/` 一并打包
- RTSP 输入优先使用 `libavformat`，老环境自动回退到 `ffmpeg pipe`
- 若需要长期稳定运行，建议配合 systemd 或 supervisor 做拉起和重启管理

## 已知限制

- `camera_ai` 的框绘制和 watermark 叠加仍有部分 CPU 开销
- `e2e` 不是完整全链路延迟，只表示分支内部耗时
- `localvideo` 当前基于压缩流读取链路，主要用于离线验证和循环推理

## 后续开发建议

1. 为 `camera_ai` 继续降低 OSD CPU 开销，进一步引入图块缓存或更高效的混合路径
2. 增加真正的全链路 `pipeline_e2e` 指标，区分于当前分支 `e2e`
3. 继续完善 `localvideo` 场景下的播放节奏控制，使其更接近真实实时源
4. 为主码流/副码流增加更多独立编码参数配置能力
5. 如需更复杂的输入类型，可继续扩展 `source` 抽象层
