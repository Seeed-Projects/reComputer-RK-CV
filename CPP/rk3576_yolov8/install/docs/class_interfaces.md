# 类接口设计说明

## 目标

本文件说明 V2 工程当前规划的核心类接口，以及它们之间的职责关系。

## 设计原则

1. 输入、解码、分发、AI、编码、发布分层明确
2. 数据流通过统一结构传递
3. 单模块尽量只做一件事
4. 先定义接口，再逐步补齐实现

## 核心接口

### 1. `YoloRtspApplication`

职责：

- 解析 V2 配置
- 选择输入源类型
- 组织兼容基线或 V2 架构入口
- 统一打印运行模式和迭代状态

当前状态：

- 已实现新入口和分支初始化
- 当前以 decode-only 方式复用稳定的兼容采集解码链
- 已支持将解码输出镜像成 `UnifiedFrame` 并送入 `FrameDispatcher`
- 已由 V2 `BranchOutput` 统一生成主码流、副码流和 AI 调试流

### 2. `IInputSource`

职责：

- 屏蔽本地输入和网络输入差异
- 提供统一的打开、关闭、能力查询接口

建议接口：

- `Open()`
- `Close()`
- `Describe()`
- `SupportsCompressedPackets()`
- `SupportsRawFrames()`

当前实现规划：

- `V4L2SourceAdapter`
- `RtspSource`

### 3. `V4L2SourceAdapter`

职责：

- 复用已有 `V4L2Camera`
- 将本地采集能力接入 V2 输入层
- 为后续多格式 V4L2 输入演进提供适配点

当前状态：

- 第一轮实现为适配器骨架
- 仍以兼容基线为主

### 4. `RtspSource`

职责：

- 作为网络输入源的统一入口
- 后续内部对接 `FFmpeg/libavformat`
- 负责拉流、重连、网络协议配置

当前状态：

- 第一轮仅占位
- 后续实现 `RTSP -> AVPacket -> MPP Decoder`

### 5. `Yolov8Engine`

职责：

- 加载 RKNN 模型
- 管理推理输入输出
- 执行 YOLOv8 推理

V2 约束：

- 输入固定为 `640x640`
- 模型从 `model/` 目录加载

当前状态：

- 当前仍为轻量接口
- 已支持自动探测 `model/` 下的默认模型
- 已接入 `rknn_init/query/run`
- 已支持从解码帧做 `640x640` CPU letterbox 预处理并执行真实推理
- 已接入 YOLOv8 量化输出后处理与 NMS
- 当前会输出 `DetectionBox` 列表并供 `camera_ai` 分支叠加彩色框线

### 6. `StreamBranch`

职责：

- 描述主码流、副码流、AI 调试码流的共性
- 为每个分支定义：
  - 输出尺寸
  - 是否叠加 AI 结果
  - 发布路径

当前规划的具体分支：

- `MainStreamBranch`
- `SubStreamBranch`
- `AiDebugBranch`

当前状态：

- 当前主要负责分支队列与 worker 生命周期
- 真实 `sub/ai` 输出已先通过分支桥接层落地，避免 DMA 生命周期在异步队列中失控

### 6.1 `BranchOutput`

职责：

- 复用采集解码链输出的 `DecodedFrame`
- 为主码流、副码流和 AI 调试流各自维护独立的 `MppEncoder + RtspServer`
- 统一生成真实 `camera`、`camera_sub` 和 `camera_ai` 流

当前状态：

- 已落地 `main/sub/ai` 三路 V2 编码发布链
- 若 `RGA` 不可用，自动降级到软件 copy，并在必要时切换到源分辨率输出
- AI 调试流已接入 RKNN 检测框、类别名和 FPS 文本叠加

### 7. `FrameDispatcher`

职责：

- 接收统一帧
- 复制或引用投递到各分支队列
- 为 AI 元数据与原始帧建立关联

当前状态：

- 已实现基础广播分发
- 已支持输出分支统计
- 当前接入的是兼容基线解码帧的“元数据镜像”，尚未托管 DMA buffer 生命周期

### 8. `PublisherChannel`

职责：

- 抽象推流输出
- 后续支持：
  - RTSP 发布
  - MediaMTX 发布
  - 可选 RTMP

## 复用的原项目类

以下类当前仍作为稳定基线保留：

- `PipelineApp`
- `V4L2Camera`
- `MppDecoder`
- `MppEncoder`
- `RgaProcessor`
- `RtspServer`
- `WatermarkRenderer`
- `SystemMonitor`

其中 `PipelineApp` 当前已补充解码帧观察回调和 decode-only 模式，用于把稳定采集解码链中的 `DecodedFrame` 桥接到 V2 新总线；主码流编码和推流已由 V2 `BranchOutput` 接管。

## 数据结构规划

建议统一保留以下数据结构：

- `SourceDescriptor`
- `CompressedPacket`
- `UnifiedFrame`
- `DetectionBox`
- `DetectionFrame`
- `StreamProfile`
- `BranchStats`

当前第一轮代码已落地的重点是：

- 新应用入口配置
- 输入源抽象骨架
- AI 引擎占位接口

当前新增落地的重点是：

- `UnifiedFrame` 桥接
- `FrameDispatcher`
- `StreamBranch`
- 分支统计与基础 worker 生命周期
- `BranchOutput`
- `sub/ai` 两路独立 RTSP 发布

## 后续类接口扩展建议

### 第二轮

- `FramePool`
- `DetectionMetadataBus`
- `AiPreprocessor`
- `Yolov8PostProcessor`

### 第三轮

- `MediaMtxPublisher`
- `FfmpegRtspSource`
- `FfmpegRtspPublisher`
- `BranchWorker`
