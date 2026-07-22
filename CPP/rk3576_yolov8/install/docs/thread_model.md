# 线程模型设计说明

## 目标

本文件描述 V2 的推荐线程模型，以及第一轮代码迁移中实际落地的范围。

## V2 推荐线程模型

### 1. Source Thread

职责：

- 本地 V4L2 采集或 RTSP 拉流
- 输出：
  - 压缩包
  - 或原始帧

### 2. Decode / Normalize Thread

职责：

- 压缩流走 `MPP decode`
- 原始流走 `RGA normalize`
- 统一输出：
  - `NV12 + FD`

### 3. Dispatcher Thread

职责：

- 接收统一帧
- 投递到：
  - 主码流分支
  - 副码流分支
  - AI 分支

### 4. Main Encode Thread

职责：

- 处理主码流
- 缩放/OSD
- 编码
- 发布

### 5. Sub Encode Thread

职责：

- 处理副码流
- `640x480` 输出
- 保持比例后补边
- 编码
- 发布

### 6. AI Infer Thread

职责：

- 接收统一帧
- `RGA letterbox -> 640x640`
- 送入 RKNN 推理池
- 产出检测元数据

### 7. AI Debug Encode Thread

职责：

- 将检测元数据映射回原图
- 绘制框线、FPS、时延信息
- 编码并发布 AI 调试流

### 8. Publisher Thread Pool

职责：

- 发送编码后码流到发布端
- 后续可扩展多种协议和多路推送目标

## 当前已落地情况

当前新工程尚未实现完整的多线程 V2 架构，而是采用分阶段落地：

1. **文档先行**
   - 线程模型已经固定
2. **代码骨架先行**
   - 输入源、AI 引擎、应用入口已拆分
3. **运行基线保留**
   - 仍可落回原有 `PipelineApp` 单链路运行

当前已经完成：

- **线程模型设计完成**
- **线程模型代码骨架开始搭建**
- **分发器和三分支 worker 已初始化**
- **兼容基线解码帧已可镜像进入 V2 总线**
- **主码流、副码流和 AI 调试流已具备 V2 独立编码发布链**
- **兼容采集解码链已进入 decode-only 模式**：仅负责 `V4L2 -> MPP Decode -> DecodedFrame callback`
- **AI Infer Thread 已初步落地**：解码回调只投递最新帧，AI 线程独立执行 `RGA letterbox -> RKNN -> postprocess`，`camera_ai` 使用最近一次检测结果叠加，避免 NPU 推理阻塞主/副码流链路

## 队列建议

建议后续实现如下队列：

- `compressed_packet_queue`
- `normalized_frame_queue`
- `main_branch_queue`
- `sub_branch_queue`
- `ai_input_queue`
- `ai_result_queue`
- `publisher_queue`

每条队列建议：

- 初始容量：`3~4`
- 拥塞策略：先丢最旧帧，优先保活实时性

当前已落地的 AI 队列策略：

- `ai_input_queue` 当前容量为 `2`
- 解码线程只负责 `dup dmabuf fd` 后投递任务
- AI 线程若处理不过来，会主动丢弃最旧待推理帧
- `camera_ai` 叠加最近一次成功检测结果，不阻塞主码流和副码流

## 为什么不在本轮直接全量上线线程化

原因：

1. 原项目当前已在板端完成基线验证
2. 一次性改为全线程化 + 多源 + 多码流 + NPU，风险过大
3. 当前更重要的是先把：
   - 新工程
   - 新目录
   - 新接口
   - 新文档
   稳定下来

## 后续落地顺序建议

### 第 1 步

- Source Thread
- Decode/Normalize Thread
- Dispatcher Thread

### 第 2 步

- Main/Sub 双编码线程

### 第 3 步

- AI Infer Thread
- AI Debug Encode Thread

### 第 4 步

- Publisher Thread Pool
- MediaMTX 输出

## 线程调试建议

当前 V2 已开始输出：

- `main/sub/ai` 分支 `process/encode/push` 平均耗时
- 分支队列长度
- 分支丢帧数
- fallback 状态

后续仍建议继续扩展：

- 当前帧号
- AI 时间戳匹配状态
- 最近错误信息

这样更方便板端联调与性能定位。

## 当前能力边界

本轮虽然已经把兼容基线的 `DecodedFrame` 镜像成 `UnifiedFrame` 并投递到三分支 worker，但这里仍属于“元数据级桥接”：

- 当前分支 worker 只消费帧描述信息，不直接读写底层图像数据
- DMA fd 与虚拟地址目前仅作为后续接线时的上下文透传
- 真实 `main/sub/ai` 发布当前通过同步分支输出层完成，以降低 DMA 生命周期风险
- 真正的零拷贝生命周期管理、RTSP 输入和多源统一接入仍将在后续迭代接入
