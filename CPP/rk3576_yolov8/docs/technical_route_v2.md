# V2 技术路线摘要

## 总体路线

推荐的 V2 路线为：

```text
Local V4L2 Source / RTSP Source
        |
        v
Source Probe + Format Detect
        |
        +-- compressed -> MPP Decode
        |
        +-- raw frame   -> RGA Normalize
        |
        v
Unified Frame Bus (NV12 + FD)
        |
        +--> Main Stream Branch
        +--> Sub Stream Branch
        +--> AI Branch
```

## 三条分支

### 主码流

- 默认与输入源同分辨率
- 由 `RGA + MPP Encode + Publisher` 完成

### 副码流

- 默认 `640x480`
- 默认 `letterbox`
- 由 `RGA + MPP Encode + Publisher` 完成

### AI 调试流

- 显示原始分辨率
- 叠加检测框与时延/FPS 信息
- YOLOv8 推理输入固定 `640x640`

## YOLOv8 路线

```text
UnifiedFrame(NV12)
  -> RGA letterbox to 640x640
  -> RKNN inference
  -> postprocess
  -> map boxes back to source frame
  -> RGA draw boxes
  -> MPP encode
  -> Publisher
```

## 当前第一轮状态

本文件对应的是目标架构摘要。

当前代码中已经完成：

- 新工程目录
- 新入口
- 新接口骨架
- 兼容基线保留
- `UnifiedFrame` 统一帧数据结构
- `FrameDispatcher` 与三分支 worker 骨架
- 兼容基线解码输出到 V2 总线的元数据镜像
- `main/sub/ai` 三路独立编码与 RTSP 发布链路
- 兼容采集解码链 decode-only 化，只负责 `V4L2 -> MPP Decode`
- `RKNN runtime` 并入工程
- `AI 640x640` RGA 预处理与真实 `rknn_run` 链路
- `YOLOv8` 后处理、NMS 与检测框输出
- `camera_ai` 彩色框线与 FPS 文本叠加
- AI 推理异步线程化，主/副码流不再等待 NPU 推理完成

当前限制：

- 本地 V4L2/MJPG 输入仍复用稳定的兼容采集解码链
- RTSP 输入、MIPI/HDMI 多源输入仍待后续接入

后续重点将转向多输入源和发布层扩展。
