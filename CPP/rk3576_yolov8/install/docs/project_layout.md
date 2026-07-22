# 工程目录拆分说明

## 目标

本文件说明 `rk3576_yolov8tortsp_demo` 为什么采用当前目录布局，以及哪些目录来自原项目复用，哪些目录是 V2 迭代新增。

## 拆分原则

本次工程拆分遵循以下原则：

1. 不直接修改原项目
2. 先保留可运行基线，再扩展 V2 能力
3. 文档、模型、配置、依赖都进入新工程目录
4. 兼顾“当前能跑”和“后续好扩展”

## 目录说明

### 顶层目录

- `CMakeLists.txt`
  - 新工程独立构建入口
- `README.md`
  - 新工程总说明与运行入口
- `cmake/`
  - 交叉编译工具链配置
- `configs/`
  - 预留运行配置与推流配置文件
- `docs/`
  - 所有迭代说明文档
- `licenses/`
  - 第三方依赖许可证
- `model/`
  - YOLOv8 RKNN 模型目录
- `scripts/`
  - 构建脚本
- `third_party/`
  - 随工程复制的依赖头文件和 aarch64 库

### include

- `include/rk3576_demo/`
  - 原项目的稳定基线接口
  - 当前主要承担：
    - V4L2Camera
    - MppDecoder
    - MppEncoder
    - RgaProcessor
    - RtspServer
    - WatermarkRenderer
    - SystemMonitor
    - PipelineApp
- `include/rk3576_yolo_demo/`
  - V2 新架构接口
  - 当前规划包括：
    - `app/`
    - `ai/`
    - `branch/`
    - `common/`
    - `pipeline/`
    - `publisher/`
    - `source/`

### src

- `src/camera|codec|rga|rtsp|utils|watermark`
  - 复用原项目可运行实现
- `src/app/pipeline_app.cpp`
  - 原项目总编排器，作为兼容基线继续保留
- `src/app/`
  - V2 新应用入口和架构控制器
- `src/source/`
  - 本地源与 RTSP 源适配层
- `src/ai/`
  - YOLOv8 预处理、推理、后处理
- `src/branch/`
  - 主码流、副码流、AI 调试码流分支
- `src/pipeline/`
  - 统一帧分发、元数据绑定、缓冲池管理
- `src/publisher/`
  - FFmpeg/MediaMTX 发布层
- `src/common/`
  - 公共工具、线程安全队列、统一类型定义

## 复用策略

### 当前直接复用的模块

- `V4L2Camera`
- `MppDecoder`
- `MppEncoder`
- `RgaProcessor`
- `RtspServer`
- `WatermarkRenderer`
- `SystemMonitor`
- `PipelineApp`

### 当前新增的模块

- 新配置入口
- 新应用入口
- 新输入源接口
- RTSP 源占位实现
- YOLOv8 引擎占位实现
- V2 架构说明文档

## 为什么不直接删除原模块

原因如下：

1. 原模块已经过板端验证
2. 新架构一次性全量替换风险太高
3. 保留基线可以持续验证：
   - 构建系统是否正常
   - 依赖路径是否正常
   - 安装规则是否正常
   - 板端运行是否正常

因此当前工程采用：

- **旧链路可运行**
- **新链路可扩展**

## 后续目录演进建议

后续在 V2 功能逐步落地后，建议按下面顺序收敛：

1. `source/` 完成 `V4L2Source` 和 `RtspSource`
2. `pipeline/` 完成统一帧总线
3. `branch/` 完成主/副/AI 分支
4. `publisher/` 完成 FFmpeg/MediaMTX 发布
5. `ai/` 完成 YOLOv8 全链路
6. 当 V2 链路稳定后，再考虑逐步淡化旧 `PipelineApp` 的入口地位
