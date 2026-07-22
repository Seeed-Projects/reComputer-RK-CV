# 迭代记录

## Iteration 01

### 目标

- 创建独立于原项目的新工程目录
- 基于 V2 提案拆分工程目录
- 新增类接口与线程模型文档
- 搭建新的运行入口和架构骨架
- 保留原项目已验证的单路可运行链路

### 本轮完成内容

1. 新项目创建完成
   - 路径：`/home/parker/Projects/rk3576_yolov8tortsp_demo`
2. 原项目未被直接修改
3. 新建文档：
   - `docs/project_layout.md`
   - `docs/class_interfaces.md`
   - `docs/thread_model.md`
   - `docs/technical_route_v2.md`
4. 预留 `model/` 目录
5. 新增 V2 应用入口和新配置骨架
6. 新增输入源和 AI 引擎占位接口
7. 新增独立 CMake 和安装规则

### 本轮保留的兼容基线

为了保证新工程在迁移初期仍然可用，本轮保留：

- 原有 `PipelineApp`
- 原有 `V4L2 -> MPP -> RGA -> MPP -> RTSP` 单路链路

当前新入口会优先走新的 V2 应用入口，但在实际运行链路上仍可回落到兼容基线。

### 本轮未完成内容

- RTSP 输入真正接入 FFmpeg
- 统一帧总线实现
- 三码流并行编码
- YOLOv8 RKNN 推理
- AI 检测框绘制
- MediaMTX 发布

### 下一轮建议

1. 完成 `RtspSource`
2. 引入统一帧数据结构
3. 搭建 `FrameDispatcher`
4. 完成主码流和副码流双分支
5. 接入 `YOLOv8 640x640` 推理链

## Iteration 02

### 目标

- 落地统一帧总线公共类型
- 引入 `FrameDispatcher` 与三分支 worker 骨架
- 让新应用入口开始管理主码流、副码流、AI 调试流的线程生命周期
- 同步收敛模型目录命名为 `model/`

### 本轮完成内容

1. 扩展统一运行时数据结构
   - 新增 `UnifiedFrame`
   - 新增 `PixelFormat`
   - 新增 `FrameStorageType`
   - 新增 `BranchStats`
2. 增强线程安全队列
   - 支持阻塞弹出
   - 支持关闭队列
   - 支持统计“挤掉最旧帧”的实时策略
3. 实现 `StreamBranch`
   - 每个码流分支具备独立队列和 worker 线程
   - 预留后续主/副/AI 分支分别挂接编码、AI 绘框、发布逻辑
4. 实现 `FrameDispatcher`
   - 支持向多分支广播统一帧
   - 支持导出分支统计信息
5. 更新 `YoloRtspApplication`
   - 启动时构建主码流、副码流、AI 调试流三个 `StreamProfile`
   - 初始化分发器与分支 worker
   - 保持兼容基线运行不变
6. 目录命名收敛
   - 将模型目录固定为 `model/`
   - 更新 `README`、`CMakeLists.txt`、接口说明文档

### 本轮仍未打通的链路

- `UnifiedFrame` 还未由真实解码输出喂入 `FrameDispatcher`
- `StreamBranch` 当前仅完成线程和队列骨架，尚未接入编码器/发布器
- `RtspSource` 仍为接口占位
- `Yolov8Engine` 仍为模型存在性检查占位

### 下一轮建议

1. 将兼容基线中的解码输出桥接到 `UnifiedFrame`
2. 先打通主码流和副码流的真实分发
3. 为 AI 分支补 `640x640` 预处理入口
4. 再接入 RTSP 输入和 RKNN 推理

## Iteration 03

### 目标

- 将兼容基线中的 `DecodedFrame` 桥接到 V2 的 `UnifiedFrame`
- 在不破坏旧链路可运行性的前提下，把解码输出镜像到 `FrameDispatcher`
- 利用已补全的 `model/` 目录，增加默认模型自动探测
- 同步更新 README 与设计文档

### 本轮完成内容

1. 扩展兼容基线桥接能力
   - 在 `PipelineApp` 中新增解码帧观察回调
   - 旧链路仍保留原有编码和 RTSP 推流路径
2. 打通 `DecodedFrame -> UnifiedFrame`
   - 将宽高、stride、fd、虚拟地址、解码耗时、源描述等信息映射到新总线
   - 统一输出到 `FrameDispatcher`
3. 启用分支镜像日志
   - 主码流、副码流、AI 调试流 worker 可以看到镜像后的统一帧描述
   - 输出基础分支统计，便于后续接入真实处理链
4. 增加默认模型探测
   - 若未传 `--model`，自动按 `yolov8n -> yolov8s -> yolov8m` 顺序检测 `model/`
5. 更新文档
   - `README.md`
   - `docs/class_interfaces.md`
   - `docs/thread_model.md`
   - `docs/technical_route_v2.md`

### 本轮能力边界

- 当前镜像进入 V2 分发器的是“统一帧元数据”，不是完整的零拷贝分支处理链
- 分支 worker 当前只消费帧描述信息，还未对接 RGA/编码/发布
- 模型当前完成的是目录自动发现和加载校验，尚未接入 RKNN 推理

### 下一轮建议

1. 让主码流分支先接入真实 RGA/编码链
2. 复用同一份解码输出，打通副码流 `640x480` 分支
3. 在 AI 分支增加 `640x640` 预处理入口
4. 再把检测结果回灌到 AI 调试流

## Iteration 07

### 目标

- 将 AI 推理从兼容基线解码回调中拆成异步线程
- 让主码流、副码流不再被 AI 推理耗时直接阻塞
- 保留当前已稳定的三路 RTSP 与 AI 叠框能力

### 本轮完成内容

1. 引入 AI 异步任务队列
   - 新增 `ai_task_queue`
   - 队列容量固定为 `2`
   - 拥塞时主动丢弃最旧待推理帧，优先保实时性
2. 引入 AI 独立 worker 线程
   - 兼容基线回调中不再直接执行 `yolov8_engine_.Infer`
   - 改为投递 `dup` 后的 `dmabuf fd` 到 AI 线程
3. 引入最近结果回灌
   - AI 线程推理成功后缓存最近一次 `DetectionFrame`
   - `camera_ai` 输出时直接叠加最近一次有效检测结果
4. 修正 AI 异步推理的输入约束
   - `Yolov8Engine` 允许“仅有 `dmabuf fd`、无 `virt_addr`”的异步输入帧
   - 继续优先走已打通的 `RGA 640x640` 预处理

### 本轮收益

- AI 推理耗时不再直接占用兼容基线回调路径
- 主码流和副码流更容易稳定保持实时输出
- AI 调试流在推理繁忙时允许使用最近结果，整体画面更平滑

### 当前能力边界

- 当前 AI 结果采用“最近一次成功检测结果回灌”，不是严格逐帧时间对齐
- 主码流仍然由兼容基线负责，尚未完全切换到 V2 独立编码链
- `sub/ai` 两路发布仍使用同步桥接输出，后续仍可继续推进完整异步零拷贝化

### 下一轮建议

1. 观察异步化后主码流 `e2e_submit` 是否继续下降
2. 评估 AI 显示 FPS 与实际检测结果刷新频率
3. 如需要，再做 AI 结果时间戳匹配与主码流 V2 化

## Iteration 04

### 目标

- 让 `camera_sub` 和 `camera_ai` 两路先具备真实可拉流能力
- 保持主码流继续走兼容基线，降低当前迭代风险
- 解决 ZLMediaKit 单端口下多路媒体共存问题
- 同步更新使用说明和当前能力边界

### 本轮完成内容

1. 新增分支输出桥接层
   - 新建 `BranchOutput`
   - 为副码流和 AI 调试流各自维护独立 `MppEncoder + RtspServer`
2. 打通真实 `sub/ai` 发布链
   - 在兼容基线解码回调中直接复用 `DecodedFrame`
   - 生成 `camera_sub` 与 `camera_ai` 的 H.264 码流并发布
3. 调整 `RtspServer`
   - 支持同一 RTSP 端口下多个 `mk_media`
   - 避免一个流停止时误杀全部服务
4. 保持旧链路稳定
   - 主码流仍由兼容基线负责
   - `sub/ai` 新链路先采用同步桥接，避免 DMA fd 生命周期在异步队列里失控
5. 完成文档更新
   - README 增加三路地址说明
   - 类接口文档补 `BranchOutput`
   - 线程模型文档补充同步桥接说明

### 本轮能力边界

- 主码流还没有切换到 V2 独立编码链
- AI 调试流当前仍是独立调试镜像流，尚未接入 RKNN 检测框
- 若 `RGA` 无权限，副码流会降级为源分辨率输出
- 已修正副码流在 `RGA fallback` 后下一帧被错误切回 `640x480` 的重建问题；当前进入软件降级后会稳定保持源分辨率输出

### 下一轮建议

1. 把主码流也迁移到 V2 分支输出链
2. 为 AI 分支接入 `640x640` 预处理和 RKNN 推理
3. 把检测结果叠加到 `camera_ai`
4. 再逐步把同步桥接改为真正的异步零拷贝分支

### 补充修正

- 板端使用 `sudo` 后已确认 `/dev/rga` 权限问题解除，`camera_sub` 可以正常以 `640x480` 注册
- 新暴露的问题为 `main/ai` 分支时间水印的 RGBA 图块尺寸是奇数，触发 `RGA watermark blend failed`
- 已把时间水印的坐标和宽高统一修正为偶数对齐，适配 NV12/RGA 的叠加约束
- 继续排查后确认 `main/ai` 的 blend 还存在 `handle/fd` 混用问题，现已把 `src/dst/watermark` 统一切换到 `importbuffer_* + wrapbuffer_handle()` 路径
- 若板端 `RGA alpha blend` 仍失败，则改为保留 `RGA resize/copy`，仅把时间水印改为 CPU 直接叠加到编码输入 NV12 缓冲；这样可以绕开当前驱动路径的不稳定点，同时把 CPU 开销限制在很小的 OSD 区域

## Iteration 05

### 目标

- 让 `Yolov8Engine` 从“模型文件存在性检查”升级为真实 `RKNN` 推理引擎
- 将 `RKNN runtime` 纳入新工程，随 `install/` 一起分发
- 打通 `640x640` 预处理和 `rknn_run`，为下一轮后处理和画框做准备

### 本轮完成内容

1. 引入 `RKNN runtime`
   - 新增 `third_party/rknn/include`
   - 新增 `third_party/rknn/lib/aarch64/librknnrt.so`
   - 更新 `CMakeLists.txt` 参与链接和安装
2. 升级 `Yolov8Engine`
   - 支持 `rknn_init`
   - 支持 `rknn_query` 输入输出属性
   - 支持真实 `rknn_run`
   - 支持 `rknn_outputs_get/release`
3. 打通 AI 输入预处理
   - 从解码后的 NV12/YUV420SP/YUV422SP 帧做 `640x640` letterbox
   - 当前采用 CPU 预处理生成 RGB 输入张量
4. 接入应用主控
   - 在兼容基线解码回调中执行 AI 推理
   - 输出 `preprocess` 与 `npu` 耗时日志

### 本轮能力边界

- 当前已经是真实 `RKNN` 推理，而非占位
- 当前 `DetectionFrame.boxes` 仍为空，说明检测框后处理与 `camera_ai` 画框叠加尚未接入
- 当前 AI 预处理先走 CPU `letterbox`，后续可再替换为 `RGA 640x640` 预处理链

### 下一轮建议

1. 引入 YOLOv8 后处理与 NMS
2. 把检测结果映射为 `DetectionBox`
3. 将检测框和 FPS 叠加到 `camera_ai`
4. 视性能情况再把 `640x640` 预处理改为 `RGA`

## Iteration 06

### 目标

- 把 `RKNN` 原始输出真正变成可用检测框
- 将 AI 推理结果叠加到 `camera_ai` 调试流

### 本轮完成内容

1. 补齐 YOLOv8 后处理
   - 基于 `output_attrs` 解析 3 个检测分支
   - 支持量化输出反量化
   - 支持 DFL 解码
   - 支持按类别 NMS
   - 生成 `DetectionBox`
2. 扩展 AI 日志
   - 当前日志会输出 `boxes` 数量
   - 若存在检测框，会额外打印首个目标的类别、坐标和分数
3. 扩展 `camera_ai` 叠加
   - 按用户要求使用 2 像素框线
   - 颜色按红、橙、黄、绿、青、蓝、紫循环
   - 增加 FPS 文本叠加
   - 在框左上角增加类别名标签
4. 保持稳定性优先
   - 框线与 FPS 仍走 CPU 小区域叠加
   - 继续避免板端 `RGA alpha blend` 不稳定风险

### 当前能力边界

- 已经具备 `检测框 -> camera_ai` 的完整软件链路
- 当前 AI 调试流仍是“兼容基线解码 + AI 结果回灌”的结构
- 当前 `640x640` 预处理仍走 CPU，若后续要降低 AI 分支时延，可继续替换为 `RGA letterbox`
- 当前显示的 FPS 为估算值，计算口径为 `1 / (decode + ai_preprocess + npu + ai_process + ai_encode)`，不是最终流实际输出 FPS 统计

### 下一轮建议

1. 在板端验证真实检测框和 FPS 叠加效果
2. 若检测框正常，继续优化 `640x640` 预处理为 `RGA`
3. 再考虑把主码流也迁移到 V2 独立编码链

## Iteration 07

### 目标

- 将 AI `640x640` 预处理从 CPU `letterbox` 替换为 `RGA`
- 尽量压低 `preprocess_us`，提升 AI 分支估算 FPS

### 本轮完成内容

1. 在 `Yolov8Engine` 中接入 `RGA` 预处理
   - 优先使用 `DMABUF -> RGB888 virtual buffer` 的 `RGA letterbox`
   - 背景仍用 `114` 填充，避免引入额外不稳定的 `RGA fill` 色值兼容问题
2. 保留 CPU 自动兜底
   - 若 `RGA` 预处理失败，会自动回退到旧的 CPU 预处理
   - 回退时会打印明确日志，便于板端判断当前使用的实际路径
3. 修正 `RGA` 预处理返回码判断
   - `imcheck()` 成功返回 `IM_STATUS_NOERROR`
   - `improcess()` 成功返回 `IM_STATUS_SUCCESS`
   - 之前将两者混为一谈，导致日志出现“`No errors during operation` 但仍回退”的假失败

### 当前能力边界

- 当前 AI 预处理已经优先走 `RGA`
- 当前框线/类别名/FPS 文本叠加仍保持 CPU 小区域绘制
- 若后续需要继续压缩 CPU 占用，可再评估把 AI overlay 也迁移到更稳定的硬件路径

## Iteration 08

### 目标

- 将 AI 推理从解码回调中拆出，避免 NPU 推理阻塞主/副码流
- 使用“最新帧推理 + 最近结果回灌”策略保持实时性

### 本轮完成内容

1. 新增 AI 异步线程
   - 解码回调只负责复制/投递 `dmabuf fd`
   - AI 线程独立执行 `RGA letterbox -> RKNN -> postprocess`
2. 新增 AI 队列
   - 队列容量为 `2`
   - 拥塞时丢弃最旧待推理帧
3. 新增最近检测结果缓存
   - `camera_ai` 使用最近一次有效检测结果叠加
   - 主码流和副码流不再等待 NPU 推理完成
4. 板端验证结果
   - `preprocess_us` 稳定下降到约 `5ms ~ 8ms`
   - 主链路 `e2e_submit_us` 明显下降

### 当前能力边界

- AI 检测结果当前采用最近结果回灌，暂未做严格逐帧时间戳匹配
- 若后续对框位置时序一致性要求更高，可继续实现检测结果时间戳匹配

## Iteration 09

### 目标

- 完成主码流 V2 化
- 让主码流、副码流、AI 调试流全部由 V2 分支输出层统一负责
- 将旧 `PipelineApp` 收敛为稳定采集解码链

### 本轮完成内容

1. 增加 `PipelineApp` decode-only 模式
   - 保留 `V4L2 -> MPP Decode`
   - 停止旧主码流的 `RGA -> MPP Encode -> RTSP` 路径
2. 主码流接入 V2 `BranchOutput`
   - `live/camera` 现在由 V2 主分支编码和发布
   - 避免旧主码流和 V2 主码流重复注册同一流名
3. 三路输出统一
   - `live/camera`
   - `live/camera_sub`
   - `live/camera_ai`
   均由 V2 分支输出层负责
4. 文档同步
   - README 已更新当前运行方式
   - 类接口、线程模型、技术路线文档已同步当前职责边界

### 当前能力边界

- 本地 V4L2/MJPG 输入仍复用原稳定采集解码实现
- DMA buffer 生命周期仍通过同步分支输出控制，尚未做完整零拷贝帧池
- 下一阶段建议转向 RTSP 输入源或更细化的 V2 分支性能统计
