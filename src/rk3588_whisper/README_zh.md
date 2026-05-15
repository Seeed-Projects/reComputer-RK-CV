# RK3588 Whisper 语音识别与 Web 接口项目

该项目基于 RKNN-Toolkit-Lite2，在瑞芯微 RK3588 平台上实现 OpenAI Whisper 语音识别模型的高性能部署。项目采用纯 Python 架构，集成了 FastAPI 提供 Web API 和简单易用的网页端 UI 界面。

## 目录结构

- `model/`: 存放 Whisper 模型的 RKNN 转换文件 (例如：`whisper_encoder_base_20s.rknn` 和 `whisper_decoder_base_20s.rknn`)
- `audio/`: 存放用于测试的音频文件 (`test_en.wav`, `test_zh.wav`)
- `lib/`: 存放 NPU 运行时的依赖库 (`librknnrt.so`)
- `rknn-toolkit-lite2-packages/`: 存放 RKNN-Toolkit-Lite2 的 Python 安装包
- `py_utils/`: 包含 Log-Mel 频谱计算和 Tokenizer 文本解码工具
- `web_service.py`: 核心主程序，提供 Web UI 及同步/异步 RESTful API 服务
- `requirements.txt`: Python 依赖列表

## 环境准备

### 1. 硬件要求
- 瑞芯微 RK3588 开发板 (例如: reComputer RK-CV 系列)

### 2. 系统要求
- 使用 armbian

### 3. 安装依赖

```bash
# 1. 更新系统并安装系统依赖
sudo apt update
sudo apt install -y python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 ffmpeg

# 2. 安装 Python 基础依赖
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 安装 RKNN-Toolkit-Lite2 (根据 Python 版本选择)
# 以 Python 3.11 为例：
pip3 install rknn-toolkit-lite2-packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

## 运行指南

本项目默认以 Web 服务模式启动，专为远程访问与 API 调用设计。

> **注意：** 在运行之前，请确保已将量化好的 `.rknn` 模型放入 `model/` 目录下。

### 启动 Web 服务 (web_service.py)

```bash
# 启动服务
python3 web_service.py
```

## Web 访问与 API 说明

服务启动后，默认运行在 `0.0.0.0:8000`。

### 1. Web 界面预览
在浏览器中访问：`http://<开发板IP>:8000`
页面提供基础的模型热切换（语言、大小）以及音频上传识别功能。

### 2. RESTful API 接口

#### 获取与更新配置
- **获取当前状态**: `GET /api/system/status`
- **热更新模型与语言**: `POST /api/system/config`
  - 参数：`model_size` (如 base), `language` (如 en, zh)

#### 同步短音频推理 (适合 20s 内音频)
- **推理预测**: `POST /api/models/whisper/predict`
  - 支持上传音频 (`file`)
  - 支持指定目标语言 (`language`)
  - 接口将直接阻塞并返回完整的转录文本 (`text`)

```json
{
  "status": "success",
  "data": {
    "text": "你好，世界",
    "language": "zh",
    "duration": 5.2,
    "inference_time": 1.1
  }
}
```

#### 异步长音频推理 (适合 20s 以上长音频/视频)
- **创建任务**: `POST /api/models/whisper/task`
  - 同样接收 `file` 和 `language`
  - 立即返回一个 `task_id`，不阻塞连接
- **查询任务进度**: `GET /api/models/whisper/task/{task_id}`
  - 返回任务状态 (pending/processing/completed/failed) 和当前的进度或最终文本

## 性能说明

- 项目利用了 RK3588 的多个 NPU 核心 (6 TOPS 算力)。默认逻辑中将 Encoder 与 Decoder 分配至不同的核心以优化负载。
- 采用了滑动窗口策略与 20 秒特征切割，确保流式和长音频转录过程中的稳定性。
