# Python 模型项目迁移规范

每个模型按“平台 + 模型”建立独立部署单元，目录名采用
`rk<芯片>_<模型>`，例如 `rk3588_mobilenet`。不同芯片的 RKNN 模型和
`librknnrt.so` 不得放在同一个部署目录中。

## 必需结构

```text
src/rk<平台>_<模型>/
├── web_classification.py       # FastAPI、Web UI、推理入口
├── requirements.txt
├── README.md
├── README_zh.md
├── .dockerignore
├── model/*.rknn
├── py_utils/
├── video/*.mp4
├── lib/librknnrt.so
└── rknn-toolkit-lite2-packages/*.whl

docker/rk<平台>/<模型>.dockerfile
```

模型转换使用的 ONNX/PyTorch 文件不进入运行镜像；如果需要保留，应放入
独立的转换工程或由 `.dockerignore` 排除。

每个项目的 README 必须记录原始模型路径、转换脚本路径、目标平台参数和
输出模型文件名，确保部署模型可追溯和可重建。

## 必需接口

- `GET /`：浏览器预览或文件分析页面
- `GET /api/health`：服务、平台和模型状态
- `GET|POST /api/config`：运行时阈值配置
- `POST /api/models/<模型>/predict`：标准推理入口
- `GET /api/video_feed`：MJPEG 预览
- `POST /api/video/upload`
- `POST /api/video/analyze`
- `GET /api/video/status`

FastAPI 自动提供 `/docs` 和 `/openapi.json`。推理接口至少支持图片；适用时
支持视频、`timestamp`、`conf` 和 `topk`。

## Docker 与 CI

- 基础运行环境与 RKNN Lite wheel 的 Python ABI 必须一致。
- 默认命令应能直接运行内置样例，容器监听 `8000` 端口。
- Dockerfile 必须提供 `/api/health` 健康检查。
- 新项目必须加入 `.github/workflows/docker-build.yml` 的 paths 和 matrix。
- `.rknn`、`.mp4`、`.whl`、`.so` 使用仓库现有 Git LFS 属性。

## 验证要求

提交前至少执行 Python 语法检查、Dockerfile 路径检查和 Git LFS 状态检查。
最终仍需在对应开发板上完成 Docker 构建、NPU 推理和 API 请求验证。
