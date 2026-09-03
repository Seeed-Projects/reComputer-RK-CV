# RK3576 CLIP 部署指南

[English](./README.md) | [中文]

本项目是在 RK3576 NPU 上运行的 OpenAI CLIP ViT-B/32 演示，既可做零样本图片分类，也可进行自然语言图片检索。

## 核心功能

- 使用自定义候选标签对单张图片分类
- 使用自然语言对最多 32 张上传图片排序
- Web 图片预览、REST API 和命令行推理
- 224 × 224 预处理与归一化余弦相似度

## 目录结构

- `model/`：图片/文本 RKNN 编码器及 CLIP 词表
- `samples/`：示例图片和提示词
- `task_runtime.py`：分词、图片预处理与相似度排序
- `web_service.py` / `inference.py` / `web/`：API、CLI 与英文界面

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 -e RKNN_LOG_LEVEL=0 \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    ghcr.io/seeed-projects/recomputer-rk-cv/rk3576-clip:latest
```

访问 `http://<开发板IP>:8000`。

```bash
python inference.py --platform rk3576 --model_dir model \
    --file samples/dog_224x224.jpg \
    --prompts 'a photo of a dog|a photo of a cat'

python inference.py --platform rk3576 --model_dir model \
    --files image1.jpg image2.jpg --query 'a red car' --topk 2
```

```bash
curl -X POST http://127.0.0.1:8000/api/models/clip/predict \
    -F 'file=@samples/dog_224x224.jpg' \
    -F 'prompts=a photo of a dog|a photo of a cat'

curl -X POST http://127.0.0.1:8000/api/models/clip/retrieve \
    -F 'files=@image1.jpg' -F 'files=@image2.jpg' -F 'text=a red car'
```

`prompts` 支持 JSON 数组、换行或 `|` 分隔；文本模型固定长度为 20 Token。完整接口见 `/docs`。
