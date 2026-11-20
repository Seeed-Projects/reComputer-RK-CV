# QA-CLIP ViT-L/14 · W8A8 转换 / conversion

本目录供 x86 Ubuntu 开发者复现转换。ARM64 Docker 运行镜像只需要模型、权重与 RKNN3 Lite，不需要 PyTorch 或 x86 Toolkit。

1. 安装用户提供的 RK1820/RK1828 AI SDK 1.1.0 中的 RKNN3 Toolkit x86 wheel，使用 Python 3.10 环境。
2. 本次导出使用 CPU；YOLO 导出环境为 Ultralytics 8.4.69、ONNX 1.18.0、NumPy 2.2.6。导出可在独立虚拟环境中进行。
3. 以下命令从本项目根目录执行。固定 target=rk1820、core_num=1、quantized_dtype=w8a8、do_quantization=True、normal/channel，无手工 W16A16 子图。

```bash
python conversion/make_calibration.py
```

这会生成指向当前目录的绝对校准路径；移动目录后重新生成。普通视觉模型使用官方 COCO 20 张子集；RAFT 使用官方 20 对图像。演示校准和功能检查不等于完整数据集精度评估。

先把 TencentARC/QA-CLIP-ViT-L-14 的 config.json、pytorch_model.bin、preprocessor_config.json、vocab.txt 下载到 source/QA-CLIP-ViT-L-14。使用 transformers==4.53.3，外置词嵌入保留 FP32，两个 RKNN 编码器均为 W8A8。

```bash
python conversion/export_qaclip.py --model-dir source/QA-CLIP-ViT-L-14 --output-dir conversion/exports
python conversion/compile_partition.py --stage qaclip-text --onnx conversion/exports/qaclip-text.onnx --dataset conversion/exports/text-calibration.txt --output model/qaclip-text.rknn
python conversion/compile_partition.py --stage qaclip-vision --onnx conversion/exports/qaclip-vision.onnx --dataset conversion/calibration.txt --output model/qaclip-vision.rknn
cp conversion/exports/word_embeddings.npy model/
cp -r conversion/exports/tokenizer model/
```

固定文本长度 128 tokens。运行端先生成候选文本向量并释放文本上下文，再加载视觉编码器；修改候选文本时重复此过程。FPS 使用缓存文本后的逐帧视觉处理，文本初始化耗时单独记录。两编码器均编译为单核，实机 core mask 为 0x1。


转换将生成同名 `.rknn` 与 `.weight`，必须成对使用。`conversion/*.json` 记录本次已验证产物的参数、原始 ONNX SHA256。运行库、工具和固件均匹配 1.1.0，未混用上游仓库的 1.2 runtime。

[官方示例源码](https://github.com/airockchip/rknn3-model-zoo/tree/7f8c7d848448fa14cf16ba359ca98405c8bb94b5/examples/QAClip)

## Large vision weight

The 297 MiB vision weight is stored in four regular Git chunks below 100 MiB each. Docker automatically reconstructs and SHA256-verifies the original file, then removes the chunks from the runtime image. This avoids dependence on Git LFS connectivity or quota and does not change the tested W8A8 model. Before direct source execution or full checksum verification, run from the project directory:

```bash
python3 scripts/prepare_assets.py
sha256sum -c SHA256SUMS
```

Chunk sizes and hashes are recorded in `model/qaclip-vision.parts.json`.
