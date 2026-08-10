FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    RKNN_LOG_LEVEL=0 \
    RK_CV_WORKSPACE=/app/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rknn-toolkit-lite2-packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl .
RUN pip install --no-cache-dir \
    rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

COPY lib/librknnrt.so /usr/lib/librknnrt.so
RUN chmod 755 /usr/lib/librknnrt.so && ldconfig

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python3", "web_detection.py", "--platform", "rk3576", "--model_path", "/app/model/yolo_world_v2s_i8.rknn", "--text_model", "/app/model/clip_text_fp16.rknn", "--text_features", "/app/model/coco_text_outp.npy", "--vocab_path", "/app/model/clip_vocab.txt", "--class_path", "/app/model/detect_classes.txt", "--video_path", "/app/video/test.mp4", "--host", "0.0.0.0", "--port", "8000"]
