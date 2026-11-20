FROM ubuntu:26.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR=http://ports.ubuntu.com/ubuntu-ports
RUN sed -i "s|http://ports.ubuntu.com/ubuntu-ports/|${UBUNTU_MIRROR}/|g" /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git curl ca-certificates \
    python3-dev python3-yaml pybind11-dev nlohmann-json3-dev libdrm-dev \
    libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev \
    && rm -rf /var/lib/apt/lists/*
COPY tools/setup-sdk.sh /tmp/setup-sdk.sh
RUN bash /tmp/setup-sdk.sh
WORKDIR /source
COPY CMakeLists.txt ./
COPY native ./native
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/rkapi \
    && cmake --build /build --parallel 4 && cmake --install /build

FROM ubuntu:26.04 AS runtime
ARG UBUNTU_MIRROR=http://ports.ubuntu.com/ubuntu-ports
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OPENCV_IO_MAX_IMAGE_PIXELS=67108864 \
    LD_LIBRARY_PATH=/opt/rockchip/rga/lib:/opt/rockchip/rknn/lib:/opt/rockchip/mpp/lib \
    PATH=/opt/venv/bin:$PATH
RUN sed -i "s|http://ports.ubuntu.com/ubuntu-ports/|${UBUNTU_MIRROR}/|g" /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3-venv libavdevice62 libavformat62 libavcodec62 libavutil60 libdrm2 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --only-binary=:all: --index-url "$PIP_INDEX_URL" -r /tmp/requirements.txt
COPY --from=build /opt/rockchip/rga/lib/librga.so /opt/rockchip/rga/lib/
COPY --from=build /opt/rockchip/rknn/lib/librknnrt.so /opt/rockchip/rknn/lib/
COPY --from=build /opt/rockchip/mpp/lib/librockchip_mpp.so.0 /opt/rockchip/mpp/lib/
RUN ln -s librockchip_mpp.so.0 /opt/rockchip/mpp/lib/librockchip_mpp.so.1 \
    && ln -s librockchip_mpp.so.1 /opt/rockchip/mpp/lib/librockchip_mpp.so
COPY --from=build /opt/rkapi /opt/rkapi
COPY rkapi /opt/rkapi/rkapi
WORKDIR /opt/rkapi
ENV RK_PACKAGE=/opt/model/package.json
LABEL org.opencontainers.image.title="Rockchip CV API" \
    org.opencontainers.image.description="FastAPI service backed by the Rockchip C++ RGA/DMA/RKNN engine"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "rkapi.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16"]

# The catalog selects weights and adapter; no Dockerfile edit per new model.
FROM build AS package
COPY rkapi /source/rkapi
COPY tools/models.py /source/tools/models.py
COPY models /source/models
ARG MODEL_ID
RUN python3 tools/models.py prepare --model "$MODEL_ID" --output /model-package

FROM runtime AS model
COPY --from=package /model-package/ /opt/model/
