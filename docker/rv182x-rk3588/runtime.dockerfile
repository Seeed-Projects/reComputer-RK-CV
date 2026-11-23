FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 RKNN_LOG_LEVEL=0 \
    LD_LIBRARY_PATH=/opt/rk182x-runtime/lib:/usr/lib
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG APT_MIRROR=http://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
       libglib2.0-0 libgomp1 ffmpeg kmod make gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/rk182x-runtime/requirements.txt
COPY rknn3-toolkit-lite-packages/ /wheels/
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
      -r /opt/rk182x-runtime/requirements.txt /wheels/*.whl \
    && rm -rf /wheels
COPY . /opt/rk182x-runtime/
# Lite 1.1.0 discovers libraries by standard filesystem paths before dlopen.
RUN ln -s /opt/rk182x-runtime/lib/librknn3_api.so /usr/lib/librknn3_api.so \
    && ln -s /opt/rk182x-runtime/lib/librknn3_api_rkcp.so /usr/lib/librknn3_api_rkcp.so \
    && python3 -c "import sys; sys.path.insert(0,'/opt/rk182x-runtime'); from runtime import verify_assets; verify_assets()"
LABEL org.opencontainers.image.title="RK3588 + RK1820 shared runtime" \
      io.seeed.rk182x.sdk="1.1.0" io.seeed.rk182x.bootstrap="1" \
      io.seeed.rk182x.firmware-board="ext-m2"
ENTRYPOINT ["python3", "/opt/rk182x-runtime/runtime.py"]
CMD ["doctor"]
