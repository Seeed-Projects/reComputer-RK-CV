FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LD_LIBRARY_PATH=/usr/lib RKNN_LOG_LEVEL=0
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG APT_MIRROR=http://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources && apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
COPY rknn3-toolkit-lite-packages/ /wheels/
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements.txt /wheels/*.whl \
    && rm -rf /wheels
COPY lib/ /usr/lib/
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)"
CMD ["python", "web_detection.py"]
