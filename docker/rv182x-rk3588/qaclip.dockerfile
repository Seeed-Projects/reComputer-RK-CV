ARG RK182X_BASE_IMAGE=ghcr.io/seeed-projects/recomputer-rk-cv/rv182x-rk3588-runtime:1.1.0
FROM ${RK182X_BASE_IMAGE} AS project-assets
WORKDIR /app
COPY . .
RUN python3 scripts/prepare_assets.py && rm -f model/qaclip-vision.weight.part-* && sha256sum -c SHA256SUMS
FROM ${RK182X_BASE_IMAGE}
WORKDIR /app
COPY --from=project-assets /app/ /app/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)"
CMD ["python3", "web_detection.py", "--video_path", "video/test.mp4"]
