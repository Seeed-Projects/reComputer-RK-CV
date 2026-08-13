FROM ghcr.io/seeed-projects/recomputer-rk-cv/rk3588-mobilesam:latest

COPY web_service.py task_runtime.py /app/
COPY README.md README_zh.md /app/
