# Use Python 3.11 slim image as base (matching the wheel requirement)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RKNN Toolkit Lite2 wheel
COPY rknn-toolkit-lite2-packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl .

# Install the local wheel
RUN pip install --no-cache-dir rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Copy librknnrt.so to /usr/lib/
COPY lib/librknnrt.so /usr/lib/
RUN chmod 755 /usr/lib/librknnrt.so

# Copy the rest of the application code
COPY . .

# Set the default command to run the detection script
CMD ["python", "realtime_detection.py", "--model_path", "model/yolo11n.rknn", "--video_path", "video/test.mp4"]
