# Use Python 3.11 slim image as base (matching the wheel requirement)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and others
# libgl1: replacement for libgl1-mesa-glx in Debian 12 (Bookworm)
# libglib2.0-0: for cv2
# libgomp1: for OpenMP support if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RKNN Toolkit Lite2 wheel
COPY rknn-toolkit-lite2-packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl .

# Install the local wheel
RUN pip install --no-cache-dir rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Copy the rest of the application code
COPY . .

# Set the default command to run the detection script
# Users can override this command
# Using a default model path (assuming one exists or user mounts it)
# Since we don't know which model the user wants, we can just set python as entrypoint or give a help message
# But the user asked for "one line docker command", so a default CMD is good.
# Let's assume yolo11n.rknn is a good default if it exists.
CMD ["python", "realtime_detection.py", "--model_path", "model/yolo11n.rknn", "--video_path", "video/test.mp4"]
