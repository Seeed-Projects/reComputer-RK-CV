# Use Python 3.11 slim image as base (matching the wheel requirement)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
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

# Expose port for Web API
EXPOSE 8000

# Set the default command to run the web service with the base model
CMD ["python", "web_service.py", "--default_model", "base"]