#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

namespace rk3576_yolo_demo {

enum class SourceKind {
  kV4L2,
  kRtsp,
  kLocalVideo,
};

enum class ResizeMode {
  kStretch,
  kLetterbox,
  kCenterCrop,
};

enum class StreamRole {
  kMain,
  kSub,
  kAiDebug,
};

enum class PixelFormat {
  kUnknown,
  kNv12,
  kRgb888,
  kRgba8888,
  kMjpeg,
  kH264,
  kH265,
};

enum class FrameStorageType {
  kUnknown,
  kDmabuf,
  kVirtualMemory,
};

struct SourceDescriptor {
  SourceKind kind = SourceKind::kV4L2;
  std::string name;
  std::string location;
  bool compressed_input = true;
  bool raw_input = false;
};

struct StreamProfile {
  StreamRole role = StreamRole::kMain;
  std::string app_name;
  std::string stream_name;
  int width = 0;
  int height = 0;
  int bitrate = 0;
  bool enable_osd = false;
  bool enable_ai_overlay = false;
  ResizeMode resize_mode = ResizeMode::kStretch;
};

struct FrameTransformInfo {
  ResizeMode mode = ResizeMode::kStretch;
  int src_x = 0;
  int src_y = 0;
  int src_width = 0;
  int src_height = 0;
  int dst_x = 0;
  int dst_y = 0;
  int dst_width = 0;
  int dst_height = 0;
  int output_width = 0;
  int output_height = 0;
  bool valid = false;
};

struct DetectionBox {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  int class_id = -1;
  float score = 0.0f;
  std::string class_name;
};

struct DetectionFrame {
  std::uint64_t pts_ms = 0;
  int source_width = 0;
  int source_height = 0;
  int input_width = 0;
  int input_height = 0;
  std::uint64_t preprocess_us = 0;
  std::uint64_t npu_us = 0;
  std::vector<DetectionBox> boxes;
};

struct UnifiedFrame {
  std::uint64_t frame_id = 0;
  std::uint64_t pts_ms = 0;
  int width = 0;
  int height = 0;
  int hor_stride = 0;
  int ver_stride = 0;
  PixelFormat format = PixelFormat::kUnknown;
  int native_format = 0;
  FrameStorageType storage = FrameStorageType::kUnknown;
  bool compressed = false;
  bool key_frame = false;
  int dma_fd = -1;
  const void* data = nullptr;
  std::size_t bytes = 0;
  std::uint64_t decode_us = 0;
  std::uint64_t preprocess_us = 0;
  std::uint64_t npu_us = 0;
  std::uint64_t encode_us = 0;
  SourceDescriptor source;
};

struct BranchStats {
  StreamRole role = StreamRole::kMain;
  std::uint64_t submitted = 0;
  std::uint64_t dropped = 0;
  std::uint64_t processed = 0;
  std::size_t queue_size = 0;
};

inline const char* ToString(SourceKind kind) {
  switch (kind) {
    case SourceKind::kV4L2:
      return "v4l2";
    case SourceKind::kRtsp:
      return "rtsp";
    case SourceKind::kLocalVideo:
      return "localvideo";
  }
  return "unknown";
}

inline const char* ToString(ResizeMode mode) {
  switch (mode) {
    case ResizeMode::kStretch:
      return "stretch";
    case ResizeMode::kLetterbox:
      return "letterbox";
    case ResizeMode::kCenterCrop:
      return "center_crop";
  }
  return "unknown";
}

inline const char* ToString(StreamRole role) {
  switch (role) {
    case StreamRole::kMain:
      return "main";
    case StreamRole::kSub:
      return "sub";
    case StreamRole::kAiDebug:
      return "ai";
  }
  return "unknown";
}

inline const char* ToString(PixelFormat format) {
  switch (format) {
    case PixelFormat::kUnknown:
      return "unknown";
    case PixelFormat::kNv12:
      return "nv12";
    case PixelFormat::kRgb888:
      return "rgb888";
    case PixelFormat::kRgba8888:
      return "rgba8888";
    case PixelFormat::kMjpeg:
      return "mjpeg";
    case PixelFormat::kH264:
      return "h264";
    case PixelFormat::kH265:
      return "h265";
  }
  return "unknown";
}

inline const char* ToString(FrameStorageType storage) {
  switch (storage) {
    case FrameStorageType::kUnknown:
      return "unknown";
    case FrameStorageType::kDmabuf:
      return "dmabuf";
    case FrameStorageType::kVirtualMemory:
      return "virtual";
  }
  return "unknown";
}

inline int AlignEvenDown(int value) {
  return value & ~1;
}

inline int AlignEvenSize(int value) {
  if (value <= 1) {
    return value;
  }
  return (value % 2 == 0) ? value : (value - 1);
}

inline FrameTransformInfo ComputeFrameTransform(ResizeMode mode, int src_width, int src_height,
                                                int output_width, int output_height) {
  FrameTransformInfo info;
  info.mode = mode;
  info.output_width = output_width;
  info.output_height = output_height;
  if (src_width <= 0 || src_height <= 0 || output_width <= 0 || output_height <= 0) {
    return info;
  }

  info.valid = true;
  if (mode == ResizeMode::kStretch) {
    info.src_width = src_width;
    info.src_height = src_height;
    info.dst_width = output_width;
    info.dst_height = output_height;
    return info;
  }

  if (mode == ResizeMode::kLetterbox) {
    const double scale = std::min(static_cast<double>(output_width) / static_cast<double>(src_width),
                                  static_cast<double>(output_height) / static_cast<double>(src_height));
    int scaled_width = std::max(1, static_cast<int>(std::round(src_width * scale)));
    int scaled_height = std::max(1, static_cast<int>(std::round(src_height * scale)));
    scaled_width = AlignEvenSize(scaled_width);
    scaled_height = AlignEvenSize(scaled_height);
    info.src_width = src_width;
    info.src_height = src_height;
    info.dst_width = std::min(output_width, scaled_width);
    info.dst_height = std::min(output_height, scaled_height);
    info.dst_x = AlignEvenDown((output_width - info.dst_width) / 2);
    info.dst_y = AlignEvenDown((output_height - info.dst_height) / 2);
    return info;
  }

  const double scale = std::max(static_cast<double>(output_width) / static_cast<double>(src_width),
                                static_cast<double>(output_height) / static_cast<double>(src_height));
  int crop_width = std::max(1, static_cast<int>(std::round(output_width / scale)));
  int crop_height = std::max(1, static_cast<int>(std::round(output_height / scale)));
  crop_width = std::min(src_width, AlignEvenSize(crop_width));
  crop_height = std::min(src_height, AlignEvenSize(crop_height));
  info.src_width = crop_width;
  info.src_height = crop_height;
  info.src_x = AlignEvenDown((src_width - crop_width) / 2);
  info.src_y = AlignEvenDown((src_height - crop_height) / 2);
  info.dst_width = output_width;
  info.dst_height = output_height;
  return info;
}

}  // namespace rk3576_yolo_demo
