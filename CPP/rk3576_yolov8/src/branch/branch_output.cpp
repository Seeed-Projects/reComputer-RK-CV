#include "rk3576_yolo_demo/branch/branch_output.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <chrono>

#include "rk3576_yolo_demo/common/logger.hpp"
#include "rockchip/mpp_frame.h"

namespace rk3576_yolo_demo {

namespace {

std::uint64_t SystemNowMs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

bool CopySemiPlanarRowToNv12(const std::uint8_t* src_row, int width, bool swap_uv, std::uint8_t* dst_row) {
  if (src_row == nullptr || dst_row == nullptr || width <= 0) {
    return false;
  }

  if (!swap_uv) {
    std::memcpy(dst_row, src_row, static_cast<std::size_t>(width));
    return true;
  }

  for (int x = 0; x + 1 < width; x += 2) {
    dst_row[x] = src_row[x + 1];
    dst_row[x + 1] = src_row[x];
  }
  return true;
}

int ClampToByte(int value) {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return value;
}

void RgbToYuv(std::uint8_t r, std::uint8_t g, std::uint8_t b, int* y, int* u, int* v) {
  *y = ClampToByte(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
  *u = ClampToByte(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
  *v = ClampToByte(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
}

int BlendChannel(int src, int dst, int alpha) {
  return ClampToByte((src * alpha + dst * (255 - alpha) + 127) / 255);
}

const std::uint8_t kOverlayColors[][3] = {
    {255, 0, 0},    // red
    {255, 165, 0},  // orange
    {255, 255, 0},  // yellow
    {0, 255, 0},    // green
    {0, 255, 255},  // cyan
    {0, 0, 255},    // blue
    {128, 0, 255},  // purple
};

struct YuvColor {
  std::uint8_t y = 0;
  std::uint8_t u = 128;
  std::uint8_t v = 128;
};

int AlignEvenDown(int value) {
  return value & ~1;
}

int AlignEvenUp(int value) {
  return (value + 1) & ~1;
}

YuvColor MakeYuvColor(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  int y = 0;
  int u = 0;
  int v = 0;
  RgbToYuv(r, g, b, &y, &u, &v);
  YuvColor color;
  color.y = static_cast<std::uint8_t>(y);
  color.u = static_cast<std::uint8_t>(u);
  color.v = static_cast<std::uint8_t>(v);
  return color;
}

void FillRectNv12(std::uint8_t* y_plane, std::uint8_t* uv_plane, int stride, int image_width, int image_height,
                  int x0, int y0, int x1, int y1, const YuvColor& color) {
  if (y_plane == nullptr || uv_plane == nullptr || stride <= 0 || image_width <= 0 || image_height <= 0) {
    return;
  }
  x0 = std::max(0, std::min(image_width, x0));
  y0 = std::max(0, std::min(image_height, y0));
  x1 = std::max(0, std::min(image_width, x1));
  y1 = std::max(0, std::min(image_height, y1));
  if (x0 >= x1 || y0 >= y1) {
    return;
  }

  for (int y = y0; y < y1; ++y) {
    std::memset(y_plane + static_cast<std::size_t>(y * stride + x0), color.y, static_cast<std::size_t>(x1 - x0));
  }

  const int uv_x0 = AlignEvenDown(x0);
  const int uv_y0 = AlignEvenDown(y0);
  const int uv_x1 = std::min(image_width, AlignEvenUp(x1));
  const int uv_y1 = std::min(image_height, AlignEvenUp(y1));
  for (int y = uv_y0; y < uv_y1; y += 2) {
    std::uint8_t* uv_row = uv_plane + static_cast<std::size_t>((y / 2) * stride);
    for (int x = uv_x0; x < uv_x1; x += 2) {
      uv_row[x + 0] = color.u;
      uv_row[x + 1] = color.v;
    }
  }
}

rk3576_demo::WatermarkImage MoveWatermarkTo(rk3576_demo::WatermarkImage image, int x, int y) {
  image.x = x;
  image.y = y;
  return image;
}

}  // namespace

BranchOutput::BranchOutput(StreamProfile profile, const AppConfigV2& config)
    : profile_(std::move(profile)), config_(config) {}

BranchOutput::~BranchOutput() {
  Stop();
}

bool BranchOutput::ProcessFrame(const rk3576_demo::DecodedFrame& frame, const DetectionFrame* detection) {
  if (frame.width <= 0 || frame.height <= 0) {
    return false;
  }

  const int requested_width = software_fallback_active_ ? frame.width : profile_.width;
  const int requested_height = software_fallback_active_ ? frame.height : profile_.height;
  if (!EnsureReady(requested_width, requested_height)) {
    return false;
  }

  bool prepared = false;
  const auto process_begin = std::chrono::steady_clock::now();
  if (software_fallback_active_) {
    prepared = PrepareFrameWithoutRga(frame);
  } else {
    rk3576_demo::WatermarkImage watermark;
    const rk3576_demo::WatermarkImage* watermark_ptr = nullptr;
    if (profile_.enable_osd) {
      if (profile_.enable_ai_overlay && detection != nullptr) {
        watermark_ptr = &BuildAiPerfWatermark(frame.pts_ms);
      } else {
        watermark_ptr = &BuildTimeWatermark(frame.pts_ms);
      }
    }
    prepared = processor_.ProcessFrame(frame, watermark_ptr != nullptr ? *watermark_ptr : watermark, &encoder_,
                                       profile_.resize_mode, &last_transform_);
    if (!prepared && ActivateSoftwareFallback(frame)) {
      prepared = PrepareFrameWithoutRga(frame);
      if (prepared && watermark_ptr != nullptr && !watermark_ptr->rgba.empty()) {
        prepared = BlendWatermarkOnCurrentFrame(*watermark_ptr);
      }
    }
  }

  if (!prepared) {
    return false;
  }

  if (profile_.enable_ai_overlay && detection != nullptr) {
    prepared = DrawDetectionOverlayOnCurrentFrame(*detection);
    if (!prepared) {
      RKLOG_ERROR("APP") << "[V2][" << ToString(profile_.role) << "] draw AI overlay failed\n";
      return false;
    }
  }

  if (!header_sent_) {
    std::vector<std::uint8_t> header;
    if (encoder_.GetHeader(&header) && !header.empty()) {
      rtsp_.PushH264(header.data(), header.size(), frame.pts_ms);
    }
    header_sent_ = true;
  }

  std::vector<std::uint8_t> packet;
  const auto encode_begin = std::chrono::steady_clock::now();
  if (!encoder_.EncodeCurrentFrame(frame.pts_ms, &packet)) {
    RKLOG_ERROR("V2") << "[" << ToString(profile_.role) << "] encode failed";
    return false;
  }
  const auto encode_end = std::chrono::steady_clock::now();
  last_process_us_ = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(encode_begin - process_begin).count());
  last_encode_us_ = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_begin).count());
  const auto push_begin = std::chrono::steady_clock::now();
  if (!rtsp_.PushH264(packet.data(), packet.size(), frame.pts_ms)) {
    RKLOG_ERROR("APP") << "[V2][" << ToString(profile_.role) << "] push rtsp failed\n";
    return false;
  }
  const auto push_end = std::chrono::steady_clock::now();
  const std::uint64_t push_us = ToMicroseconds(push_end - push_begin);
  const std::uint64_t e2e_us = ToMicroseconds(push_end - process_begin);
  push_acc_.Add(push_us);
  process_acc_.Add(last_process_us_);
  encode_acc_.Add(last_encode_us_);
  e2e_acc_.Add(e2e_us);
  ++processed_frames_;

  MaybePrintPerfSummary(0, dropped_frames_);
  return true;
}

void BranchOutput::Stop() {
  rtsp_.Stop();
  encoder_.Close();
  header_sent_ = false;
  encoder_ready_ = false;
  software_fallback_active_ = false;
  last_transform_ = FrameTransformInfo {};
  actual_width_ = 0;
  actual_height_ = 0;
  last_process_us_ = 0;
  last_encode_us_ = 0;
  osd_clock_initialized_ = false;
  osd_clock_base_pts_ms_ = 0;
  osd_clock_base_system_ms_ = 0;
  osd_clock_last_sync_display_ms_ = 0;
  latest_perf_fps_ = 0.0;
  cached_time_text_.clear();
  cached_time_watermark_ = rk3576_demo::WatermarkImage {};
  cached_ai_perf_text_.clear();
  cached_ai_perf_watermark_ = rk3576_demo::WatermarkImage {};
  cached_label_watermarks_.clear();
}

std::uint64_t BranchOutput::ToMicroseconds(std::chrono::steady_clock::duration duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
}

void BranchOutput::MaybePrintPerfSummary(std::size_t queue_size, std::uint64_t dropped_frames) {
  if (config_.perf_log_interval_ms <= 0) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (last_perf_log_at_.time_since_epoch().count() == 0) {
    last_perf_log_at_ = now;
    perf_window_begin_ = now;
    return;
  }

  const auto interval = std::chrono::milliseconds(config_.perf_log_interval_ms);
  if (now - last_perf_log_at_ < interval) {
    return;
  }

  const double wall_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - perf_window_begin_).count();
  const double fps = wall_seconds <= 0.0 ? 0.0 : static_cast<double>(process_acc_.count) / wall_seconds;
  latest_perf_fps_ = fps;

  auto format_stat = [](const PerfAccumulator& stat) -> std::string {
    std::ostringstream oss;
    if (stat.count == 0) {
      oss << "avg=N/A max=N/A";
    } else {
      oss << "avg=" << static_cast<std::uint64_t>(stat.Average())
          << "us max=" << stat.max_us << "us";
    }
    return oss.str();
  };

  RKLOG_INFO("V2-PERF")
      << "[" << ToString(profile_.role) << "]"
      << " frames=" << process_acc_.count
      << " process=" << format_stat(process_acc_)
      << " encode=" << format_stat(encode_acc_)
      << " push=" << format_stat(push_acc_)
      << " e2e=" << format_stat(e2e_acc_)
      << " fps=" << std::fixed << std::setprecision(1) << fps
      << " queue=" << queue_size
      << " drop=" << dropped_frames
      << " fallback=" << (software_fallback_active_ ? "on" : "off")
      << " output=" << actual_width_ << "x" << actual_height_;

  last_perf_log_at_ = now;
  perf_window_begin_ = now;
  process_acc_.Reset();
  encode_acc_.Reset();
  push_acc_.Reset();
  e2e_acc_.Reset();
}

bool BranchOutput::EnsureReady(int width, int height) {
  if (encoder_ready_ && actual_width_ == width && actual_height_ == height) {
    return true;
  }

  rtsp_.Stop();
  encoder_.Close();
  header_sent_ = false;
  encoder_ready_ = false;

  if (!encoder_.Open(width, height, config_.fps, profile_.bitrate > 0 ? profile_.bitrate : config_.bitrate)) {
    RKLOG_ERROR("APP") << "[V2][" << ToString(profile_.role) << "] init encoder failed\n";
    return false;
  }

  rk3576_demo::AppConfig stream_config = MakeStreamConfig();
  if (!rtsp_.Start(stream_config, width, height)) {
    RKLOG_ERROR("APP") << "[V2][" << ToString(profile_.role) << "] init RTSP server failed\n";
    encoder_.Close();
    return false;
  }

  actual_width_ = width;
  actual_height_ = height;
  last_transform_ = ComputeFrameTransform(profile_.resize_mode, width, height, width, height);
  encoder_ready_ = true;
  RKLOG_INFO("APP") << "[V2][" << ToString(profile_.role) << "] RTSP ready at " << rtsp_.RtspUrl()
            << " output=" << actual_width_ << "x" << actual_height_ << "\n";
  return true;
}

bool BranchOutput::ActivateSoftwareFallback(const rk3576_demo::DecodedFrame& frame) {
  if (software_fallback_active_) {
    return true;
  }
  software_fallback_active_ = true;

  if (!EnsureReady(frame.width, frame.height)) {
    return false;
  }

  RKLOG_ERROR("APP") << "[V2][" << ToString(profile_.role)
            << "] fallback to software copy/conversion. Resize/overlay dependent on RGA are disabled; "
            << "output switches to " << frame.width << "x" << frame.height << ".\n";
  return true;
}

bool BranchOutput::PrepareFrameWithoutRga(const rk3576_demo::DecodedFrame& frame) {
  if (!encoder_ready_ || encoder_.input_addr() == nullptr || frame.virt_addr == nullptr) {
    return false;
  }
  if (frame.width != encoder_.width() || frame.height != encoder_.height()) {
    RKLOG_ERROR("APP") << "[V2][" << ToString(profile_.role) << "] software fallback only supports matching dimensions. "
              << "decoded=" << frame.width << "x" << frame.height
              << " encoder=" << encoder_.width() << "x" << encoder_.height() << "\n";
    return false;
  }

  const int src_fmt = frame.format & MPP_FRAME_FMT_MASK;
  if (src_fmt != MPP_FMT_YUV420SP && src_fmt != MPP_FMT_YUV420SP_VU &&
      src_fmt != MPP_FMT_YUV422SP && src_fmt != MPP_FMT_YUV422SP_VU) {
    RKLOG_ERROR("APP") << "[V2][" << ToString(profile_.role) << "] unsupported software fallback format: "
              << frame.format << "\n";
    return false;
  }

  auto* dst = static_cast<std::uint8_t*>(encoder_.input_addr());
  const int dst_y_stride = encoder_.hor_stride();
  const int dst_uv_stride = encoder_.hor_stride();
  const int dst_ver_stride = encoder_.ver_stride();
  std::memset(dst, 0, static_cast<std::size_t>(dst_y_stride * dst_ver_stride));
  std::memset(dst + static_cast<std::size_t>(dst_y_stride * dst_ver_stride), 128,
              static_cast<std::size_t>(dst_uv_stride * dst_ver_stride / 2));

  const auto* src_y = static_cast<const std::uint8_t*>(frame.virt_addr);
  const auto* src_uv = src_y + static_cast<std::size_t>(frame.hor_stride * frame.ver_stride);
  auto* dst_y = dst;
  auto* dst_uv = dst + static_cast<std::size_t>(dst_y_stride * dst_ver_stride);

  for (int y = 0; y < frame.height; ++y) {
    std::memcpy(dst_y + static_cast<std::size_t>(y * dst_y_stride),
                src_y + static_cast<std::size_t>(y * frame.hor_stride),
                static_cast<std::size_t>(frame.width));
  }

  const bool swap_uv = (src_fmt == MPP_FMT_YUV420SP_VU || src_fmt == MPP_FMT_YUV422SP_VU);
  if (src_fmt == MPP_FMT_YUV420SP || src_fmt == MPP_FMT_YUV420SP_VU) {
    for (int y = 0; y < frame.height / 2; ++y) {
      if (!CopySemiPlanarRowToNv12(src_uv + static_cast<std::size_t>(y * frame.hor_stride),
                                   frame.width, swap_uv,
                                   dst_uv + static_cast<std::size_t>(y * dst_uv_stride))) {
        return false;
      }
    }
    return true;
  }

  for (int y = 0; y < frame.height / 2; ++y) {
    const auto* src_uv_row = src_uv + static_cast<std::size_t>((y * 2) * frame.hor_stride);
    if (!CopySemiPlanarRowToNv12(src_uv_row, frame.width, swap_uv,
                                 dst_uv + static_cast<std::size_t>(y * dst_uv_stride))) {
      return false;
    }
  }
  return true;
}

bool BranchOutput::BlendWatermarkOnCurrentFrame(const rk3576_demo::WatermarkImage& watermark) {
  if (encoder_.input_addr() == nullptr || watermark.rgba.empty()) {
    return false;
  }

  auto* base = static_cast<std::uint8_t*>(encoder_.input_addr());
  auto* y_plane = base;
  auto* uv_plane = base + static_cast<std::size_t>(encoder_.hor_stride() * encoder_.ver_stride());

  const int overlay_x0 = std::max(0, watermark.x);
  const int overlay_y0 = std::max(0, watermark.y);
  const int overlay_x1 = std::min(actual_width_, watermark.x + watermark.width);
  const int overlay_y1 = std::min(actual_height_, watermark.y + watermark.height);
  if (overlay_x0 >= overlay_x1 || overlay_y0 >= overlay_y1) {
    return true;
  }

  for (int y = overlay_y0; y < overlay_y1; ++y) {
    const int wm_y = y - watermark.y;
    for (int x = overlay_x0; x < overlay_x1; ++x) {
      const int wm_x = x - watermark.x;
      const std::size_t rgba_index = static_cast<std::size_t>(wm_y * watermark.width + wm_x) * 4;
      const int alpha = watermark.rgba[rgba_index + 3];
      if (alpha == 0) {
        continue;
      }

      int yuv_y = 0;
      int yuv_u = 0;
      int yuv_v = 0;
      RgbToYuv(watermark.rgba[rgba_index + 0], watermark.rgba[rgba_index + 1], watermark.rgba[rgba_index + 2],
               &yuv_y, &yuv_u, &yuv_v);
      const std::size_t y_index = static_cast<std::size_t>(y * encoder_.hor_stride() + x);
      y_plane[y_index] = static_cast<std::uint8_t>(BlendChannel(yuv_y, y_plane[y_index], alpha));
    }
  }

  const int uv_x0 = overlay_x0 & ~1;
  const int uv_y0 = overlay_y0 & ~1;
  const int uv_x1 = overlay_x1 & ~1;
  const int uv_y1 = overlay_y1 & ~1;
  for (int y = uv_y0; y < uv_y1; y += 2) {
    const int uv_row = y / 2;
    for (int x = uv_x0; x < uv_x1; x += 2) {
      int r_sum = 0;
      int g_sum = 0;
      int b_sum = 0;
      int a_sum = 0;
      int sample_count = 0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const int px = x + dx;
          const int py = y + dy;
          if (px < overlay_x0 || px >= overlay_x1 || py < overlay_y0 || py >= overlay_y1) {
            continue;
          }
          const int wm_x = px - watermark.x;
          const int wm_y = py - watermark.y;
          const std::size_t rgba_index = static_cast<std::size_t>(wm_y * watermark.width + wm_x) * 4;
          r_sum += watermark.rgba[rgba_index + 0];
          g_sum += watermark.rgba[rgba_index + 1];
          b_sum += watermark.rgba[rgba_index + 2];
          a_sum += watermark.rgba[rgba_index + 3];
          ++sample_count;
        }
      }
      if (sample_count == 0 || a_sum == 0) {
        continue;
      }
      int yuv_y = 0;
      int yuv_u = 0;
      int yuv_v = 0;
      RgbToYuv(static_cast<std::uint8_t>(r_sum / sample_count), static_cast<std::uint8_t>(g_sum / sample_count),
               static_cast<std::uint8_t>(b_sum / sample_count), &yuv_y, &yuv_u, &yuv_v);
      const int alpha = a_sum / sample_count;
      const std::size_t uv_index = static_cast<std::size_t>(uv_row * encoder_.hor_stride() + x);
      uv_plane[uv_index + 0] = static_cast<std::uint8_t>(BlendChannel(yuv_u, uv_plane[uv_index + 0], alpha));
      uv_plane[uv_index + 1] = static_cast<std::uint8_t>(BlendChannel(yuv_v, uv_plane[uv_index + 1], alpha));
    }
  }
  return true;
}

bool BranchOutput::DrawDetectionOverlayOnCurrentFrame(const DetectionFrame& detection) {
  if (encoder_.input_addr() == nullptr) {
    return false;
  }
  if (detection.source_width <= 0 || detection.source_height <= 0) {
    return true;
  }

  auto* base = static_cast<std::uint8_t*>(encoder_.input_addr());
  auto* y_plane = base;
  auto* uv_plane = base + static_cast<std::size_t>(encoder_.hor_stride() * encoder_.ver_stride());
  FrameTransformInfo transform = last_transform_;
  if (!transform.valid) {
    transform = ComputeFrameTransform(profile_.resize_mode, detection.source_width, detection.source_height,
                                      actual_width_, actual_height_);
  }

  for (std::size_t i = 0; i < detection.boxes.size(); ++i) {
    const DetectionBox& box = detection.boxes[i];
    const std::uint8_t* color = kOverlayColors[i % (sizeof(kOverlayColors) / sizeof(kOverlayColors[0]))];
    const YuvColor yuv_color = MakeYuvColor(color[0], color[1], color[2]);
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
    if (transform.mode == ResizeMode::kCenterCrop && transform.src_width > 0 && transform.src_height > 0) {
      const float scale_x = static_cast<float>(transform.dst_width) / static_cast<float>(transform.src_width);
      const float scale_y = static_cast<float>(transform.dst_height) / static_cast<float>(transform.src_height);
      left = static_cast<int>((box.x - transform.src_x) * scale_x);
      top = static_cast<int>((box.y - transform.src_y) * scale_y);
      right = static_cast<int>(((box.x + box.width) - transform.src_x) * scale_x);
      bottom = static_cast<int>(((box.y + box.height) - transform.src_y) * scale_y);
    } else {
      const float scale_x = transform.dst_width > 0 && detection.source_width > 0
                                ? static_cast<float>(transform.dst_width) / static_cast<float>(detection.source_width)
                                : 1.0f;
      const float scale_y = transform.dst_height > 0 && detection.source_height > 0
                                ? static_cast<float>(transform.dst_height) / static_cast<float>(detection.source_height)
                                : 1.0f;
      left = transform.dst_x + static_cast<int>(box.x * scale_x);
      top = transform.dst_y + static_cast<int>(box.y * scale_y);
      right = transform.dst_x + static_cast<int>((box.x + box.width) * scale_x);
      bottom = transform.dst_y + static_cast<int>((box.y + box.height) * scale_y);
    }
    left = std::max(0, std::min(actual_width_ - 1, left));
    top = std::max(0, std::min(actual_height_ - 1, top));
    right = std::max(0, std::min(actual_width_ - 1, right));
    bottom = std::max(0, std::min(actual_height_ - 1, bottom));
    if (right <= left || bottom <= top) {
      continue;
    }

    for (int t = 0; t < 2; ++t) {
      FillRectNv12(y_plane, uv_plane, encoder_.hor_stride(), actual_width_, actual_height_,
                   left, top + t, right + 1, top + t + 1, yuv_color);
      FillRectNv12(y_plane, uv_plane, encoder_.hor_stride(), actual_width_, actual_height_,
                   left, bottom - t, right + 1, bottom - t + 1, yuv_color);
      FillRectNv12(y_plane, uv_plane, encoder_.hor_stride(), actual_width_, actual_height_,
                   left + t, top, left + t + 1, bottom + 1, yuv_color);
      FillRectNv12(y_plane, uv_plane, encoder_.hor_stride(), actual_width_, actual_height_,
                   right - t, top, right - t + 1, bottom + 1, yuv_color);
    }

    if (!box.class_name.empty()) {
      const int label_x = left;
      const int label_y = (top >= 28) ? (top - 28) : std::min(actual_height_ - 20, top + 4);
      rk3576_demo::WatermarkImage label = BuildCachedLabelWatermark(box.class_name, label_x, label_y);
      if (!label.rgba.empty()) {
        BlendWatermarkOnCurrentFrame(label);
      }
    }
  }
  return true;
}

const rk3576_demo::WatermarkImage& BranchOutput::BuildTimeWatermark(std::uint64_t pts_ms) {
  const std::string text = BuildOsdTimestampText(pts_ms);
  if (text != cached_time_text_ || cached_time_watermark_.rgba.empty()) {
    cached_time_text_ = text;
    cached_time_watermark_ = watermark_.RenderText(text);
  }
  return cached_time_watermark_;
}

const rk3576_demo::WatermarkImage& BranchOutput::BuildAiPerfWatermark(std::uint64_t pts_ms) {
  const double fps = CurrentAiOsdFps();
  std::ostringstream oss;
  oss << BuildOsdTimestampText(pts_ms) << " FPS " << std::fixed << std::setprecision(1) << fps;
  const std::string text = oss.str();
  if (text != cached_ai_perf_text_ || cached_ai_perf_watermark_.rgba.empty()) {
    cached_ai_perf_text_ = text;
    cached_ai_perf_watermark_ = watermark_.RenderText(text);
  }
  return cached_ai_perf_watermark_;
}

std::string BranchOutput::BuildOsdTimestampText(std::uint64_t pts_ms) {
  const std::uint64_t display_ms = ResolveOsdTimestampMs(pts_ms);
  const std::time_t seconds = static_cast<std::time_t>(display_ms / 1000ULL);
  std::tm local_tm {};
  localtime_r(&seconds, &local_tm);
  std::ostringstream oss;
  oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

std::uint64_t BranchOutput::ResolveOsdTimestampMs(std::uint64_t pts_ms) {
  const std::uint64_t now_ms = SystemNowMs();
  if (!osd_clock_initialized_ || pts_ms < osd_clock_base_pts_ms_) {
    osd_clock_initialized_ = true;
    osd_clock_base_pts_ms_ = pts_ms;
    osd_clock_base_system_ms_ = now_ms;
    osd_clock_last_sync_display_ms_ = now_ms;
    return now_ms;
  }

  const std::uint64_t delta_ms = pts_ms - osd_clock_base_pts_ms_;
  std::uint64_t display_ms = osd_clock_base_system_ms_ + delta_ms;
  if (display_ms >= osd_clock_last_sync_display_ms_ + 60ULL * 1000ULL) {
    osd_clock_base_pts_ms_ = pts_ms;
    osd_clock_base_system_ms_ = now_ms;
    osd_clock_last_sync_display_ms_ = now_ms;
    display_ms = now_ms;
  }
  return display_ms;
}

rk3576_demo::WatermarkImage BranchOutput::BuildCachedLabelWatermark(const std::string& text, int x, int y) {
  auto it = cached_label_watermarks_.find(text);
  if (it == cached_label_watermarks_.end()) {
    it = cached_label_watermarks_.emplace(text, watermark_.RenderText(text, 0, 0)).first;
  }
  return MoveWatermarkTo(it->second, x, y);
}

double BranchOutput::CurrentAiOsdFps() const {
  if (latest_perf_fps_ > 0.0) {
    return latest_perf_fps_;
  }
  return static_cast<double>(config_.fps > 0 ? config_.fps : 0);
}

rk3576_demo::AppConfig BranchOutput::MakeStreamConfig() const {
  rk3576_demo::AppConfig config;
  config.device = config_.device;
  config.input_codec = rk3576_demo::InputCodec::kMjpeg;
  config.rtsp_app = profile_.app_name;
  config.rtsp_stream = profile_.stream_name;
  config.camera_width = config_.camera_width;
  config.camera_height = config_.camera_height;
  config.output_width = actual_width_;
  config.output_height = actual_height_;
  config.fps = config_.fps;
  config.rtsp_port = config_.rtsp_port;
  config.bitrate = profile_.bitrate > 0 ? profile_.bitrate : config_.bitrate;
  return config;
}

}  // namespace rk3576_yolo_demo
