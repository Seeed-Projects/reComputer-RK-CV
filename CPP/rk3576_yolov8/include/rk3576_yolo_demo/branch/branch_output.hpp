#pragma once

#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

#include "rk3576_demo/app_config.hpp"
#include "rk3576_demo/media_types.hpp"
#include "rk3576_demo/mpp_encoder.hpp"
#include "rk3576_demo/rga_processor.hpp"
#include "rk3576_demo/rtsp_server.hpp"
#include "rk3576_demo/watermark_renderer.hpp"
#include "rk3576_yolo_demo/app/app_config_v2.hpp"
#include "rk3576_yolo_demo/common/runtime_types.hpp"

namespace rk3576_yolo_demo {

class BranchOutput {
 public:
  BranchOutput(StreamProfile profile, const AppConfigV2& config);
  ~BranchOutput();

  bool ProcessFrame(const rk3576_demo::DecodedFrame& frame, const DetectionFrame* detection = nullptr);
  void Stop();
  std::string stream_name() const { return profile_.stream_name; }
  StreamRole role() const { return profile_.role; }
  int actual_width() const { return actual_width_; }
  int actual_height() const { return actual_height_; }
  bool fallback_active() const { return software_fallback_active_; }

 private:
  struct PerfAccumulator {
    std::uint64_t count = 0;
    std::uint64_t total_us = 0;
    std::uint64_t max_us = 0;

    double Average() const {
      return count == 0 ? 0.0 : static_cast<double>(total_us) / static_cast<double>(count);
    }

    void Add(std::uint64_t value_us) {
      ++count;
      total_us += value_us;
      if (value_us > max_us) {
        max_us = value_us;
      }
    }

    void Reset() {
      count = 0;
      total_us = 0;
      max_us = 0;
    }
  };

  bool EnsureReady(int width, int height);
  bool ActivateSoftwareFallback(const rk3576_demo::DecodedFrame& frame);
  bool PrepareFrameWithoutRga(const rk3576_demo::DecodedFrame& frame);
  bool BlendWatermarkOnCurrentFrame(const rk3576_demo::WatermarkImage& watermark);
  bool DrawDetectionOverlayOnCurrentFrame(const DetectionFrame& detection);
  const rk3576_demo::WatermarkImage& BuildTimeWatermark(std::uint64_t pts_ms);
  const rk3576_demo::WatermarkImage& BuildAiPerfWatermark(std::uint64_t pts_ms);
  std::string BuildOsdTimestampText(std::uint64_t pts_ms);
  std::uint64_t ResolveOsdTimestampMs(std::uint64_t pts_ms);
  rk3576_demo::WatermarkImage BuildCachedLabelWatermark(const std::string& text, int x, int y);
  double CurrentAiOsdFps() const;
  void MaybePrintPerfSummary(std::size_t queue_size, std::uint64_t dropped_frames);
  static std::uint64_t ToMicroseconds(std::chrono::steady_clock::duration duration);
  rk3576_demo::AppConfig MakeStreamConfig() const;

  StreamProfile profile_;
  AppConfigV2 config_;
  rk3576_demo::MppEncoder encoder_;
  rk3576_demo::RgaProcessor processor_;
  rk3576_demo::WatermarkRenderer watermark_;
  rk3576_demo::RtspServer rtsp_;
  bool header_sent_ = false;
  bool encoder_ready_ = false;
  bool software_fallback_active_ = false;
  FrameTransformInfo last_transform_;
  int actual_width_ = 0;
  int actual_height_ = 0;
  std::uint64_t processed_frames_ = 0;
  std::uint64_t dropped_frames_ = 0;
  std::uint64_t last_process_us_ = 0;
  std::uint64_t last_encode_us_ = 0;
  bool osd_clock_initialized_ = false;
  std::uint64_t osd_clock_base_pts_ms_ = 0;
  std::uint64_t osd_clock_base_system_ms_ = 0;
  std::uint64_t osd_clock_last_sync_display_ms_ = 0;
  PerfAccumulator process_acc_;
  PerfAccumulator encode_acc_;
  PerfAccumulator push_acc_;
  PerfAccumulator e2e_acc_;
  double latest_perf_fps_ = 0.0;
  std::string cached_time_text_;
  rk3576_demo::WatermarkImage cached_time_watermark_;
  std::string cached_ai_perf_text_;
  rk3576_demo::WatermarkImage cached_ai_perf_watermark_;
  std::unordered_map<std::string, rk3576_demo::WatermarkImage> cached_label_watermarks_;
  std::chrono::steady_clock::time_point last_perf_log_at_ {};
  std::chrono::steady_clock::time_point perf_window_begin_ {};
};

}  // namespace rk3576_yolo_demo
