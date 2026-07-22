#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <vector>

#include "rk3576_demo/app_config.hpp"
#include "rk3576_demo/mpp_decoder.hpp"
#include "rk3576_demo/mpp_encoder.hpp"
#include "rk3576_demo/rga_processor.hpp"
#include "rk3576_demo/rtsp_server.hpp"
#include "rk3576_demo/system_monitor.hpp"
#include "rk3576_demo/v4l2_camera.hpp"
#include "rk3576_demo/watermark_renderer.hpp"

namespace rk3576_demo {

class PipelineApp {
 public:
  using DecodedFrameObserver = std::function<void(const DecodedFrame&)>;

  explicit PipelineApp(AppConfig config);
  ~PipelineApp();

  bool Run();
  void SetDecodedFrameObserver(const DecodedFrameObserver& observer) { decoded_frame_observer_ = observer; }

 private:
  struct FrameTrace {
    std::chrono::steady_clock::time_point captured_at;
    std::size_t mjpg_size = 0;
  };

  struct PerfAccumulator {
    std::uint64_t count = 0;
    std::uint64_t total_us = 0;
    std::uint64_t max_us = 0;

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

  bool InitEncoderAndRtsp(int output_width = 0, int output_height = 0);
  bool ActivateSoftwareFallback(const DecodedFrame& frame);
  bool PrepareFrameWithoutRga(const DecodedFrame& frame);
  void OnResourceSnapshot(const ResourceSnapshot& snapshot);
  void MaybePrintPerfSummary();
  static std::uint64_t ToMicroseconds(std::chrono::steady_clock::duration duration);
  void OnDecodedFrame(const DecodedFrame& frame);
  bool OpenDumpFiles();
  void CloseDumpFiles();
  void DumpMjpgPacket(const std::uint8_t* data, std::size_t size);
  void DumpPacket(const std::vector<std::uint8_t>& packet);

  AppConfig config_;
  V4L2Camera camera_;
  MppDecoder decoder_;
  MppEncoder encoder_;
  RgaProcessor processor_;
  WatermarkRenderer watermark_;
  RtspServer rtsp_;
  std::FILE* dump_h264_file_ = nullptr;
  std::FILE* dump_mjpg_file_ = nullptr;
  std::uint64_t capture_count_ = 0;
  std::uint64_t decoded_count_ = 0;
  std::uint64_t encoded_count_ = 0;
  bool header_sent_ = false;
  bool encoder_ready_ = false;
  bool software_fallback_active_ = false;
  int active_output_width_ = 0;
  int active_output_height_ = 0;
  SystemMonitor system_monitor_;
  std::unordered_map<std::uint64_t, FrameTrace> frame_traces_;
  std::chrono::steady_clock::time_point last_perf_log_at_ {};
  ResourceSnapshot latest_resource_snapshot_ {};
  bool has_latest_resource_snapshot_ = false;
  PerfAccumulator decode_total_acc_;
  PerfAccumulator decode_wait_acc_;
  PerfAccumulator process_acc_;
  PerfAccumulator encode_acc_;
  PerfAccumulator header_acc_;
  PerfAccumulator rtsp_push_acc_;
  PerfAccumulator end_to_end_acc_;
  DecodedFrameObserver decoded_frame_observer_;
};

}  // namespace rk3576_demo
