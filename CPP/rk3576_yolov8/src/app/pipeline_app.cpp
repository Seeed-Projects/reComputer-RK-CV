#include "rk3576_demo/pipeline_app.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_demo {

namespace {

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

}  // namespace

PipelineApp::PipelineApp(AppConfig config) : config_(std::move(config)) {}

PipelineApp::~PipelineApp() {
  system_monitor_.Stop();
  CloseDumpFiles();
}

bool PipelineApp::Run() {
  if (!OpenDumpFiles()) {
    return false;
  }

  if (!decoder_.Open(config_.input_codec, config_.fps, config_.camera_width, config_.camera_height,
                     [this](const DecodedFrame& frame) { OnDecodedFrame(frame); })) {
    RKLOG_ERROR("APP") << "Failed to initialize MPP decoder\n";
    return false;
  }

  if (!camera_.Open(config_)) {
    RKLOG_ERROR("APP") << "Failed to initialize V4L2 camera\n";
    return false;
  }

  if (config_.perf_log_interval_ms > 0) {
    last_perf_log_at_ = std::chrono::steady_clock::now();
    system_monitor_.Start(std::chrono::milliseconds(config_.perf_log_interval_ms),
                          [this](const ResourceSnapshot& snapshot) { OnResourceSnapshot(snapshot); });
  }

  if (config_.decode_only) {
    RKLOG_INFO("APP") << "PipelineApp decode-only mode enabled. Encoding and RTSP publishing are handled by V2 branches.\n";
  } else if (!InitEncoderAndRtsp()) {
    return false;
  }

  RKLOG_INFO("APP") << "Capture started from " << config_.device << "\n";
  const bool ok = camera_.CaptureLoop(
      [this](const std::uint8_t* data, std::size_t size, std::uint64_t pts_ms) {
        ++capture_count_;
        frame_traces_[pts_ms] = {std::chrono::steady_clock::now(), size};
        if (capture_count_ <= 5 || (capture_count_ % 30) == 0) {
          RKLOG_INFO("APP") << "Captured MJPG packet #" << capture_count_
                    << " size=" << size
                    << " pts_ms=" << pts_ms << "\n";
        }
        DumpMjpgPacket(data, size);
        return decoder_.Decode(data, size, false, pts_ms);
      },
      config_.frame_limit);
  system_monitor_.Stop();
  return ok;
}

bool PipelineApp::InitEncoderAndRtsp(int output_width, int output_height) {
  const int out_width = output_width > 0 ? output_width : config_.output_width;
  const int out_height = output_height > 0 ? output_height : config_.output_height;

  if (encoder_ready_ && out_width == active_output_width_ && out_height == active_output_height_) {
    return true;
  }

  if (encoder_ready_) {
    rtsp_.Stop();
    encoder_.Close();
    encoder_ready_ = false;
  }
  header_sent_ = false;

  if (!encoder_.Open(out_width, out_height, config_.fps, config_.bitrate)) {
    RKLOG_ERROR("APP") << "Failed to initialize MPP encoder\n";
    return false;
  }
  if (!rtsp_.Start(config_, out_width, out_height)) {
    RKLOG_ERROR("APP") << "Failed to initialize RTSP server\n";
    return false;
  }

  RKLOG_INFO("APP") << "RTSP service ready at " << rtsp_.RtspUrl() << "\n";
  active_output_width_ = out_width;
  active_output_height_ = out_height;
  encoder_ready_ = true;
  return true;
}

void PipelineApp::OnResourceSnapshot(const ResourceSnapshot& snapshot) {
  latest_resource_snapshot_ = snapshot;
  has_latest_resource_snapshot_ = true;
  MaybePrintPerfSummary();
}

void PipelineApp::MaybePrintPerfSummary() {
  if (config_.perf_log_interval_ms <= 0) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (last_perf_log_at_.time_since_epoch().count() == 0) {
    last_perf_log_at_ = now;
    return;
  }
  if (now - last_perf_log_at_ < std::chrono::milliseconds(config_.perf_log_interval_ms)) {
    return;
  }
  last_perf_log_at_ = now;

  auto format_stat = [](const PerfAccumulator& stat) -> std::string {
    std::ostringstream oss;
    if (stat.count == 0) {
      oss << "avg=N/A max=N/A";
    } else {
      oss << "avg=" << (stat.total_us / stat.count) << "us max=" << stat.max_us << "us";
    }
    return oss.str();
  };

  auto format_percent_value = [](bool has_value, double value) -> std::string {
    if (!has_value) {
      return "N/A";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << value << "%";
    return oss.str();
  };

  RKLOG_INFO("APP") << "[PERF] frames cap/dec/enc=" << capture_count_ << "/" << decoded_count_ << "/" << encoded_count_
            << " | decode=" << format_stat(decode_total_acc_)
            << " wait=" << format_stat(decode_wait_acc_)
            << " process=" << format_stat(process_acc_)
            << " encode=" << format_stat(encode_acc_)
            << " header=" << format_stat(header_acc_)
            << " rtsp_push=" << format_stat(rtsp_push_acc_)
            << " e2e_submit=" << format_stat(end_to_end_acc_);

  if (has_latest_resource_snapshot_) {
    RKLOG_INFO("APP") << " | CPU=" << format_percent_value(latest_resource_snapshot_.has_cpu, latest_resource_snapshot_.cpu_percent)
              << " GPU=" << format_percent_value(latest_resource_snapshot_.has_gpu, latest_resource_snapshot_.gpu_percent)
              << " VPU=" << format_percent_value(latest_resource_snapshot_.has_vpu, latest_resource_snapshot_.vpu_percent);
    if (!latest_resource_snapshot_.gpu_source.empty()) {
      RKLOG_INFO("APP") << " gpu_src=" << latest_resource_snapshot_.gpu_source;
    }
    if (!latest_resource_snapshot_.vpu_source.empty()) {
      RKLOG_INFO("APP") << " vpu_src=" << latest_resource_snapshot_.vpu_source;
    }
  }
  RKLOG_INFO("APP") << "\n";

  decode_total_acc_.Reset();
  decode_wait_acc_.Reset();
  process_acc_.Reset();
  encode_acc_.Reset();
  header_acc_.Reset();
  rtsp_push_acc_.Reset();
  end_to_end_acc_.Reset();
}

std::uint64_t PipelineApp::ToMicroseconds(std::chrono::steady_clock::duration duration) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
}

bool PipelineApp::ActivateSoftwareFallback(const DecodedFrame& frame) {
  if (software_fallback_active_) {
    return true;
  }

  if (frame.width <= 0 || frame.height <= 0) {
    return false;
  }

  RKLOG_ERROR("APP") << "RGA unavailable, fallback to software copy/conversion. "
            << "Resize and watermark are disabled; RTSP output switches to "
            << frame.width << "x" << frame.height << ".\n";

  software_fallback_active_ = true;
  return InitEncoderAndRtsp(frame.width, frame.height);
}

bool PipelineApp::PrepareFrameWithoutRga(const DecodedFrame& frame) {
  if (!encoder_ready_ || encoder_.input_addr() == nullptr || frame.virt_addr == nullptr) {
    return false;
  }
  if (frame.width != encoder_.width() || frame.height != encoder_.height()) {
    RKLOG_ERROR("APP") << "Software fallback only supports matching dimensions. decoded="
              << frame.width << "x" << frame.height
              << " encoder=" << encoder_.width() << "x" << encoder_.height() << "\n";
    return false;
  }

  const int src_fmt = frame.format & MPP_FRAME_FMT_MASK;
  if (src_fmt != MPP_FMT_YUV420SP && src_fmt != MPP_FMT_YUV420SP_VU &&
      src_fmt != MPP_FMT_YUV422SP && src_fmt != MPP_FMT_YUV422SP_VU) {
    RKLOG_ERROR("APP") << "Software fallback does not support frame format: " << frame.format << "\n";
    return false;
  }

  auto* dst = static_cast<std::uint8_t*>(encoder_.input_addr());
  if (dst == nullptr) {
    return false;
  }

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

void PipelineApp::OnDecodedFrame(const DecodedFrame& frame) {
  ++decoded_count_;
  decode_total_acc_.Add(frame.decode_total_us);
  decode_wait_acc_.Add(frame.decode_wait_us);
  if (decoded_frame_observer_) {
    decoded_frame_observer_(frame);
  }
  RKLOG_INFO("APP") << "Pipeline received decoded frame #" << decoded_count_
            << " width=" << frame.width
            << " height=" << frame.height
            << " fmt=" << frame.format
            << " fd=" << frame.fd
            << " pts_ms=" << frame.pts_ms
            << " decode_total=" << frame.decode_total_us << "us"
            << " decode_wait=" << frame.decode_wait_us << "us"
            << " decode_put=" << frame.decode_put_us << "us\n";

  if (config_.decode_only) {
    MaybePrintPerfSummary();
    return;
  }

  const auto process_begin = std::chrono::steady_clock::now();

  bool prepared = false;
  if (software_fallback_active_) {
    prepared = PrepareFrameWithoutRga(frame);
  } else {
    const WatermarkImage watermark = watermark_.Render(frame.pts_ms);
    prepared = processor_.ProcessFrame(frame, watermark, &encoder_);
    if (!prepared && ActivateSoftwareFallback(frame)) {
      prepared = PrepareFrameWithoutRga(frame);
    }
  }

  if (!prepared) {
    RKLOG_ERROR("APP") << "Frame processing failed\n";
    return;
  }
  const auto process_end = std::chrono::steady_clock::now();
  const auto process_us = ToMicroseconds(process_end - process_begin);
  process_acc_.Add(process_us);

  if (!header_sent_) {
    const auto header_begin = std::chrono::steady_clock::now();
    std::vector<std::uint8_t> header;
    if (encoder_.GetHeader(&header)) {
      const auto header_end = std::chrono::steady_clock::now();
      const auto header_us = ToMicroseconds(header_end - header_begin);
      header_acc_.Add(header_us);
      DumpPacket(header);
      const auto push_begin = std::chrono::steady_clock::now();
      rtsp_.PushH264(header.data(), header.size(), frame.pts_ms);
      const auto push_end = std::chrono::steady_clock::now();
      rtsp_push_acc_.Add(ToMicroseconds(push_end - push_begin));
      RKLOG_INFO("APP") << "Frame #" << decoded_count_
                << " header_us=" << header_us
                << " rtsp_header_push_us=" << ToMicroseconds(push_end - push_begin)
                << " header_size=" << header.size() << "\n";
    }
    header_sent_ = true;
  }

  std::vector<std::uint8_t> encoded_frame;
  const auto encode_begin = std::chrono::steady_clock::now();
  if (!encoder_.EncodeCurrentFrame(frame.pts_ms, &encoded_frame)) {
    RKLOG_ERROR("APP") << "Frame encoding failed\n";
    return;
  }
  const auto encode_end = std::chrono::steady_clock::now();
  const auto encode_us = ToMicroseconds(encode_end - encode_begin);
  encode_acc_.Add(encode_us);

  ++encoded_count_;
  const auto push_begin = std::chrono::steady_clock::now();
  if (encoded_count_ <= 5 || (encoded_count_ % 30) == 0) {
    RKLOG_INFO("APP") << "Pipeline encoded H.264 frame #" << encoded_count_
              << " size=" << encoded_frame.size()
              << " pts_ms=" << frame.pts_ms
              << " process_us=" << process_us
              << " encode_us=" << encode_us << "\n";
  }
  DumpPacket(encoded_frame);
  if (!rtsp_.PushH264(encoded_frame.data(), encoded_frame.size(), frame.pts_ms)) {
    RKLOG_ERROR("APP") << "Push H.264 frame to RTSP failed\n";
  } else {
    const auto push_end = std::chrono::steady_clock::now();
    const auto push_us = ToMicroseconds(push_end - push_begin);
    rtsp_push_acc_.Add(push_us);

    auto trace_it = frame_traces_.find(frame.pts_ms);
    if (trace_it != frame_traces_.end()) {
      const auto e2e_us = ToMicroseconds(push_end - trace_it->second.captured_at);
      end_to_end_acc_.Add(e2e_us);
      if (decoded_count_ <= 5 || (decoded_count_ % 30) == 0) {
        RKLOG_INFO("APP") << "Frame #" << decoded_count_
                  << " submit_latency_us=" << push_us
                  << " e2e_submit_us=" << e2e_us
                  << " mjpg_size=" << trace_it->second.mjpg_size
                  << " h264_size=" << encoded_frame.size()
                  << " fallback=" << (software_fallback_active_ ? "on" : "off") << "\n";
      }
      frame_traces_.erase(trace_it);
    }
  }

  MaybePrintPerfSummary();
}

bool PipelineApp::OpenDumpFiles() {
  if (config_.dump_h264) {
    dump_h264_file_ = std::fopen(config_.dump_h264_path.c_str(), "wb");
    if (dump_h264_file_ == nullptr) {
      RKLOG_ERROR("APP") << "Failed to open H.264 dump file: " << config_.dump_h264_path << "\n";
      return false;
    }
  }
  if (config_.dump_mjpg) {
    dump_mjpg_file_ = std::fopen(config_.dump_mjpg_path.c_str(), "wb");
    if (dump_mjpg_file_ == nullptr) {
      RKLOG_ERROR("APP") << "Failed to open MJPG dump file: " << config_.dump_mjpg_path << "\n";
      return false;
    }
  }
  return true;
}

void PipelineApp::CloseDumpFiles() {
  if (dump_h264_file_ != nullptr) {
    std::fclose(dump_h264_file_);
    dump_h264_file_ = nullptr;
  }
  if (dump_mjpg_file_ != nullptr) {
    std::fclose(dump_mjpg_file_);
    dump_mjpg_file_ = nullptr;
  }
}

void PipelineApp::DumpMjpgPacket(const std::uint8_t* data, std::size_t size) {
  if (dump_mjpg_file_ == nullptr || data == nullptr || size == 0) {
    return;
  }
  std::fwrite(data, 1, size, dump_mjpg_file_);
  std::fflush(dump_mjpg_file_);
}

void PipelineApp::DumpPacket(const std::vector<std::uint8_t>& packet) {
  if (dump_h264_file_ == nullptr || packet.empty()) {
    return;
  }
  std::fwrite(packet.data(), 1, packet.size(), dump_h264_file_);
  std::fflush(dump_h264_file_);
}

}  // namespace rk3576_demo
