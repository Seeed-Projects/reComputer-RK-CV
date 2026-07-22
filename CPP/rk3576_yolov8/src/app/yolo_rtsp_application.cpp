#include "rk3576_yolo_demo/app/yolo_rtsp_application.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unistd.h>

#include "rockchip/mpp_frame.h"
#include "rk3576_demo/mpp_decoder.hpp"
#include "rk3576_yolo_demo/source/source_factory.hpp"
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_yolo_demo {

namespace {

bool FileExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::ifstream input(path.c_str(), std::ios::binary);
  return input.good();
}

PixelFormat ToPixelFormat(int mpp_format) {
  switch (mpp_format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:
    case MPP_FMT_YUV420SP_VU:
    case MPP_FMT_YUV422SP:
    case MPP_FMT_YUV422SP_VU:
      return PixelFormat::kNv12;
    case MPP_FMT_RGB888:
      return PixelFormat::kRgb888;
    case MPP_FMT_ARGB8888:
    case MPP_FMT_ABGR8888:
    case MPP_FMT_BGRA8888:
    case MPP_FMT_RGBA8888:
      return PixelFormat::kRgba8888;
    default:
      break;
  }
  return PixelFormat::kUnknown;
}

std::size_t DecodedFrameBufferSizeBytes(const rk3576_demo::DecodedFrame& frame) {
  if (frame.hor_stride <= 0 || frame.ver_stride <= 0) {
    return 0;
  }
  const std::size_t plane = static_cast<std::size_t>(frame.hor_stride) *
                            static_cast<std::size_t>(frame.ver_stride);
  switch (frame.format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:
    case MPP_FMT_YUV420SP_VU:
      return plane * 3 / 2;
    case MPP_FMT_YUV422SP:
    case MPP_FMT_YUV422SP_VU:
      return plane * 2;
    case MPP_FMT_RGB888:
      return plane * 3;
    case MPP_FMT_ARGB8888:
    case MPP_FMT_ABGR8888:
    case MPP_FMT_BGRA8888:
    case MPP_FMT_RGBA8888:
      return plane * 4;
    default:
      return 0;
  }
}

}  // namespace

YoloRtspApplication::YoloRtspApplication(AppConfigV2 config) : config_(std::move(config)) {}

YoloRtspApplication::~YoloRtspApplication() {
  StopAiWorker();
}

YoloRtspApplication::AiFrameTask::~AiFrameTask() {
  if (owned_fd >= 0) {
    close(owned_fd);
    owned_fd = -1;
  }
}

bool YoloRtspApplication::Run() {
  if (!PrintStartupSummary()) {
    return false;
  }
  if (!PrepareSource()) {
    return false;
  }
  if (!PrepareModel()) {
    return false;
  }
  if (!PrepareBranches()) {
    return false;
  }
  if (!StartAiWorker()) {
    return false;
  }

  const bool ok = RunCompatibilityBaseline();
  StopAiWorker();
  for (std::size_t i = 0; i < branch_outputs_.size(); ++i) {
    branch_outputs_[i]->Stop();
  }
  branch_outputs_.clear();
  dispatcher_.ClearBranches();
  if (source_) {
    source_->Close();
  }
  return ok;
}

bool YoloRtspApplication::PrintStartupSummary() const {
  const std::string main_stream =
      config_.enable_main_stream ? (config_.rtsp_app + "/" + config_.rtsp_stream) : std::string("(disabled)");
  const std::string sub_stream =
      config_.enable_sub_stream ? (config_.rtsp_app + "/" + config_.sub_stream) : std::string("(disabled)");
  const std::string ai_stream =
      config_.enable_ai_stream
          ? (config_.rtsp_app + "/" + config_.ai_stream + " (" +
             std::to_string(config_.ai_width > 0 ? config_.ai_width : config_.camera_width) + "x" +
             std::to_string(config_.ai_height > 0 ? config_.ai_height : config_.camera_height) + ")")
          : std::string("(disabled)");
  RKLOG_INFO_ALWAYS("APP") << "Starting rk3576_yolov8tortsp_demo\n"
            << "  source      : " << ToString(config_.source) << "\n"
            << "  device/url  : " << config_.DisplayInputLocation() << "\n"
            << "  streams     : " << config_.EnabledStreamsSummary() << "\n"
            << "  main stream : " << main_stream << "\n"
            << "  sub stream  : " << sub_stream << "\n"
            << "  ai stream   : " << ai_stream << "\n"
            << "  ai input    : 640x640 fixed\n"
            << "  model dir   : ./model\n"
            << "  docs        : docs/project_layout.md, docs/class_interfaces.md, docs/thread_model.md\n";
  return true;
}

bool YoloRtspApplication::PrepareSource() {
  source_ = CreateInputSource(config_);
  if (!source_) {
    RKLOG_ERROR("APP") << "Failed to create input source\n";
    return false;
  }

  const SourceDescriptor descriptor = source_->Describe();
  RKLOG_INFO("APP") << "Prepared source adapter: " << source_->Name()
            << " location=" << descriptor.location
            << " compressed=" << (descriptor.compressed_input ? "yes" : "no")
            << " raw=" << (descriptor.raw_input ? "yes" : "no") << "\n";
  return true;
}

bool YoloRtspApplication::PrepareModel() {
  if (!config_.enable_ai_stream) {
    RKLOG_INFO("APP") << "AI stream disabled by --streams; skip loading RKNN model and async inference worker.\n";
    return true;
  }

  const std::string resolved_model_path = ResolveModelPath();
  if (resolved_model_path.empty()) {
    RKLOG_INFO("APP") << "YOLOv8 model is not configured yet and no default model is found under ./model.\n";
    return true;
  }

  yolov8_engine_.Configure(config_);
  if (!yolov8_engine_.LoadModel(resolved_model_path)) {
    RKLOG_ERROR("APP") << "Failed to load model: " << yolov8_engine_.LastError() << "\n";
    return false;
  }

  RKLOG_INFO("APP") << "Prepared YOLOv8 model: " << yolov8_engine_.model_path()
            << " input=" << yolov8_engine_.input_width() << "x" << yolov8_engine_.input_height()
            << " | " << yolov8_engine_.TensorSummary() << "\n";
  return true;
}

bool YoloRtspApplication::PrepareBranches() {
  const std::vector<StreamProfile> profiles = BuildStreamProfiles();
  dispatcher_.ClearBranches();
  branches_.clear();
  branch_outputs_.clear();

  std::ostringstream oss;
  for (std::size_t i = 0; i < profiles.size(); ++i) {
    std::shared_ptr<StreamBranch> branch(new StreamBranch(profiles[i], 4));
    if (!branch->Start([this](const UnifiedFrame& frame, const StreamProfile& profile) {
          HandleDispatchedFrame(frame, profile);
        })) {
      RKLOG_ERROR("APP") << "Failed to start stream branch: " << ToString(profiles[i].role) << "\n";
      dispatcher_.ClearBranches();
      branches_.clear();
      return false;
    }

    dispatcher_.AddBranch(branch);
    branches_.push_back(branch);
    branch_outputs_.push_back(std::shared_ptr<BranchOutput>(new BranchOutput(profiles[i], config_)));

    if (i > 0) {
      oss << ", ";
    }
    oss << ToString(profiles[i].role) << "=" << profiles[i].width << "x" << profiles[i].height
        << " osd=" << (profiles[i].enable_osd ? "yes" : "no")
        << " ai_overlay=" << (profiles[i].enable_ai_overlay ? "yes" : "no");
  }

  RKLOG_INFO("APP") << "Prepared stream branches: " << oss.str() << "\n";
  return true;
}

bool YoloRtspApplication::StartAiWorker() {
  if (!config_.enable_ai_stream || !yolov8_engine_.loaded()) {
    return true;
  }
  ai_worker_running_ = true;
  ai_worker_ = std::thread(&YoloRtspApplication::AiInferLoop, this);
  return true;
}

void YoloRtspApplication::StopAiWorker() {
  ai_worker_running_ = false;
  ai_task_queue_.Close();
  if (ai_worker_.joinable()) {
    ai_worker_.join();
  }
}

bool YoloRtspApplication::RunCompatibilityBaseline() {
  if (config_.source == SourceKind::kRtsp || config_.source == SourceKind::kLocalVideo) {
    return RunCompressedInputPipeline();
  }

  RKLOG_INFO("APP") << "Entering V2 pipeline mode. "
            << "PipelineApp now runs capture/decode only; main/sub/ai RTSP streams are generated by V2 BranchOutput.\n";

  rk3576_demo::AppConfig legacy = MakeLegacyConfig();
  legacy.decode_only = true;
  rk3576_demo::PipelineApp app(legacy);
  app.SetDecodedFrameObserver([this](const rk3576_demo::DecodedFrame& frame) {
    OnCompatibilityDecodedFrame(frame);
  });
  return app.Run();
}

bool YoloRtspApplication::RunCompressedInputPipeline() {
  if (!source_) {
    RKLOG_ERROR("APP") << "Compressed input pipeline requires a prepared input source\n";
    return false;
  }
  if (!source_->SupportsPacketRead()) {
    RKLOG_ERROR("APP") << "Selected source does not support packet read: " << source_->Name() << "\n";
    return false;
  }
  if (!source_->Open()) {
    RKLOG_ERROR("APP") << "Failed to open compressed input source: " << source_->LastError() << "\n";
    return false;
  }

  RKLOG_INFO("APP") << "Entering compressed input pipeline mode. "
            << "Packets are pulled by the source adapter and decoded by MPP before entering V2 branches.\n";

  rk3576_demo::MppDecoder decoder;
  if (!decoder.Open(source_->OutputCodec(), config_.fps,
                    source_->OutputWidth() > 0 ? source_->OutputWidth() : config_.camera_width,
                    source_->OutputHeight() > 0 ? source_->OutputHeight() : config_.camera_height,
                    [this](const rk3576_demo::DecodedFrame& frame) { OnCompatibilityDecodedFrame(frame); })) {
    RKLOG_ERROR("APP") << "Failed to initialize MPP decoder for RTSP source\n";
    source_->Close();
    return false;
  }

  std::uint64_t packet_count = 0;
  while (true) {
    CompressedPacket packet;
    if (!source_->ReadPacket(&packet)) {
      RKLOG_ERROR("APP") << "Compressed packet read failed: " << source_->LastError() << "\n";
      decoder.Close();
      source_->Close();
      return false;
    }
    if (packet.Empty()) {
      continue;
    }

    ++packet_count;
    if (packet_count <= 5 || (packet_count % 60) == 0) {
      RKLOG_INFO("APP") << "Input packet #" << packet_count
                << " codec="
                << (packet.codec == rk3576_demo::InputCodec::kH264
                        ? "H264"
                        : packet.codec == rk3576_demo::InputCodec::kH265 ? "H265" : "MJPEG")
                << " size=" << packet.Size()
                << " pts_ms=" << packet.pts_ms << "\n";
    }

    if (!decoder.Decode(packet.Data(), packet.Size(), packet.eos, packet.pts_ms)) {
      RKLOG_ERROR("APP") << "MPP decode failed for RTSP packet #" << packet_count << "\n";
      decoder.Close();
      source_->Close();
      return false;
    }

    if (config_.frame_limit > 0 && packet_count >= static_cast<std::uint64_t>(config_.frame_limit)) {
      break;
    }
  }

  decoder.Close();
  source_->Close();
  return true;
}

std::string YoloRtspApplication::ResolveModelPath() const {
  if (!config_.model_path.empty()) {
    return config_.model_path;
  }

  static const char* kCandidates[] = {
      "model/yolov8n_rk3576.rknn",
      "model/yolov8s_rk3576.rknn",
      "model/yolov8m_rk3576.rknn",
  };
  for (std::size_t i = 0; i < sizeof(kCandidates) / sizeof(kCandidates[0]); ++i) {
    if (FileExists(kCandidates[i])) {
      return kCandidates[i];
    }
  }
  return std::string();
}

rk3576_demo::AppConfig YoloRtspApplication::MakeLegacyConfig() const {
  rk3576_demo::AppConfig legacy;
  legacy.device = config_.device;
  legacy.camera_width = config_.camera_width;
  legacy.camera_height = config_.camera_height;
  legacy.output_width = config_.output_width;
  legacy.output_height = config_.output_height;
  legacy.fps = config_.fps;
  legacy.bitrate = config_.bitrate;
  legacy.rtsp_port = config_.rtsp_port;
  legacy.rtsp_app = config_.rtsp_app;
  legacy.rtsp_stream = config_.rtsp_stream;
  legacy.perf_log_interval_ms = config_.perf_log_interval_ms;
  legacy.frame_limit = config_.frame_limit;
  legacy.dump_mjpg = config_.dump_mjpg;
  legacy.dump_h264 = config_.dump_h264;
  legacy.dump_mjpg_path = config_.dump_mjpg_path;
  legacy.dump_h264_path = config_.dump_h264_path;
  return legacy;
}

std::vector<StreamProfile> YoloRtspApplication::BuildStreamProfiles() const {
  std::vector<StreamProfile> profiles;

  if (config_.enable_main_stream) {
    StreamProfile main_profile;
    main_profile.role = StreamRole::kMain;
    main_profile.app_name = config_.rtsp_app;
    main_profile.stream_name = config_.rtsp_stream;
    main_profile.width = config_.output_width;
    main_profile.height = config_.output_height;
    main_profile.bitrate = config_.bitrate;
    main_profile.enable_osd = true;
    main_profile.resize_mode = ResizeMode::kLetterbox;
    profiles.push_back(main_profile);
  }

  if (config_.enable_sub_stream) {
    StreamProfile sub_profile;
    sub_profile.role = StreamRole::kSub;
    sub_profile.app_name = config_.rtsp_app;
    sub_profile.stream_name = config_.sub_stream;
    sub_profile.width = config_.sub_width;
    sub_profile.height = config_.sub_height;
    sub_profile.bitrate = config_.bitrate / 2;
    sub_profile.enable_osd = false;
    sub_profile.resize_mode = ResizeMode::kCenterCrop;
    profiles.push_back(sub_profile);
  }

  if (config_.enable_ai_stream) {
    StreamProfile ai_profile;
    ai_profile.role = StreamRole::kAiDebug;
    ai_profile.app_name = config_.rtsp_app;
    ai_profile.stream_name = config_.ai_stream;
    ai_profile.width = config_.ai_width > 0 ? config_.ai_width : config_.camera_width;
    ai_profile.height = config_.ai_height > 0 ? config_.ai_height : config_.camera_height;
    ai_profile.bitrate = config_.bitrate;
    ai_profile.enable_osd = true;
    ai_profile.enable_ai_overlay = true;
    ai_profile.resize_mode = ResizeMode::kLetterbox;
    profiles.push_back(ai_profile);
  }

  return profiles;
}

void YoloRtspApplication::OnCompatibilityDecodedFrame(const rk3576_demo::DecodedFrame& frame) {
  if (config_.enable_ai_stream && yolov8_engine_.loaded() && !EnqueueAiFrame(frame)) {
    ++ai_infer_fail_count_;
    if (ai_infer_fail_count_ <= 5) {
      RKLOG_ERROR("APP") << "[AI] enqueue failed for async inference\n";
    }
  }

  for (std::size_t i = 0; i < branch_outputs_.size(); ++i) {
    DetectionFrame detection;
    const DetectionFrame* detection_ptr = nullptr;
    if (branch_outputs_[i]->role() == StreamRole::kAiDebug) {
      if (TryGetMatchedDetection(frame.pts_ms, &detection)) {
        detection_ptr = &detection;
        ++detection_match_hit_count_;
      } else {
        ++detection_match_miss_count_;
        if (detection_match_miss_count_ <= 5 || (detection_match_miss_count_ % 120) == 0) {
          std::size_t cache_size = 0;
          {
            std::lock_guard<std::mutex> lock(detection_cache_mutex_);
            cache_size = detection_cache_.size();
          }
          RKLOG_INFO("APP") << "[AI-MATCH] miss frame_pts_ms=" << frame.pts_ms
                    << " tolerance_ms=" << DetectionMatchToleranceMs()
                    << " cache_size=" << cache_size << "\n";
        }
      }
    }
    if (!branch_outputs_[i]->ProcessFrame(frame, detection_ptr)) {
      RKLOG_ERROR("APP") << "[V2][" << ToString(branch_outputs_[i]->role())
                << "] failed to generate branch stream " << branch_outputs_[i]->stream_name() << "\n";
    }
  }

  const UnifiedFrame unified = ConvertDecodedFrame(frame);
  if (!dispatcher_.Dispatch(unified)) {
    RKLOG_ERROR("APP") << "Failed to dispatch decoded frame into V2 branch workers\n";
    return;
  }

  ++dispatched_frame_count_;
  if (dispatched_frame_count_ <= 5 || (dispatched_frame_count_ % 60) == 0) {
    RKLOG_INFO("APP") << "[V2] mirrored decoded frame #" << dispatched_frame_count_
              << " pts_ms=" << unified.pts_ms
              << " size=" << unified.width << "x" << unified.height
              << " fmt=" << ToString(unified.format)
              << " storage=" << ToString(unified.storage)
              << " | branch_stats: " << dispatcher_.DumpStats() << "\n";

    const std::vector<BranchStats> stats = dispatcher_.GetBranchStats();
    for (std::size_t i = 0; i < stats.size(); ++i) {
      RKLOG_INFO("APP") << "[V2-QUEUE][" << ToString(stats[i].role) << "]"
                << " submitted=" << stats[i].submitted
                << " processed=" << stats[i].processed
                << " queue=" << stats[i].queue_size
                << " drop=" << stats[i].dropped << "\n";
    }
  }

  if (config_.enable_ai_stream && (dispatched_frame_count_ % 60) == 0) {
    const std::uint64_t total = detection_match_hit_count_ + detection_match_miss_count_;
    const double hit_rate = total == 0 ? 0.0
                                       : (100.0 * static_cast<double>(detection_match_hit_count_) /
                                          static_cast<double>(total));
    std::size_t cache_size = 0;
    {
      std::lock_guard<std::mutex> lock(detection_cache_mutex_);
      cache_size = detection_cache_.size();
    }
    RKLOG_INFO("APP") << "[AI-MATCH-PERF]"
              << " hit=" << detection_match_hit_count_
              << " miss=" << detection_match_miss_count_
              << " hit_rate=" << std::fixed << std::setprecision(1) << hit_rate << "%"
              << " tolerance_ms=" << DetectionMatchToleranceMs()
              << " cache_size=" << cache_size << "\n";
  }
}

void YoloRtspApplication::AiInferLoop() {
  while (ai_worker_running_) {
    std::shared_ptr<AiFrameTask> task;
    if (!ai_task_queue_.WaitPop(&task)) {
      break;
    }
    if (!task) {
      continue;
    }

    DetectionFrame detection;
    if (yolov8_engine_.Infer(task->frame, &detection)) {
      CacheDetectionResult(detection);
      ++ai_infer_count_;
      if (ai_infer_count_ <= 5 || (ai_infer_count_ % 30) == 0) {
        RKLOG_INFO("APP") << "[AI] frame #" << ai_infer_count_
                  << " input=" << detection.input_width << "x" << detection.input_height
                  << " pts_ms=" << detection.pts_ms
                  << " preprocess=" << detection.preprocess_us << "us"
                  << " npu=" << detection.npu_us << "us"
                  << " boxes=" << detection.boxes.size();
        if (!detection.boxes.empty()) {
          const DetectionBox& box = detection.boxes.front();
          RKLOG_INFO("APP") << " first=" << box.class_name << "@(" << box.x << "," << box.y
                    << "," << box.width << "," << box.height << ") score=" << box.score;
        }
        RKLOG_INFO("APP") << "\n";
      }
    } else {
      ++ai_infer_fail_count_;
      if (ai_infer_fail_count_ <= 5) {
        RKLOG_ERROR("APP") << "[AI] inference failed: " << yolov8_engine_.LastError() << "\n";
      }
    }
  }
}

bool YoloRtspApplication::EnqueueAiFrame(const rk3576_demo::DecodedFrame& frame) {
  std::shared_ptr<AiFrameTask> task(new AiFrameTask());
  task->frame = frame;
  task->frame.virt_addr = nullptr;
  task->owned_fd = -1;
  if (frame.virt_addr != nullptr && frame.hor_stride > 0 && frame.ver_stride > 0) {
    const std::size_t frame_bytes = DecodedFrameBufferSizeBytes(frame);
    if (frame_bytes > 0) {
      task->owned_frame_data.resize(frame_bytes);
      std::memcpy(task->owned_frame_data.data(), frame.virt_addr, frame_bytes);
      task->frame.virt_addr = task->owned_frame_data.data();
    }
  }
  if (frame.fd >= 0) {
    task->owned_fd = dup(frame.fd);
    if (task->owned_fd < 0) {
      return false;
    }
    task->frame.fd = task->owned_fd;
  }

  bool dropped_oldest = false;
  if (!ai_task_queue_.Push(task, &dropped_oldest)) {
    return false;
  }
  if (dropped_oldest) {
    ++ai_queue_drop_count_;
    if (ai_queue_drop_count_ <= 5 || (ai_queue_drop_count_ % 60) == 0) {
      RKLOG_INFO("APP") << "[AI] async queue dropped oldest frame, drop_count=" << ai_queue_drop_count_ << "\n";
    }
  }
  return true;
}

bool YoloRtspApplication::TryGetMatchedDetection(std::uint64_t frame_pts_ms, DetectionFrame* detection) const {
  if (detection == nullptr) {
    return false;
  }

  std::lock_guard<std::mutex> lock(detection_cache_mutex_);
  if (detection_cache_.empty()) {
    return false;
  }

  const auto exact_it = detection_cache_.find(frame_pts_ms);
  if (exact_it != detection_cache_.end()) {
    *detection = exact_it->second;
    return true;
  }

  const std::uint64_t tolerance_ms = DetectionMatchToleranceMs();
  std::uint64_t best_delta_ms = std::numeric_limits<std::uint64_t>::max();
  const DetectionFrame* best_match = nullptr;
  for (const auto& item : detection_cache_) {
    const std::uint64_t delta_ms =
        item.first > frame_pts_ms ? (item.first - frame_pts_ms) : (frame_pts_ms - item.first);
    if (delta_ms > tolerance_ms) {
      continue;
    }
    if (delta_ms < best_delta_ms) {
      best_delta_ms = delta_ms;
      best_match = &item.second;
    }
  }

  if (best_match == nullptr) {
    return false;
  }

  *detection = *best_match;
  return true;
}

void YoloRtspApplication::CacheDetectionResult(const DetectionFrame& detection) {
  std::lock_guard<std::mutex> lock(detection_cache_mutex_);
  detection_cache_[detection.pts_ms] = detection;
  CleanupDetectionCacheLocked(detection.pts_ms);
}

void YoloRtspApplication::CleanupDetectionCacheLocked(std::uint64_t reference_pts_ms) {
  const std::uint64_t max_age_ms = DetectionCacheMaxAgeMs();
  for (auto it = detection_cache_.begin(); it != detection_cache_.end();) {
    const std::uint64_t age_ms =
        reference_pts_ms > it->first ? (reference_pts_ms - it->first) : (it->first - reference_pts_ms);
    if (age_ms > max_age_ms) {
      it = detection_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

std::uint64_t YoloRtspApplication::DetectionMatchToleranceMs() const {
  const int fps = config_.fps > 0 ? config_.fps : 30;
  const std::uint64_t frame_interval_ms = std::max<std::uint64_t>(1, 1000ULL / static_cast<std::uint64_t>(fps));
  return frame_interval_ms * 3;
}

std::uint64_t YoloRtspApplication::DetectionCacheMaxAgeMs() const {
  const std::uint64_t tolerance_ms = DetectionMatchToleranceMs();
  return std::max<std::uint64_t>(tolerance_ms * 4, 500);
}

UnifiedFrame YoloRtspApplication::ConvertDecodedFrame(const rk3576_demo::DecodedFrame& frame) const {
  UnifiedFrame unified;
  unified.frame_id = dispatched_frame_count_ + 1;
  unified.pts_ms = frame.pts_ms;
  unified.width = frame.width;
  unified.height = frame.height;
  unified.hor_stride = frame.hor_stride;
  unified.ver_stride = frame.ver_stride;
  unified.format = ToPixelFormat(frame.format);
  unified.native_format = frame.format;
  unified.storage = frame.fd >= 0 ? FrameStorageType::kDmabuf : FrameStorageType::kVirtualMemory;
  unified.compressed = false;
  unified.key_frame = false;
  unified.dma_fd = frame.fd;
  unified.data = frame.virt_addr;
  unified.decode_us = frame.decode_total_us;
  unified.source = source_ ? source_->Describe() : SourceDescriptor();
  return unified;
}

void YoloRtspApplication::HandleDispatchedFrame(const UnifiedFrame& frame, const StreamProfile& profile) {
  const bool should_log = frame.frame_id <= 3 || (frame.frame_id % 90) == 0;
  if (!should_log) {
    return;
  }

  RKLOG_INFO("APP") << "[V2][" << ToString(profile.role) << "] frame_id=" << frame.frame_id
            << " pts_ms=" << frame.pts_ms
            << " src=" << frame.width << "x" << frame.height
            << " dst=" << profile.width << "x" << profile.height
            << " fmt=" << ToString(frame.format)
            << " ai_overlay=" << (profile.enable_ai_overlay ? "yes" : "no") << "\n";
}

}  // namespace rk3576_yolo_demo
