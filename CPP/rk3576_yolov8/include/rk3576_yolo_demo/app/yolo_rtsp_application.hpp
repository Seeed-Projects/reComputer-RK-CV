#pragma once

#include <atomic>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rk3576_demo/app_config.hpp"
#include "rk3576_demo/pipeline_app.hpp"
#include "rk3576_yolo_demo/ai/yolov8_engine.hpp"
#include "rk3576_yolo_demo/app/app_config_v2.hpp"
#include "rk3576_yolo_demo/branch/branch_output.hpp"
#include "rk3576_yolo_demo/branch/stream_branch.hpp"
#include "rk3576_yolo_demo/common/thread_safe_queue.hpp"
#include "rk3576_yolo_demo/pipeline/frame_dispatcher.hpp"
#include "rk3576_yolo_demo/source/i_source.hpp"

namespace rk3576_yolo_demo {

class YoloRtspApplication {
 public:
  explicit YoloRtspApplication(AppConfigV2 config);
  ~YoloRtspApplication();

  bool Run();

 private:
  struct AiFrameTask {
    rk3576_demo::DecodedFrame frame;
    int owned_fd = -1;
    std::vector<std::uint8_t> owned_frame_data;

    ~AiFrameTask();
  };

  bool PrintStartupSummary() const;
  bool PrepareSource();
  bool PrepareModel();
  bool PrepareBranches();
  bool StartAiWorker();
  void StopAiWorker();
  bool RunCompatibilityBaseline();
  bool RunCompressedInputPipeline();
  std::string ResolveModelPath() const;
  rk3576_demo::AppConfig MakeLegacyConfig() const;
  std::vector<StreamProfile> BuildStreamProfiles() const;
  void OnCompatibilityDecodedFrame(const rk3576_demo::DecodedFrame& frame);
  void AiInferLoop();
  bool EnqueueAiFrame(const rk3576_demo::DecodedFrame& frame);
  bool TryGetMatchedDetection(std::uint64_t frame_pts_ms, DetectionFrame* detection) const;
  void CacheDetectionResult(const DetectionFrame& detection);
  void CleanupDetectionCacheLocked(std::uint64_t reference_pts_ms);
  std::uint64_t DetectionMatchToleranceMs() const;
  std::uint64_t DetectionCacheMaxAgeMs() const;
  UnifiedFrame ConvertDecodedFrame(const rk3576_demo::DecodedFrame& frame) const;
  void HandleDispatchedFrame(const UnifiedFrame& frame, const StreamProfile& profile);

  AppConfigV2 config_;
  std::unique_ptr<IInputSource> source_;
  Yolov8Engine yolov8_engine_;
  FrameDispatcher dispatcher_;
  std::vector<std::shared_ptr<StreamBranch>> branches_;
  std::vector<std::shared_ptr<BranchOutput>> branch_outputs_;
  ThreadSafeQueue<std::shared_ptr<AiFrameTask>> ai_task_queue_{2};
  std::thread ai_worker_;
  std::atomic<bool> ai_worker_running_{false};
  mutable std::mutex detection_cache_mutex_;
  std::unordered_map<std::uint64_t, DetectionFrame> detection_cache_;
  std::uint64_t dispatched_frame_count_ = 0;
  std::uint64_t ai_infer_count_ = 0;
  std::uint64_t ai_infer_fail_count_ = 0;
  std::uint64_t ai_queue_drop_count_ = 0;
  std::uint64_t detection_match_hit_count_ = 0;
  std::uint64_t detection_match_miss_count_ = 0;
};

}  // namespace rk3576_yolo_demo
