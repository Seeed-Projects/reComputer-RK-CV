#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "rk3576_yolo_demo/common/thread_safe_queue.hpp"
#include "rk3576_yolo_demo/common/runtime_types.hpp"

namespace rk3576_yolo_demo {

class StreamBranch {
 public:
  using FrameHandler = std::function<void(const UnifiedFrame&, const StreamProfile&)>;

  explicit StreamBranch(StreamProfile profile, std::size_t queue_capacity = 4);
  ~StreamBranch();

  bool Start(const FrameHandler& handler);
  void Stop();
  bool Submit(const UnifiedFrame& frame);

  const StreamProfile& profile() const { return profile_; }
  BranchStats GetStats() const;

 private:
  void WorkerLoop();

  StreamProfile profile_;
  ThreadSafeQueue<UnifiedFrame> queue_;
  FrameHandler handler_;
  std::unique_ptr<std::thread> worker_;
  mutable std::mutex stats_mutex_;
  BranchStats stats_;
  bool running_ = false;
};

}  // namespace rk3576_yolo_demo
