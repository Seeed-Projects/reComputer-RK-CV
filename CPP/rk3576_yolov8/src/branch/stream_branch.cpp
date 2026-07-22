#include "rk3576_yolo_demo/branch/stream_branch.hpp"

#include <utility>

namespace rk3576_yolo_demo {

StreamBranch::StreamBranch(StreamProfile profile, std::size_t queue_capacity)
    : profile_(std::move(profile)), queue_(queue_capacity) {
  stats_.role = profile_.role;
}

StreamBranch::~StreamBranch() {
  Stop();
}

bool StreamBranch::Start(const FrameHandler& handler) {
  if (running_) {
    return true;
  }
  if (!handler) {
    return false;
  }

  handler_ = handler;
  running_ = true;
  worker_.reset(new std::thread(&StreamBranch::WorkerLoop, this));
  return true;
}

void StreamBranch::Stop() {
  if (!running_) {
    return;
  }

  queue_.Close();
  if (worker_ && worker_->joinable()) {
    worker_->join();
  }
  worker_.reset();
  running_ = false;
}

bool StreamBranch::Submit(const UnifiedFrame& frame) {
  bool dropped_oldest = false;
  if (!queue_.Push(frame, &dropped_oldest)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(stats_mutex_);
  ++stats_.submitted;
  if (dropped_oldest) {
    ++stats_.dropped;
  }
  stats_.queue_size = queue_.Size();
  return true;
}

BranchStats StreamBranch::GetStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  BranchStats stats = stats_;
  stats.queue_size = queue_.Size();
  return stats;
}

void StreamBranch::WorkerLoop() {
  UnifiedFrame frame;
  while (queue_.WaitPop(&frame)) {
    handler_(frame, profile_);
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++stats_.processed;
    stats_.queue_size = queue_.Size();
  }
}

}  // namespace rk3576_yolo_demo
