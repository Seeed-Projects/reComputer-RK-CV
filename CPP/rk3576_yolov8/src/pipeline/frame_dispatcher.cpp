#include "rk3576_yolo_demo/pipeline/frame_dispatcher.hpp"

#include <sstream>
#include <utility>

namespace rk3576_yolo_demo {

FrameDispatcher::~FrameDispatcher() {
  ClearBranches();
}

void FrameDispatcher::AddBranch(std::shared_ptr<StreamBranch> branch) {
  if (!branch) {
    return;
  }
  branches_.push_back(std::move(branch));
}

void FrameDispatcher::ClearBranches() {
  for (std::size_t i = 0; i < branches_.size(); ++i) {
    branches_[i]->Stop();
  }
  branches_.clear();
}

bool FrameDispatcher::Dispatch(const UnifiedFrame& frame) {
  bool submitted = false;
  for (std::size_t i = 0; i < branches_.size(); ++i) {
    submitted = branches_[i]->Submit(frame) || submitted;
  }
  return submitted;
}

std::vector<BranchStats> FrameDispatcher::GetBranchStats() const {
  std::vector<BranchStats> stats;
  stats.reserve(branches_.size());
  for (std::size_t i = 0; i < branches_.size(); ++i) {
    stats.push_back(branches_[i]->GetStats());
  }
  return stats;
}

std::string FrameDispatcher::DumpStats() const {
  std::ostringstream oss;
  const std::vector<BranchStats> stats = GetBranchStats();
  for (std::size_t i = 0; i < stats.size(); ++i) {
    const BranchStats& item = stats[i];
    if (i > 0) {
      oss << " | ";
    }
    oss << ToString(item.role) << ": submitted=" << item.submitted
        << " dropped=" << item.dropped
        << " processed=" << item.processed
        << " queue=" << item.queue_size;
  }
  return oss.str();
}

}  // namespace rk3576_yolo_demo
