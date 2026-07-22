#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rk3576_yolo_demo/branch/stream_branch.hpp"
#include "rk3576_yolo_demo/common/runtime_types.hpp"

namespace rk3576_yolo_demo {

class FrameDispatcher {
 public:
  FrameDispatcher() = default;
  ~FrameDispatcher();

  void AddBranch(std::shared_ptr<StreamBranch> branch);
  void ClearBranches();
  bool Dispatch(const UnifiedFrame& frame);
  std::vector<BranchStats> GetBranchStats() const;
  std::string DumpStats() const;

 private:
  std::vector<std::shared_ptr<StreamBranch>> branches_;
};

}  // namespace rk3576_yolo_demo
