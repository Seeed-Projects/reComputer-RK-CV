#include "rk3576_yolo_demo/source/v4l2_source_adapter.hpp"

namespace rk3576_yolo_demo {

V4L2SourceAdapter::V4L2SourceAdapter(const AppConfigV2& config) : config_(config) {
  legacy_config_.device = config.device;
  legacy_config_.camera_width = config.camera_width;
  legacy_config_.camera_height = config.camera_height;
  legacy_config_.fps = config.fps;
}

V4L2SourceAdapter::~V4L2SourceAdapter() {
  Close();
}

bool V4L2SourceAdapter::Open() {
  if (opened_) {
    return true;
  }
  if (!camera_.Open(legacy_config_)) {
    last_error_ = "Failed to open V4L2 device via legacy camera adapter";
    return false;
  }
  opened_ = true;
  return true;
}

void V4L2SourceAdapter::Close() {
  if (opened_) {
    camera_.Close();
    opened_ = false;
  }
}

SourceDescriptor V4L2SourceAdapter::Describe() const {
  SourceDescriptor descriptor;
  descriptor.kind = SourceKind::kV4L2;
  descriptor.name = "v4l2-local-input";
  descriptor.location = config_.device;
  descriptor.compressed_input = true;
  descriptor.raw_input = true;
  return descriptor;
}

}  // namespace rk3576_yolo_demo
