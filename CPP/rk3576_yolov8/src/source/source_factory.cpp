#include "rk3576_yolo_demo/source/source_factory.hpp"

#include "rk3576_yolo_demo/source/rtsp_source.hpp"
#include "rk3576_yolo_demo/source/v4l2_source_adapter.hpp"

namespace rk3576_yolo_demo {

std::unique_ptr<IInputSource> CreateInputSource(const AppConfigV2& config) {
  switch (config.source) {
    case SourceKind::kV4L2:
      return std::unique_ptr<IInputSource>(new V4L2SourceAdapter(config));
    case SourceKind::kRtsp:
    case SourceKind::kLocalVideo:
      return std::unique_ptr<IInputSource>(new RtspSource(config));
  }
  return nullptr;
}

}  // namespace rk3576_yolo_demo
