#pragma once

#include "rk3576_demo/media_types.hpp"
#include "rk3576_demo/mpp_encoder.hpp"
#include "rk3576_yolo_demo/common/runtime_types.hpp"

namespace rk3576_demo {

class RgaProcessor {
 public:
  bool ProcessFrame(const DecodedFrame& frame, const WatermarkImage& watermark, MppEncoder* encoder,
                    rk3576_yolo_demo::ResizeMode resize_mode = rk3576_yolo_demo::ResizeMode::kStretch,
                    rk3576_yolo_demo::FrameTransformInfo* transform = nullptr);

 private:
  int ToRgaFormat(int mpp_format) const;
  bool BlendWatermarkOnNv12(const WatermarkImage& watermark, MppEncoder* encoder) const;
};

}  // namespace rk3576_demo
