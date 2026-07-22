#pragma once

#include <string>

#include "rk3576_demo/app_config.hpp"
#include "rk3576_demo/v4l2_camera.hpp"
#include "rk3576_yolo_demo/app/app_config_v2.hpp"
#include "rk3576_yolo_demo/source/i_source.hpp"

namespace rk3576_yolo_demo {

class V4L2SourceAdapter : public IInputSource {
 public:
  explicit V4L2SourceAdapter(const AppConfigV2& config);
  ~V4L2SourceAdapter() override;

  const char* Name() const override { return "V4L2SourceAdapter"; }
  bool Open() override;
  void Close() override;
  SourceDescriptor Describe() const override;
  std::string LastError() const override { return last_error_; }

 private:
  AppConfigV2 config_;
  rk3576_demo::AppConfig legacy_config_;
  rk3576_demo::V4L2Camera camera_;
  bool opened_ = false;
  std::string last_error_;
};

}  // namespace rk3576_yolo_demo
