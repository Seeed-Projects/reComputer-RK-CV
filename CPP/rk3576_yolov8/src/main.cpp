#include <iostream>

#include "rk3576_yolo_demo/app/app_config_v2.hpp"
#include "rk3576_yolo_demo/app/yolo_rtsp_application.hpp"
#include "rk3576_yolo_demo/common/logger.hpp"

int main(int argc, char** argv) {
  rk3576_yolo_demo::AppConfigV2 config;
  if (!rk3576_yolo_demo::AppConfigV2::Parse(argc, argv, &config)) {
    return 1;
  }
  if (config.show_help) {
    return 0;
  }

  rk3576_yolo_demo::Logger::Instance().SetInfoEnabled(config.detail_info);

  rk3576_yolo_demo::YoloRtspApplication app(config);
  if (!app.Run()) {
    RKLOG_ERROR("APP") << "rk3576_yolov8tortsp_demo exited with failure\n";
    return 1;
  }

  return 0;
}
