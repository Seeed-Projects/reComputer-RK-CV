#pragma once

#include <memory>

#include "rk3576_yolo_demo/app/app_config_v2.hpp"
#include "rk3576_yolo_demo/source/i_source.hpp"

namespace rk3576_yolo_demo {

std::unique_ptr<IInputSource> CreateInputSource(const AppConfigV2& config);

}  // namespace rk3576_yolo_demo
