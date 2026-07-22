#pragma once

#include <cstdint>
#include <string>

#include "rk3576_demo/app_config.hpp"

struct mk_media_t;
using mk_media = void*;

namespace rk3576_demo {

class RtspServer {
 public:
  RtspServer() = default;
  ~RtspServer();

  bool Start(const AppConfig& config, int width, int height);
  bool PushH264(const std::uint8_t* data, std::size_t size, std::uint64_t pts_ms);
  void Stop();
  std::string RtspUrl() const;

 private:
  mk_media media_ = nullptr;
  int port_ = 0;
  std::string app_;
  std::string stream_;
};

}  // namespace rk3576_demo
