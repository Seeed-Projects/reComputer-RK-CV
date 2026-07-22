#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "rk3576_demo/app_config.hpp"

namespace rk3576_demo {

class V4L2Camera {
 public:
  using FrameHandler = std::function<bool(const std::uint8_t* data, std::size_t size, std::uint64_t pts_ms)>;

  V4L2Camera() = default;
  ~V4L2Camera();

  bool Open(const AppConfig& config);
  bool CaptureLoop(const FrameHandler& handler, int frame_limit);
  void Close();

 private:
  struct Buffer {
    void* start = nullptr;
    std::size_t length = 0;
  };

  bool InitDevice(const AppConfig& config);
  bool InitMmap(std::uint32_t buffer_count);
  bool StartStreaming();
  void StopStreaming();

  int fd_ = -1;
  bool streaming_ = false;
  std::string device_;
  std::vector<Buffer> buffers_;
};

}  // namespace rk3576_demo
