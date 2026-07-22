#pragma once

#include <cstdint>
#include <vector>

#include "rockchip/mpp_buffer.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/rk_mpi.h"

#include "rk3576_demo/media_types.hpp"

namespace rk3576_demo {

class MppEncoder {
 public:
  MppEncoder() = default;
  ~MppEncoder();

  bool Open(int width, int height, int fps, int bitrate);
  bool GetHeader(std::vector<std::uint8_t>* packet);
  bool EncodeCurrentFrame(std::uint64_t pts_ms, std::vector<std::uint8_t>* packet);
  void Close();

  int width() const { return width_; }
  int height() const { return height_; }
  int hor_stride() const { return hor_stride_; }
  int ver_stride() const { return ver_stride_; }
  std::size_t frame_size() const { return frame_size_; }
  MppBuffer input_buffer() const { return frame_buffer_; }
  int input_fd() const;
  void* input_addr() const;

 private:
  bool SetupConfig(int fps, int bitrate);

  int width_ = 0;
  int height_ = 0;
  int hor_stride_ = 0;
  int ver_stride_ = 0;
  std::size_t frame_size_ = 0;
  std::size_t mdinfo_size_ = 0;

  MppCtx ctx_ = nullptr;
  MppApi* mpi_ = nullptr;
  MppEncCfg cfg_ = nullptr;
  MppBufferGroup buffer_group_ = nullptr;
  MppBuffer frame_buffer_ = nullptr;
  MppBuffer packet_buffer_ = nullptr;
  MppBuffer motion_buffer_ = nullptr;
};

}  // namespace rk3576_demo
