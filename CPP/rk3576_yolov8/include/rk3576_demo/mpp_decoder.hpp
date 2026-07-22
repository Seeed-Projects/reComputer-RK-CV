#pragma once

#include <functional>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rockchip/mpp_buffer.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/mpp_meta.h"
#include "rockchip/mpp_packet.h"
#include "rockchip/rk_mpi.h"

#include "rk3576_demo/media_types.hpp"

namespace rk3576_demo {

class MppDecoder {
 public:
  using FrameCallback = std::function<void(const DecodedFrame&)>;

  MppDecoder() = default;
  ~MppDecoder();

  bool Open(InputCodec codec, int fps, int width, int height, const FrameCallback& callback);
  bool Decode(const std::uint8_t* data, std::size_t size, bool eos, std::uint64_t pts_ms);
  void Reset();
  void Close();

 private:
  MppCodingType ToMppCodingType(InputCodec codec) const;
  bool SetupMjpegResources(std::size_t min_output_buffer_size);
  bool CreateInputPacket(const std::uint8_t* data, std::size_t size, bool eos,
                         std::uint64_t pts_ms, MppPacket* packet);
  void RecycleMjpegInputPacket(MppFrame frame, MppPacket* fallback_packet);

  MppCtx ctx_ = nullptr;
  MppApi* mpi_ = nullptr;
  MppBufferGroup frame_group_ = nullptr;
  MppBufferGroup mjpeg_input_group_ = nullptr;
  MppFrame mjpeg_output_frame_ = nullptr;
  FrameCallback callback_;
  InputCodec codec_ = InputCodec::kMjpeg;
  bool use_external_group_ = false;
  int fps_ = 30;
  int width_ = 0;
  int height_ = 0;
  std::size_t mjpeg_output_buffer_size_ = 0;
  std::uint64_t packet_count_ = 0;
  std::uint64_t frame_count_ = 0;
};

}  // namespace rk3576_demo
