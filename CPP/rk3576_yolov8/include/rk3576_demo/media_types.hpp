#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rk3576_demo {

enum class InputCodec {
  kMjpeg,
  kH264,
  kH265,
};

struct DecodedFrame {
  int width = 0;
  int height = 0;
  int hor_stride = 0;
  int ver_stride = 0;
  int format = 0;
  int fd = -1;
  void* virt_addr = nullptr;
  std::uint64_t pts_ms = 0;
  std::uint64_t decode_put_us = 0;
  std::uint64_t decode_wait_us = 0;
  std::uint64_t decode_total_us = 0;
};

struct WatermarkImage {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  std::string text;
  std::vector<std::uint8_t> rgba;
};

inline int AlignTo(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace rk3576_demo
