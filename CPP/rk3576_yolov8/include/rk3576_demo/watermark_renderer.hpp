#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>

#include "rk3576_demo/media_types.hpp"

namespace rk3576_demo {

class WatermarkRenderer {
 public:
  WatermarkRenderer();

  WatermarkImage Render(std::uint64_t pts_ms) const;
  WatermarkImage RenderText(const std::string& text, int x = 24, int y = 24) const;

 private:
  using Glyph = std::array<std::uint8_t, 7>;

  void DrawGlyph(std::uint8_t* rgba, int image_width, int image_height, int x, int y, char c) const;
  void SetPixel(std::uint8_t* rgba, int image_width, int image_height, int x, int y,
                std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) const;

  std::unordered_map<char, Glyph> glyphs_;
};

}  // namespace rk3576_demo
