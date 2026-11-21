#pragma once

#include "engine/dma_buffer.hpp"
#include "engine/image.hpp"
#include "engine/geometry.hpp"

#include <array>
#include <map>
#include <memory>
#include <im2d.hpp>

namespace nvr
{
// One instance per processing thread. The destination must outlive this object.
// Clear source imports before replacing a source buffer pool whose FDs may be reused.
class Preprocessor
{
    std::map<int, rga_buffer_handle_t> sources_;
    rga_buffer_handle_t destination_ = 0;
    std::array<std::unique_ptr<DmaBuffer>, 2> scaled_;
    std::array<rga_buffer_handle_t, 2> scaled_handles_{};
    int width_, height_, stride_;
    DmaBuffer &input_;
    bool painted_ = false, center_crop_ = false;
    Geometry previous_;

    rga_buffer_t resize_buffer(size_t slot, int width, int height);

public:
    Preprocessor(DmaBuffer &input, int width, int height, int stride,
                 bool center_crop = false);
    ~Preprocessor();

    Preprocessor(const Preprocessor &) = delete;
    Preprocessor &operator=(const Preprocessor &) = delete;

    Geometry process(const Nv12ImageView &image);
    void clear_sources();
};
} // namespace nvr
