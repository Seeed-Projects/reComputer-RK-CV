#pragma once

#include <cstddef>

namespace nvr
{
// Borrowed NV12 storage. The caller keeps the DMA-BUF alive through
// processing. Strides describe the allocated luma plane, including padding.
struct Nv12ImageView
{
    int width = 0, height = 0;
    int horizontal_stride = 0, vertical_stride = 0;
    int fd = -1;
    size_t size = 0;
};
} // namespace nvr
