#pragma once

#include <cstddef>

namespace nvr
{
// Owns one uncached DMA-BUF allocation and its CPU mapping.
class DmaBuffer
{
public:
    int fd = -1;
    void *data = nullptr;
    size_t size = 0;

    explicit DmaBuffer(size_t bytes);
    ~DmaBuffer();
    DmaBuffer(const DmaBuffer &) = delete;
    DmaBuffer &operator=(const DmaBuffer &) = delete;
};
} // namespace nvr
