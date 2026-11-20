#pragma once

#include "engine/common.hpp"
#include "engine/image.hpp"

#include <cstdint>
#include <memory>
#include <rk_mpi.h>

extern "C"
{
#include <libavutil/avutil.h>
}

namespace nvr
{
// Decoded frame ownership.
struct Frame
{
    MppFrame value = nullptr;
    uint64_t sequence = 0;
    int64_t pts_us = AV_NOPTS_VALUE;
    uint64_t cycle = 0;

    explicit Frame(MppFrame frame);
    ~Frame();

    Nv12ImageView image() const;

    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;
};

using FramePtr = std::shared_ptr<Frame>;
} // namespace nvr
