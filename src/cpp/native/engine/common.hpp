#pragma once

#include <chrono>
#include <stdexcept>
#include <string>

namespace nvr
{
using Clock = std::chrono::steady_clock;

inline double milliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline void require(bool ok, const std::string &message)
{
    if (!ok)
    {
        throw std::runtime_error(message);
    }
}
} // namespace nvr
