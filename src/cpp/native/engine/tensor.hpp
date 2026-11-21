#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace nvr
{
// Borrowed tensor storage. The caller retains the buffer through decoding.
// Quantized values are converted only when read, rather than expanding a whole image.
struct TensorView
{
    enum class Type
    {
        Float32,
        Float16,
        Int8,
        Uint8,
        Int32
    };

    const void *data = nullptr;
    size_t count = 0;
    Type type = Type::Float32;
    float scale = 1;
    int zero_point = 0;

    // Optional native channel blocking; indices presented to callers remain NCHW.
    size_t width = 0, height = 0, row_stride = 0, channel_block = 0;

    float operator[](size_t index) const
    {
        if (channel_block)
        {
            const size_t plane = width * height;
            const size_t channel = index / plane, pixel = index % plane;
            index =
                ((channel / channel_block * height + pixel / width) * row_stride + pixel % width) *
                    channel_block +
                channel % channel_block;
        }
        return scalar(index);
    }

    float spatial(size_t channel, size_t x, size_t y) const
    {
        const size_t index =
            ((channel / channel_block * height + y) * row_stride + x) * channel_block +
            channel % channel_block;
        return scalar(index);
    }

private:
    float scalar(size_t index) const
    {
        switch (type)
        {
        case Type::Float32:
        {
            float value;
            std::memcpy(&value, static_cast<const uint8_t *>(data) + index * 4, 4);
            return value;
        }
        case Type::Float16:
        {
#if defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
            __fp16 value;
            std::memcpy(&value, static_cast<const uint8_t *>(data) + index * 2, 2);
            return float(value);
#else
            uint16_t bits;
            std::memcpy(&bits, static_cast<const uint8_t *>(data) + index * 2, 2);
            const float sign = bits & 0x8000 ? -1.f : 1.f;
            const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
            if (exponent == 31)
            {
                return fraction ? std::numeric_limits<float>::quiet_NaN()
                                : sign * std::numeric_limits<float>::infinity();
            }
            return sign * std::ldexp(float(exponent ? fraction + 1024 : fraction),
                                     exponent ? exponent - 25 : -24);
#endif
        }
        case Type::Int8:
            return (float(static_cast<const int8_t *>(data)[index]) - float(zero_point)) * scale;
        case Type::Uint8:
            return (float(static_cast<const uint8_t *>(data)[index]) - float(zero_point)) * scale;
        case Type::Int32:
        {
            int32_t value;
            std::memcpy(&value, static_cast<const uint8_t *>(data) + index * 4, 4);
            return (float(value) - float(zero_point)) * scale;
        }
        }
        throw std::runtime_error("Invalid tensor scalar type");
    }
};
} // namespace nvr
