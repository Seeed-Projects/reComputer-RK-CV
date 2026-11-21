#include "engine/task_model.hpp"
#include "engine/common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nvr
{
void TaskPostprocessor::append_run(ResultRaster &raster, uint32_t value)
{
    if (!raster.runs.empty() && raster.runs[raster.runs.size() - 2] == value)
    {
        ++raster.runs.back();
    }
    else
    {
        raster.runs.push_back(value);
        raster.runs.push_back(1);
    }
}

TaskResult TaskPostprocessor::decode_dense(const Geometry &g, const TensorView &output) const
{
    require(layout_ == Layout::Dense, "Model does not produce dense predictions");
    require(output.data && output.count == specs_[0].elements(), "Dense output size mismatch");
    require(g.width > 0 && g.height > 0 && g.rw > 0 && g.rh > 0, "Invalid result geometry");
    const float source_width = float(g.width), source_height = float(g.height);
    const float resize_width = float(g.rw), resize_height = float(g.rh);
    const float padding_x = float(g.px), padding_y = float(g.py);
    const float model_width = float(input_width_), model_height = float(input_height_);
    TaskResult result;
    const auto &spec = specs_[0];
    const auto &d = spec.dimensions;
    int channels = d.size() == 4 ? d[spec.channels_first ? 1 : 3] : 1;
    int height = d[d.size() == 3 ? 1 : spec.channels_first ? 2 : 1];
    int width = d[d.size() == 3 ? 2 : spec.channels_first ? 3 : 2];
    auto value_at = [&](int channel, int x, int y)
    {
        if (output.channel_block)
        {
            return output.spatial(size_t(channel), size_t(x), size_t(y));
        }
        return output[spec.channels_first || d.size() == 3
                          ? size_t(channel) * width * height + size_t(y) * width + x
                          : (size_t(y) * width + x) * channels + channel];
    };
    ResultRaster raster;
    raster.width = std::min(128, g.width);
    raster.height =
        std::max(1, int(std::round(float(raster.width) * source_height / source_width)));
    if (raster.height > 128)
    {
        raster.height = 128;
        raster.width =
            std::max(1, int(std::round(float(raster.height) * source_width / source_height)));
    }
    raster.box = {0, 0, float(source_width), float(source_height)};
    std::vector<float> samples;
    for (int y = 0; y < raster.height; ++y)
    {
        for (int x = 0; x < raster.width; ++x)
        {
            int px =
                std::clamp(int(((float(x) + .5f) * resize_width / float(raster.width) + padding_x) *
                               float(width) / model_width),
                           0, width - 1);
            int py = std::clamp(
                int(((float(y) + .5f) * resize_height / float(raster.height) + padding_y) *
                    float(height) / model_height),
                0, height - 1);
            int x0 = px, y0 = py, x1 = px, y1 = py;
            float dx = 0, dy = 0;
            if (options_.dense_resize != DenseResize::None)
            {
                // Sample the same model pixels as a full-resolution output, then
                // reconstruct only their logits/depth with the export's resize convention.
                const int mx = std::clamp(
                    int((float(x) + .5f) * resize_width / float(raster.width) + padding_x), 0,
                    input_width_ - 1);
                const int my = std::clamp(
                    int((float(y) + .5f) * resize_height / float(raster.height) + padding_y), 0,
                    input_height_ - 1);
                const bool corners = options_.dense_resize == DenseResize::AlignCorners;
                float fx = corners ? float(mx) * float(width - 1) / (model_width - 1)
                                   : (float(mx) + .5f) * float(width) / model_width - .5f;
                float fy = corners ? float(my) * float(height - 1) / (model_height - 1)
                                   : (float(my) + .5f) * float(height) / model_height - .5f;
                fx = std::clamp(fx, 0.f, float(width - 1));
                fy = std::clamp(fy, 0.f, float(height - 1));
                x0 = int(fx);
                y0 = int(fy);
                x1 = std::min(x0 + 1, width - 1);
                y1 = std::min(y0 + 1, height - 1);
                dx = fx - float(x0);
                dy = fy - float(y0);
            }
            auto sample = [&](int channel)
            {
                if (options_.dense_resize == DenseResize::None)
                {
                    return value_at(channel, px, py);
                }
                const float a = value_at(channel, x0, y0), b = value_at(channel, x1, y0);
                const float c = value_at(channel, x0, y1), d = value_at(channel, x1, y1);
                const float top = a + (b - a) * dx, bottom = c + (d - c) * dx;
                return top + (bottom - top) * dy;
            };
            if (options_.task == ModelTask::Depth)
            {
                samples.push_back(sample(0));
            }
            else
            {
                int label = 0;
                if (d.size() == 3)
                {
                    float value = value_at(0, px, py);
                    require(std::isfinite(value) && value >= 0 && value < float(options_.classes) &&
                                std::abs(value - std::round(value)) < .01f,
                            "Semantic output contains an invalid class ID");
                    label = int(std::round(value));
                }
                else if (channels == 1)
                {
                    label = 0;
                }
                else
                {
                    float best = -std::numeric_limits<float>::infinity();
                    for (int c = 0; c < channels; ++c)
                    {
                        float value = sample(c);
                        if (std::isfinite(value) && value > best)
                        {
                            best = value;
                            label = c;
                        }
                    }
                    require(std::isfinite(best), "Semantic output has no finite class score");
                }
                append_run(raster, uint32_t(label));
            }
        }
    }
    if (options_.task == ModelTask::Depth)
    {
        float minimum = std::numeric_limits<float>::infinity(), maximum = 0;
        for (float value : samples)
        {
            if (std::isfinite(value) && value > 0)
            {
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
        }
        require(std::isfinite(minimum), "Depth output has no valid positive samples");
        raster.offset = minimum;
        raster.scale = maximum > minimum ? (maximum - minimum) / 65535 : 0;
        for (float value : samples)
        {
            append_run(
                raster,
                std::isfinite(value) && value > 0
                    ? uint32_t(std::clamp(
                          std::round(raster.scale > 0 ? (value - minimum) / raster.scale : 0), 0.f,
                          65535.f))
                    : 65536);
        }
    }
    result.dense = std::move(raster);
    return result;
}
} // namespace nvr
