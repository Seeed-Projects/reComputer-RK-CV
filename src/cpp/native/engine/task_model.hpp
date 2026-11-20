#pragma once

#include "engine/geometry.hpp"
#include "engine/postprocess.hpp"
#include "engine/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nvr
{
enum class ModelTask
{
    Detect,
    Segment,
    Pose,
    Obb,
    Classify,
    Semantic,
    Depth
};

ModelTask model_task(const std::string &name);
const char *task_name(ModelTask task);

enum class DenseResize
{
    None,
    HalfPixel,
    AlignCorners
};

struct ModelOptions
{
    ModelTask task = ModelTask::Detect;
    int classes = 0;
    int keypoints = 17, keypoint_dimensions = 3;
    bool normalized = false, end_to_end = false;
    bool center_crop = false;
    bool apply_nms = true;
    DenseResize dense_resize = DenseResize::None;
};

struct TensorSpec
{
    std::string name;
    std::vector<int> dimensions;
    bool channels_first = true;

    size_t elements() const;
};

// Row-major runs alternate value and count. box locates the raster in source pixels.
// Depth values use offset + value * scale; semantic values are exact class IDs.
struct ResultRaster
{
    int width = 0, height = 0;
    std::array<float, 4> box{};
    std::vector<uint32_t> runs;
    float offset = 0, scale = 1;
};

struct TaskDetection
{
    Detection detection;
    std::vector<std::array<float, 3>> keypoints;
    std::vector<std::array<float, 2>> polygon;
    std::optional<ResultRaster> mask;
};

struct ClassPrediction
{
    int class_id = 0;
    float confidence = 0;
};

struct TaskResult
{
    std::vector<TaskDetection> detections;
    std::vector<ClassPrediction> classification;
    std::optional<ResultRaster> dense;
};

// Tensor contracts are validated once, and shared by model inspection and inference.
class TaskPostprocessor
{
public:
    enum class Layout
    {
        Decoded,
        EndToEnd,
        Split,
        Dfl,
        Classification,
        Dense
    };

private:
    ModelOptions options_;
    std::vector<TensorSpec> specs_;
    int input_width_, input_height_;
    Layout layout_ = Layout::Decoded;
    int prediction_ = 0, boxes_ = -1, scores_ = -1, prototypes_ = -1, auxiliary_ = -1;
    int features_ = 0, proposals_ = 0, mask_channels_ = 0;
    bool feature_major_ = true, box_major_ = true, score_major_ = true;
    std::vector<int> regressions_, class_scores_, extras_;

    static void append_run(ResultRaster &raster, uint32_t value);

public:
    TaskPostprocessor(const ModelOptions &options, const std::vector<TensorSpec> &specs,
                      int input_width, int input_height);

    Layout layout() const
    {
        return layout_;
    }

    int classes() const
    {
        return options_.classes;
    }

    int proposals() const
    {
        return proposals_;
    }

    const char *description() const;

    // Dense outputs can be decoded directly from native FP16 or quantized storage.
    TaskResult decode_dense(const Geometry &geometry, const TensorView &output) const;

    TaskResult decode(const Geometry &geometry, const std::vector<std::vector<float>> &outputs,
                      float confidence, float nms_iou) const;
};
} // namespace nvr
