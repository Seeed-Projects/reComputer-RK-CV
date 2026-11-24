#pragma once

#include "engine/geometry.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace nvr
{
struct Detection
{
    int class_id = -1;
    float confidence = 0;
    std::array<float, 4> box{}; // Source-image coordinates: x1, y1, x2, y2.
};

// Describes one affine INT8 DFL branch without depending on an inference SDK.
struct DflGrid
{
    int width = 0, height = 0;
    int box_zero_point = 0, score_zero_point = 0, sum_zero_point = 0;
    float box_scale = 0, score_scale = 0, sum_scale = 0;
};

struct YoloLayout
{
    enum class Kind
    {
        Decoded,
        ScoresAndBoxes,
        Dfl
    };

    Kind kind = Kind::Decoded;
    int input_width = 0, input_height = 0;
    int classes = 0, proposals = 0;
    bool features_first = true, scores_first = true, boxes_first = true;
    std::array<DflGrid, 3> grids{};
    bool apply_nms = true;
};

// Stateless per-frame decoding. Outputs are borrowed only during decode().
// Float storage is decoded YOLO, or contiguous [scores][boxes] for split heads.
// DFL storage contains [boxes, scores, confidence sums] for each of three grids.
class YoloPostprocessor
{
    YoloLayout layout_;
    float threshold_, iou_;

public:
    YoloPostprocessor(const YoloLayout &layout, float confidence, float nms_iou);

    void configure(float confidence, float iou)
    {
        threshold_ = confidence;
        iou_ = iou;
    }

    std::vector<Detection> decode(const Geometry &geometry, const std::vector<float> &output,
                                  const std::vector<std::vector<int8_t>> &quantized = {},
                                  double *nms_ms = nullptr,
                                  std::vector<Detection> *unfiltered = nullptr,
                                  bool candidates_only = false) const;
};
} // namespace nvr
