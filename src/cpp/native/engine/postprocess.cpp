#include "engine/postprocess.hpp"
#include "engine/common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace nvr
{
namespace
{
struct Box
{
    float x, y, w, h, score;
    int cls;
};

// Scan contiguous proposal blocks while preserving class order and threshold ties.
template <typename Score, typename Consume>
void select_classes(int proposals, int classes, bool feature_major, float threshold, Score score_at,
                    Consume consume)
{
    constexpr int block_size = 128;
    const int step = feature_major ? block_size : 1;
    for (int base = 0; base < proposals; base += step)
    {
        const int count = std::min(step, proposals - base);
        float best_scores[block_size];
        int best_classes[block_size];
        std::fill_n(best_scores, count, threshold);
        std::fill_n(best_classes, count, -1);
        for (int cls = 0; cls < classes; ++cls)
        {
            for (int j = 0; j < count; ++j)
            {
                const float score = score_at(base + j, cls);
                if (std::isfinite(score) && score > best_scores[j])
                {
                    best_scores[j] = score;
                    best_classes[j] = cls;
                }
            }
        }
        for (int j = 0; j < count; ++j)
        {
            if (best_classes[j] >= 0)
            {
                consume(base + j, best_classes[j], best_scores[j]);
            }
        }
    }
}

// Affine INT8 DFL heads provide boxes, class scores, and confidence sums per grid.
void decode_dfl(const YoloLayout &layout, const std::vector<std::vector<int8_t>> &quantized,
                float threshold, std::vector<Box> &candidates)
{
    for (int branch = 0; branch < 3; ++branch)
    {
        const auto &grid = layout.grids[size_t(branch)];
        const auto &r = quantized[size_t(branch) * 3];
        const auto &s = quantized[size_t(branch) * 3 + 1];
        const auto &sums_data = quantized[size_t(branch) * 3 + 2];
        int gh = grid.height, gw = grid.width, cells = gh * gw;
        int stride = layout.input_width / gw;
        for (int p = 0; p < cells; ++p)
        {
            // The model-zoo sum head bounds the maximum class probability. Leave
            // one quantization step of margin to avoid filtering borderline hits.
            if (float(int(sums_data[size_t(p)]) - grid.sum_zero_point) * grid.sum_scale +
                    grid.sum_scale <
                threshold)
            {
                continue;
            }
            int best = -128, cls = -1;
            for (int c = 0; c < layout.classes; ++c)
            {
                int score = s[size_t(c) * cells + p];
                if (score > best)
                {
                    best = score;
                    cls = c;
                }
            }
            float probability = float(best - grid.score_zero_point) * grid.score_scale;
            if (cls < 0 || probability < threshold)
            {
                continue;
            }
            float distances[4]{};
            for (int side = 0; side < 4; ++side)
            {
                float logits[16], max_logit = -1e30f;
                for (int bin = 0; bin < 16; ++bin)
                {
                    float x =
                        float(int(r[size_t(side * 16 + bin) * cells + p]) - grid.box_zero_point) *
                        grid.box_scale;
                    logits[bin] = x;
                    max_logit = std::max(max_logit, x);
                }
                float total = 0, weighted = 0;
                for (int bin = 0; bin < 16; ++bin)
                {
                    float weight = std::exp(logits[bin] - max_logit);
                    total += weight;
                    weighted += float(bin) * weight;
                }
                distances[side] = weighted / total * float(stride);
            }
            float cx = (float(p % gw) + 0.5f) * float(stride),
                  cy = (float(p / gw) + 0.5f) * float(stride);
            candidates.push_back({cx - distances[0], cy - distances[1], distances[0] + distances[2],
                                  distances[1] + distances[3], probability, cls});
        }
    }
}

// The split one-to-one head supplies probabilities and pixel-space xyxy boxes.
void decode_split(const YoloLayout &layout, const std::vector<float> &output, float threshold,
                  std::vector<Box> &candidates)
{
    const auto *boxes = output.data() + size_t(layout.classes) * size_t(layout.proposals);
    select_classes(
        layout.proposals, layout.classes, layout.scores_first, threshold,
        [&](int proposal, int cls)
        {
            return output[layout.scores_first ? size_t(cls) * layout.proposals + proposal
                                              : size_t(proposal) * layout.classes + cls];
        },
        [&](int proposal, int cls, float score)
        {
            auto corner = [&](int coordinate)
            {
                return boxes[layout.boxes_first ? size_t(coordinate) * layout.proposals + proposal
                                                : size_t(proposal) * 4 + coordinate];
            };
            float x = corner(0), y = corner(1), w = corner(2) - x, h = corner(3) - y;
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(w) && std::isfinite(h) &&
                w > 0 && h > 0)
            {
                candidates.push_back({x, y, w, h, score, cls});
            }
        });
}

// Decoded heads supply xywh boxes followed by class probabilities.
void decode_decoded(const YoloLayout &layout, const std::vector<float> &output, float threshold,
                    std::vector<Box> &candidates)
{
    auto value = [&](int proposal, int feature)
    {
        return output[layout.features_first ? size_t(feature) * layout.proposals + proposal
                                            : size_t(proposal) * (layout.classes + 4) + feature];
    };
    select_classes(
        layout.proposals, layout.classes, layout.features_first, threshold,
        [&](int proposal, int cls)
        {
            return value(proposal, cls + 4);
        },
        [&](int proposal, int cls, float score)
        {
            float w = value(proposal, 2), h = value(proposal, 3);
            float x = value(proposal, 0) - w / 2, y = value(proposal, 1) - h / 2;
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(w) && std::isfinite(h) &&
                w > 0 && h > 0)
            {
                candidates.push_back({x, y, w, h, score, cls});
            }
        });
}
} // namespace

YoloPostprocessor::YoloPostprocessor(const YoloLayout &layout, float confidence, float nms_iou)
    : layout_(layout), threshold_(confidence), iou_(nms_iou)
{
    require(std::isfinite(confidence) && confidence > 0 && confidence < 1 &&
                std::isfinite(nms_iou) && nms_iou > 0 && nms_iou < 1,
            "thresholds must be between 0 and 1");
    require(layout.classes > 0 && layout.classes <= 1020 && layout.proposals > 0,
            "invalid YOLO class or proposal count");
    if (layout.kind == YoloLayout::Kind::Dfl)
    {
        require(layout.input_width > 0 && layout.input_height > 0, "invalid DFL input size");
        size_t proposals = 0;
        for (const auto &grid : layout.grids)
        {
            require(grid.width > 0 && grid.height > 0 && layout.input_width % grid.width == 0 &&
                        layout.input_height % grid.height == 0 &&
                        layout.input_width / grid.width == layout.input_height / grid.height,
                    "invalid DFL grid stride");
            require(std::isfinite(grid.box_scale) && grid.box_scale > 0 &&
                        std::isfinite(grid.score_scale) && grid.score_scale > 0 &&
                        std::isfinite(grid.sum_scale) && grid.sum_scale > 0,
                    "invalid DFL quantization scales");
            proposals += size_t(grid.width) * size_t(grid.height);
        }
        require(proposals == size_t(layout.proposals), "DFL proposal count mismatch");
    }
}

std::vector<Detection>
YoloPostprocessor::decode(const Geometry &g, const std::vector<float> &output,
                          const std::vector<std::vector<int8_t>> &quantized, double *nms_ms,
                          std::vector<Detection> *unfiltered, bool candidates_only) const
{
    require(g.width > 0 && g.height > 0 && g.rw > 0 && g.rh > 0, "invalid detection geometry");
    if (layout_.kind == YoloLayout::Kind::Dfl)
    {
        require(quantized.size() == 9, "DFL needs nine output buffers");
        for (size_t branch = 0; branch < layout_.grids.size(); ++branch)
        {
            const auto &grid = layout_.grids[branch];
            const size_t cells = size_t(grid.width) * size_t(grid.height);
            require(quantized[branch * 3].size() == cells * 64 &&
                        quantized[branch * 3 + 1].size() == cells * size_t(layout_.classes) &&
                        quantized[branch * 3 + 2].size() == cells,
                    "DFL output buffer size mismatch");
        }
    }
    else
    {
        require(output.size() == size_t(layout_.classes + 4) * size_t(layout_.proposals),
                "YOLO output buffer size mismatch");
    }

    std::vector<Box> candidates;
    // Use the actual even resized dimensions to undo the RGA letterbox.
    const float sx = float(g.width) / float(g.rw), sy = float(g.height) / float(g.rh);
    auto source_detection = [&](const Box &box) -> std::optional<Detection>
    {
        float x1 = std::clamp((box.x - float(g.px)) * sx, 0.f, float(g.width));
        float y1 = std::clamp((box.y - float(g.py)) * sy, 0.f, float(g.height));
        float x2 = std::clamp((box.x + box.w - float(g.px)) * sx, 0.f, float(g.width));
        float y2 = std::clamp((box.y + box.h - float(g.py)) * sy, 0.f, float(g.height));
        if (x2 <= x1 || y2 <= y1) return std::nullopt;
        return Detection{box.cls, box.score, {x1, y1, x2, y2}};
    };
    switch (layout_.kind)
    {
    case YoloLayout::Kind::Dfl:
        decode_dfl(layout_, quantized, threshold_, candidates);
        break;
    case YoloLayout::Kind::ScoresAndBoxes:
        if (unfiltered)
        {
            candidates.reserve(layout_.proposals);
            decode_split(layout_, output, -std::numeric_limits<float>::infinity(), candidates);
            unfiltered->clear();
            unfiltered->reserve(candidates.size());
            for (const auto &box : candidates)
            {
                if (auto detection = source_detection(box))
                    unfiltered->push_back(*detection);
            }
            if (candidates_only)
                return {};
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                            [&](const Box &box) { return box.score <= threshold_; }),
                             candidates.end());
        }
        else
            decode_split(layout_, output, threshold_, candidates);
        break;
    case YoloLayout::Kind::Decoded:
        decode_decoded(layout_, output, threshold_, candidates);
        break;
    }

    // NMS policy is explicit: output count alone does not identify an NMS-free head.
    auto suppression_started = Clock::now();
    std::sort(candidates.begin(), candidates.end(),
              [](const Box &a, const Box &b)
              {
                  return a.score > b.score;
              });
    if (candidates.size() > 1000)
    {
        candidates.resize(1000);
    }
    std::vector<Box> kept;
    for (const auto &a : candidates)
    {
        bool suppressed = false;
        if (layout_.apply_nms)
        {
            for (const auto &b : kept)
            {
                if (a.cls != b.cls)
                {
                    continue;
                }
                float iw = std::max(0.f, std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x));
                float ih = std::max(0.f, std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
                float inter = iw * ih, uni = a.w * a.h + b.w * b.h - inter;
                if (uni > 0 && inter / uni > iou_)
                {
                    suppressed = true;
                    break;
                }
            }
        }
        if (suppressed)
        {
            continue;
        }
        kept.push_back(a);
        if (kept.size() >= 300)
        {
            break;
        }
    }
    if (nms_ms)
    {
        *nms_ms = layout_.apply_nms ? milliseconds(suppression_started) : 0;
    }
    std::vector<Detection> result;
    result.reserve(kept.size());
    for (const auto &a : kept)
    {
        if (auto detection = source_detection(a))
            result.push_back(*detection);
    }
    return result;
}
} // namespace nvr
