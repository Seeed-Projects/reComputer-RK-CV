#include "engine/task_model.hpp"
#include "engine/common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nvr
{
ModelTask model_task(const std::string &name)
{
    if (name == "detect")
    {
        return ModelTask::Detect;
    }
    if (name == "segment")
    {
        return ModelTask::Segment;
    }
    if (name == "pose")
    {
        return ModelTask::Pose;
    }
    if (name == "obb")
    {
        return ModelTask::Obb;
    }
    if (name == "classify")
    {
        return ModelTask::Classify;
    }
    if (name == "semantic")
    {
        return ModelTask::Semantic;
    }
    if (name == "depth")
    {
        return ModelTask::Depth;
    }
    throw std::runtime_error("Unsupported model task: " + name);
}

const char *task_name(ModelTask task)
{
    switch (task)
    {
    case ModelTask::Detect:
        return "detect";
    case ModelTask::Segment:
        return "segment";
    case ModelTask::Pose:
        return "pose";
    case ModelTask::Obb:
        return "obb";
    case ModelTask::Classify:
        return "classify";
    case ModelTask::Semantic:
        return "semantic";
    case ModelTask::Depth:
        return "depth";
    }
    throw std::runtime_error("Invalid model task");
}

size_t TensorSpec::elements() const
{
    require(!dimensions.empty() && dimensions.size() <= 4, "Unsupported tensor rank");
    size_t count = 1;
    for (int dimension : dimensions)
    {
        require(dimension > 0 && count <= 268435456 / size_t(dimension),
                "Invalid or excessive tensor dimensions");
        count *= size_t(dimension);
    }
    return count;
}

TaskPostprocessor::TaskPostprocessor(const ModelOptions &options,
                                     const std::vector<TensorSpec> &specs, int width, int height)
    : options_(options), specs_(specs), input_width_(width), input_height_(height)
{
    require(width > 0 && height > 0 && !specs.empty() && specs.size() <= 16,
            "Invalid model input or output count");
    require(options.classes >= 0 && options.classes <= 65535, "Invalid metadata class count");
    require(!options.center_crop || options.task == ModelTask::Classify,
            "Center-crop preprocessing is only supported for classification");
    require(options.dense_resize == DenseResize::None || options.task == ModelTask::Semantic ||
                options.task == ModelTask::Depth,
            "Dense interpolation requires a dense task");
    for (const auto &spec : specs)
    {
        spec.elements();
        require(spec.dimensions.size() < 3 || spec.dimensions[0] == 1,
                "Only batch-1 model outputs are supported");
    }
    if (options.task == ModelTask::Classify)
    {
        require(specs.size() == 1 && specs[0].dimensions.size() <= 4,
                "Classification requires one probability tensor");
        const auto &d = specs[0].dimensions;
        require(d.size() == 1 || d[0] == 1, "Classification requires batch 1");
        int count = int(specs[0].elements());
        require(options.classes == 0 || options.classes == count,
                "Classification class count does not match metadata");
        require(count > 0 && count <= 65535, "Invalid classification output size");
        options_.classes = count;
        layout_ = Layout::Classification;
        return;
    }
    if (options.task == ModelTask::Semantic || options.task == ModelTask::Depth)
    {
        require(specs.size() == 1, "Dense prediction requires one output tensor");
        const auto &s = specs[0];
        const auto &d = s.dimensions;
        require(d.size() == 3 || d.size() == 4, "Dense output must be [1,H,W] or [1,C,H,W]");
        if (d.size() == 4)
        {
            int channels = d[s.channels_first ? 1 : 3];
            require(options.task != ModelTask::Depth || channels == 1,
                    "Depth output requires one channel");
            require(options.task != ModelTask::Semantic || options.classes == 0 ||
                        channels == options.classes,
                    "Semantic class count differs from metadata");
            options_.classes = channels;
        }
        else
        {
            require(options.task != ModelTask::Semantic || options.classes > 0,
                    "A semantic class map requires metadata class names");
        }
        require(options.dense_resize == DenseResize::None || d.size() == 4,
                "Class-ID maps cannot be interpolated as logits");
        layout_ = Layout::Dense;
        return;
    }
    if (options.task == ModelTask::Pose)
    {
        require(options.keypoints > 0 && options.keypoints <= 512 &&
                    (options.keypoint_dimensions == 2 || options.keypoint_dimensions == 3),
                "kpt_shape must be [count,2] or [count,3]");
    }
    // A prototype image is distinct from the per-proposal prediction tensor.
    if (options.task == ModelTask::Segment)
    {
        for (size_t i = 0; i < specs.size(); ++i)
        {
            const auto &s = specs[i];
            const auto &d = s.dimensions;
            if (d.size() == 4 && d[s.channels_first ? 1 : 3] <= 256)
            {
                // Rockchip raw heads append the prototype tensor after all branches.
                prototypes_ = int(i);
            }
        }
        require(prototypes_ >= 0, "Segmentation requires mask prototypes");
        const auto &proto = specs[size_t(prototypes_)];
        mask_channels_ = proto.dimensions[proto.channels_first ? 1 : 3];
    }
    const int extras = options.task == ModelTask::Segment ? mask_channels_
                       : options.task == ModelTask::Pose
                           ? options.keypoints * options.keypoint_dimensions
                       : options.task == ModelTask::Obb ? 1
                                                        : 0;
    // Standard exports: a decoded matrix, optionally followed by mask prototypes.
    if (specs.size() == size_t(prototypes_ >= 0 ? 2 : 1))
    {
        prediction_ = prototypes_ == 0 ? 1 : 0;
        const auto &d = specs[size_t(prediction_)].dimensions;
        require(d.size() == 3, "Decoded predictions require a rank-3 tensor");
        const bool end_to_end =
            d[2] == 6 + extras && (options.end_to_end || d[1] != 4 + options.classes + extras);
        if (end_to_end)
        {
            require(options.classes > 0, "End-to-end outputs require metadata class names");
            layout_ = Layout::EndToEnd;
            feature_major_ = false;
            features_ = d[2];
            proposals_ = d[1];
        }
        else
        {
            int expected = options.classes ? 4 + options.classes + extras : 0;
            feature_major_ = expected ? d[1] == expected : d[1] < d[2];
            features_ = d[feature_major_ ? 1 : 2];
            proposals_ = d[feature_major_ ? 2 : 1];
            int classes = features_ - 4 - extras;
            require(classes > 0 && (options.classes == 0 || classes == options.classes),
                    "Decoded class count or task channels differ from metadata");
            options_.classes = classes;
        }
        return;
    }
    // Rockchip's truncated head exposes scores and xyxy boxes separately.
    for (size_t i = 0; i < specs.size(); ++i)
    {
        if (int(i) == prototypes_)
        {
            continue;
        }
        const auto &d = specs[i].dimensions;
        if (d.size() == 3 && (d[1] == 4 || d[2] == 4))
        {
            if (boxes_ < 0 || specs[i].name == "boxes" ||
                specs[i].name.find("Mul_2") != std::string::npos)
            {
                boxes_ = int(i);
            }
        }
    }
    if (boxes_ >= 0)
    {
        const auto &d = specs[size_t(boxes_)].dimensions;
        box_major_ = d[1] == 4;
        proposals_ = d[box_major_ ? 2 : 1];
        for (size_t i = 0; i < specs.size(); ++i)
        {
            if (int(i) == boxes_ || int(i) == prototypes_)
            {
                continue;
            }
            const auto &shape = specs[i].dimensions;
            if (shape.size() != 3)
            {
                continue;
            }
            int channels = shape[shape[2] == proposals_ ? 1 : 2];
            if (shape[1] != proposals_ && shape[2] != proposals_)
            {
                continue;
            }
            if (extras && channels == extras && auxiliary_ < 0 &&
                (options.classes != extras || (specs[i].name != "scores" &&
                                               specs[i].name.find("Sigmoid") == std::string::npos)))
            {
                auxiliary_ = int(i);
            }
            else if (options.classes == 0 || channels == options.classes)
            {
                scores_ = int(i);
                options_.classes = channels;
                score_major_ = shape[2] == proposals_;
            }
        }
        require(scores_ >= 0 && (extras == 0 || auxiliary_ >= 0),
                "Split head lacks scores or task-specific output");
        layout_ = Layout::Split;
        return;
    }
    // Raw model-zoo exports use three spatial prediction branches.
    int count = int(specs.size()) - (prototypes_ >= 0 ? 1 : 0);
    int stride = count == 13 - 1 ? 4 : count == 9 || count == 10 ? 3 : count == 6 ? 2 : 0;
    const bool packed_task =
        (options.task == ModelTask::Pose || options.task == ModelTask::Obb) && count == 4;
    if (packed_task)
    {
        stride = 1;
    }
    require(stride > 0 && count >= stride * 3, "Unrecognized YOLO output layout");
    require(prototypes_ < 0 || prototypes_ == int(specs.size()) - 1,
            "Raw segmentation heads require the prototype tensor last");
    proposals_ = 0;
    for (int branch = 0; branch < 3; ++branch)
    {
        int reg = branch * stride, cls = packed_task ? reg : reg + 1;
        const auto &d = specs[size_t(reg)].dimensions;
        const auto &c = specs[size_t(cls)].dimensions;
        require(d.size() == 4 && c.size() == 4 && specs[size_t(reg)].channels_first &&
                    specs[size_t(cls)].channels_first && d[2] == c[2] && d[3] == c[3] &&
                    width % d[3] == 0 && height % d[2] == 0 && width / d[3] == height / d[2],
                "Raw YOLO branches require matching NCHW grids");
        int classes = packed_task ? d[1] - 64 : c[1];
        require(classes > 0 && (options_.classes == 0 || classes == options_.classes),
                "Raw head class count differs from metadata");
        options_.classes = classes;
        require(packed_task || d[1] == 4 || (d[1] >= 8 && d[1] % 4 == 0 && d[1] <= 128),
                "Invalid DFL regression channels");
        regressions_.push_back(reg);
        class_scores_.push_back(cls);
        if (stride == 4)
        {
            const auto &extra = specs[size_t(reg + 3)].dimensions;
            require(extra.size() == 4 && extra[1] == extras && extra[2] == d[2] && extra[3] == d[3],
                    "Raw task channel/grid mismatch");
            extras_.push_back(reg + 3);
        }
        proposals_ += d[2] * d[3];
    }
    if (extras && extras_.empty())
    {
        auxiliary_ = stride * 3;
        require(auxiliary_ < count &&
                    specs[size_t(auxiliary_)].elements() == size_t(extras) * proposals_,
                "Raw model is missing keypoints, angles, or mask coefficients");
    }
    layout_ = Layout::Dfl;
}

const char *TaskPostprocessor::description() const
{
    switch (layout_)
    {
    case Layout::Decoded:
        return "decoded YOLO";
    case Layout::EndToEnd:
        return "end-to-end YOLO";
    case Layout::Split:
        return "split scores/boxes";
    case Layout::Dfl:
        return "spatial YOLO heads";
    case Layout::Classification:
        return "classification probabilities";
    case Layout::Dense:
        return "dense prediction";
    }
    return "unknown";
}
} // namespace nvr
