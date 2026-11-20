#pragma once

#include "engine/frame.hpp"
#include "engine/json.hpp"
#include "engine/preprocess.hpp"
#include "engine/postprocess.hpp"
#include "engine/task_model.hpp"
#include "engine/native_output.hpp"

#include <memory>
#include <string>
#include <vector>
#include <rknn_api.h>

namespace nvr
{
// RKNN inference and task postprocessing.
class Detector
{
    rknn_context context_ = 0;
    rknn_tensor_attr input_attr_{}, output_attr_{};
    rknn_tensor_mem *tensor_ = nullptr;
    std::unique_ptr<DmaBuffer> input_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<YoloPostprocessor> postprocessor_;
    std::unique_ptr<TaskPostprocessor> task_postprocessor_;
    std::vector<std::vector<float>> task_outputs_;
    std::unique_ptr<NativeOutput> dense_output_;
    ModelOptions options_;
    bool task_output_ = false;
    float confidence_ = 0.35f, nms_iou_ = 0.45f;
    std::vector<float> output_;
    int width_ = 0, height_ = 0, stride_ = 0;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<std::vector<int8_t>> quant_outputs_;
    bool dfl_ = false;
    bool split_ = false, scores_first_ = true, boxes_first_ = true;
    int scores_index_ = -1, boxes_index_ = -1;
    int features_ = 0, proposals_ = 0;
    bool features_first_ = true;
    double last_run_ms_ = 0, last_output_ms_ = 0;

    void run_output();

public:
    Json description;

    Detector(const std::string &model, float confidence, float iou, rknn_core_mask core_mask = RKNN_NPU_CORE_ALL,
             const ModelOptions &options = {});
    ~Detector();

    Detector(const Detector &) = delete;
    Detector &operator=(const Detector &) = delete;

    bool supports_candidates() const { return split_ && !task_output_; }
    Json infer(const Frame &frame, std::vector<Detection> *unfiltered = nullptr);
    void configure(float confidence, float iou)
    {
        confidence_ = confidence;
        nms_iou_ = iou;
        if (postprocessor_)
            postprocessor_->configure(confidence, iou);
    }
    void clear_sources() { preprocessor_->clear_sources(); }
};
} // namespace nvr
