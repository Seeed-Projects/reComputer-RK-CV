#include "engine/detector.hpp"
#include "engine/task_json.hpp"

#include <fstream>

#include <algorithm>

namespace nvr
{
Detector::Detector(const std::string &model, float threshold, float iou,
                   rknn_core_mask core_mask, const ModelOptions &options)
    : options_(options), confidence_(threshold), nms_iou_(iou)
{
    try
    {
        std::ifstream file(model, std::ios::binary);
        require(bool(file), "open RKNN model");
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
        require(!bytes.empty(), "empty RKNN model");
        require(rknn_init(&context_, bytes.data(), static_cast<uint32_t>(bytes.size()), 0,
                          nullptr) == RKNN_SUCC,
                "initialize RKNN model");
        require(rknn_set_core_mask(context_, core_mask) == 0, "enable available NPU cores");

        // Validate the model input contract before allocating hardware buffers.
        rknn_input_output_num io{};
        require(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) == 0,
                "query RKNN I/O");
        require(io.n_input == 1 && io.n_output > 0 && io.n_output <= 16,
                "Model requires one image input and 1..16 output tensors");
        require(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)) == 0,
                "query RKNN input");
        require(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_)) ==
                    0,
                "query RKNN output");
        require(input_attr_.n_dims == 4 && input_attr_.dims[0] == 1,
                "model input must be batch-1 image");
        bool nhwc = input_attr_.fmt == RKNN_TENSOR_NHWC;
        height_ = static_cast<int>(input_attr_.dims[nhwc ? 1 : 2]);
        width_ = static_cast<int>(input_attr_.dims[nhwc ? 2 : 3]);
        require(input_attr_.dims[nhwc ? 3 : 1] == 3 && width_ % 16 == 0 && height_ % 2 == 0,
                "input needs RGB, width aligned to 16 and even height");

        std::vector<TensorSpec> specs;
        output_attrs_.resize(io.n_output);
        for (uint32_t i = 0; i < io.n_output; ++i)
        {
            auto &attr = output_attrs_[i];
            attr.index = i;
            require(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) == 0,
                    "query model output");
            specs.push_back({attr.name, std::vector<int>(attr.dims, attr.dims + attr.n_dims),
                             attr.fmt != RKNN_TENSOR_NHWC});
        }
        task_postprocessor_ = std::make_unique<TaskPostprocessor>(options_, specs, width_, height_);
        const auto layout_kind = task_postprocessor_->layout();
        // Preserve the quantized detection fast path; task heads use reusable float buffers.
        const bool fast_dfl =
            layout_kind == TaskPostprocessor::Layout::Dfl && !options_.end_to_end &&
            io.n_output == 9 &&
            std::all_of(output_attrs_.begin(), output_attrs_.end(),
                        [](const rknn_tensor_attr &attr)
                        {
                            return attr.type == RKNN_TENSOR_INT8 &&
                                   attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                                   attr.scale > 0;
                        }) &&
            output_attrs_[0].dims[1] == 64 && output_attrs_[3].dims[1] == 64 &&
            output_attrs_[6].dims[1] == 64;
        task_output_ = options_.task != ModelTask::Detect ||
                       layout_kind == TaskPostprocessor::Layout::EndToEnd ||
                       task_postprocessor_->classes() > 1020 ||
                       (layout_kind == TaskPostprocessor::Layout::Split && options_.normalized) ||
                       (layout_kind == TaskPostprocessor::Layout::Dfl && !fast_dfl) ||
                       (layout_kind == TaskPostprocessor::Layout::Decoded &&
                        (options_.normalized ||
                         task_postprocessor_->proposals() <= task_postprocessor_->classes() + 4));
        if (task_output_)
        {
            if (options_.task == ModelTask::Semantic ||
                (options_.task == ModelTask::Depth && options_.dense_resize != DenseResize::None))
            {
                dense_output_ = std::make_unique<NativeOutput>(context_, output_attr_);
                if (!dense_output_->bound())
                {
                    dense_output_.reset();
                }
            }
            if (!dense_output_)
            {
                task_outputs_.resize(specs.size());
                for (size_t i = 0; i < specs.size(); ++i)
                {
                    task_outputs_[i].resize(specs[i].elements());
                }
            }
            features_ = task_postprocessor_->classes() + 4;
        }
        else
        {
            // Select the output decoder and allocate its reusable result storage.
            dfl_ = io.n_output == 9;
            split_ = io.n_output == 2;
            if (dfl_)
            {
                output_attrs_.resize(9);
                quant_outputs_.resize(9);
                proposals_ = 0;
                for (uint32_t i = 0; i < 9; ++i)
                {
                    auto &a = output_attrs_[i];
                    a.index = i;
                    require(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) == 0,
                            "query DFL output");
                    require(a.n_dims == 4 && a.dims[0] == 1 && a.fmt == RKNN_TENSOR_NCHW &&
                                a.type == RKNN_TENSOR_INT8 &&
                                a.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && a.scale > 0,
                            "DFL requires affine INT8 NCHW outputs");
                    quant_outputs_[i].resize(a.n_elems);
                    if (i % 3 == 0)
                    {
                        require(a.dims[1] == 64 && a.dims[2] > 0 && a.dims[3] > 0,
                                "DFL needs 16 bins per side");
                        require(width_ % int(a.dims[3]) == 0 && height_ % int(a.dims[2]) == 0 &&
                                    width_ / int(a.dims[3]) == height_ / int(a.dims[2]),
                                "DFL stride mismatch");
                        proposals_ += int(a.dims[2] * a.dims[3]);
                    }
                    else
                    {
                        const auto &b = output_attrs_[i - i % 3];
                        require(a.dims[2] == b.dims[2] && a.dims[3] == b.dims[3],
                                "DFL grid mismatch");
                        if (i % 3 == 1)
                        {
                            if (i == 1)
                            {
                                features_ = int(a.dims[1]) + 4;
                            }
                            require(a.dims[1] == uint32_t(features_ - 4) && features_ > 4 &&
                                        features_ <= 1024,
                                    "DFL class mismatch");
                        }
                        else
                        {
                            require(a.dims[1] == 1, "DFL confidence sum must have one channel");
                        }
                    }
                }
            }
            else if (split_)
            {
                output_attrs_.resize(2);
                for (uint32_t i = 0; i < 2; ++i)
                {
                    auto &a = output_attrs_[i];
                    a.index = i;
                    require(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) == 0,
                            "query YOLO26 output");
                    require(a.n_dims == 3 && a.dims[0] == 1 && a.dims[1] > 0 && a.dims[2] > 0,
                            "YOLO26 outputs need batch-1 rank-3 tensors");
                    if (a.dims[1] == 4 || a.dims[2] == 4)
                    {
                        if (boxes_index_ < 0 ||
                            std::string(a.name).find("Mul_2") != std::string::npos)
                        {
                            boxes_index_ = int(i);
                        }
                    }
                }
                require(boxes_index_ >= 0, "YOLO26 requires a four-coordinate boxes output");
                scores_index_ = 1 - boxes_index_;
                const auto &b = output_attrs_[size_t(boxes_index_)];
                const auto &s = output_attrs_[size_t(scores_index_)];
                boxes_first_ = b.dims[1] == 4;
                proposals_ = int(b.dims[boxes_first_ ? 2 : 1]);
                scores_first_ = s.dims[2] == uint32_t(proposals_);
                require(s.dims[scores_first_ ? 2 : 1] == uint32_t(proposals_),
                        "YOLO26 score/box proposal count mismatch");
                features_ = int(s.dims[scores_first_ ? 1 : 2]) + 4;
                require(features_ > 4 && features_ <= 1024 && proposals_ > 0,
                        "unsupported YOLO26 class count");
                require(s.n_elems == uint32_t(features_ - 4) * uint32_t(proposals_) &&
                            b.n_elems == 4 * uint32_t(proposals_),
                        "YOLO26 tensor size mismatch");
                output_.resize(size_t(features_) * size_t(proposals_));
            }
            else
            {
                require(output_attr_.n_dims == 3 && output_attr_.dims[0] == 1,
                        "unsupported YOLO output rank/batch");
                features_first_ = output_attr_.dims[1] < output_attr_.dims[2];
                features_ = static_cast<int>(output_attr_.dims[features_first_ ? 1 : 2]);
                proposals_ = static_cast<int>(output_attr_.dims[features_first_ ? 2 : 1]);
                require(features_ > 4 && features_ <= 1024 && proposals_ > features_,
                        "unsupported YOLO output layout");
            }

            YoloLayout layout;
            layout.kind = dfl_     ? YoloLayout::Kind::Dfl
                          : split_ ? YoloLayout::Kind::ScoresAndBoxes
                                   : YoloLayout::Kind::Decoded;
            layout.input_width = width_;
            layout.input_height = height_;
            layout.classes = features_ - 4;
            layout.apply_nms = options_.apply_nms && !options_.end_to_end;
            layout.proposals = proposals_;
            layout.features_first = features_first_;
            layout.scores_first = scores_first_;
            layout.boxes_first = boxes_first_;
            if (dfl_)
            {
                for (size_t branch = 0; branch < layout.grids.size(); ++branch)
                {
                    const auto &boxes = output_attrs_[branch * 3];
                    const auto &scores = output_attrs_[branch * 3 + 1];
                    const auto &sums = output_attrs_[branch * 3 + 2];
                    layout.grids[branch] = {
                        int(boxes.dims[3]), int(boxes.dims[2]), boxes.zp,  scores.zp, sums.zp,
                        boxes.scale,        scores.scale,       sums.scale};
                }
            }
            postprocessor_ = std::make_unique<YoloPostprocessor>(layout, threshold, iou);
        }

        // Share one RGB DMA buffer between RGA preprocessing and RKNN input.
        stride_ = input_attr_.w_stride ? static_cast<int>(input_attr_.w_stride) : width_;
        require(stride_ >= width_ && stride_ % 16 == 0, "unsupported RKNN width stride");
        input_ = std::make_unique<DmaBuffer>(
            std::max<size_t>(input_attr_.size_with_stride, size_t(stride_) * height_ * 3));
        tensor_ = rknn_create_mem_from_fd(context_, input_->fd, input_->data,
                                          static_cast<uint32_t>(input_->size), 0);
        require(tensor_, "import RGA destination DMA-BUF into RKNN");
        input_attr_.type = RKNN_TENSOR_UINT8;
        input_attr_.fmt = RKNN_TENSOR_NHWC;
        input_attr_.pass_through = 0;
        input_attr_.h_stride = static_cast<uint32_t>(height_);
        require(rknn_set_io_mem(context_, tensor_, &input_attr_) == 0,
                "bind external RGB input to RKNN");
        if (!task_output_ && !dfl_ && !split_)
        {
            output_.resize(output_attr_.n_elems);
        }
        preprocessor_ = std::make_unique<Preprocessor>(*input_, width_, height_, stride_,
                                                       options_.center_crop);
        rknn_sdk_version version{};
        require(rknn_query(context_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == 0,
                "query RKNN version");
        description = {
            {"runtime", version.api_version},
            {"driver", version.drv_version},
            {"input_width", width_},
            {"input_height", height_},
            {"input_stride", stride_},
            {"classes", task_postprocessor_->classes()},
            {"task", task_name(options_.task)},
            {"nms", options_.apply_nms && !options_.end_to_end},
            {"proposals", task_postprocessor_->proposals()},
            {"preprocess", "rga"},
            {"output_type", get_type_string(output_attr_.type)},
            {"postprocessor", task_output_ ? task_postprocessor_->description()
                              : dfl_       ? "INT8 DFL (9 outputs)"
                              : split_     ? "YOLO26 scores/xyxy (2 outputs)"
                                           : "decoded YOLO (1 output)"},
            {"core_mask", core_mask == RKNN_NPU_CORE_ALL ? Json("all") : Json(int(core_mask))},
            {"input_api", "rknn_create_mem_from_fd + rknn_set_io_mem"}};
    }
    catch (...)
    {
        dense_output_.reset();
        preprocessor_.reset();
        if (tensor_)
        {
            rknn_destroy_mem(context_, tensor_);
        }
        tensor_ = nullptr;
        input_.reset();
        if (context_)
        {
            rknn_destroy(context_);
        }
        context_ = 0;
        throw;
    }
}

Detector::~Detector()
{
    dense_output_.reset();
    preprocessor_.reset();
    if (tensor_)
    {
        rknn_destroy_mem(context_, tensor_);
    }
    input_.reset();
    if (context_)
    {
        rknn_destroy(context_);
    }
}

void Detector::run_output()
{
    auto started = Clock::now();
    require(rknn_run(context_, nullptr) == 0, "RKNN inference");
    last_run_ms_ = milliseconds(started);
    started = Clock::now();
    if (dense_output_)
    {
        dense_output_->sync();
    }
    else if (task_output_)
    {
        std::vector<rknn_output> outputs(task_outputs_.size());
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            outputs[i].index = uint32_t(i);
            outputs[i].want_float = 1;
            outputs[i].is_prealloc = 1;
            outputs[i].buf = task_outputs_[i].data();
            outputs[i].size = uint32_t(task_outputs_[i].size() * sizeof(float));
        }
        require(rknn_outputs_get(context_, uint32_t(outputs.size()), outputs.data(), nullptr) == 0,
                "get task outputs");
        require(rknn_outputs_release(context_, uint32_t(outputs.size()), outputs.data()) == 0,
                "release task outputs");
    }
    else if (dfl_)
    {
        std::vector<rknn_output> outputs(9);
        for (uint32_t i = 0; i < 9; ++i)
        {
            auto &out = outputs[i];
            out.index = i;
            out.want_float = 0;
            out.is_prealloc = 1;
            out.buf = quant_outputs_[i].data();
            out.size = uint32_t(quant_outputs_[i].size());
        }
        require(rknn_outputs_get(context_, 9, outputs.data(), nullptr) == 0,
                "get quantized RKNN outputs");
        require(rknn_outputs_release(context_, 9, outputs.data()) == 0,
                "release quantized RKNN outputs");
    }
    else if (split_)
    {
        rknn_output outputs[2]{};
        for (uint32_t i = 0; i < 2; ++i)
        {
            auto &out = outputs[i];
            out.index = i;
            out.want_float = 1;
            out.is_prealloc = 1;
            out.buf = output_.data() +
                      (int(i) == scores_index_ ? 0 : size_t(features_ - 4) * size_t(proposals_));
            out.size = static_cast<uint32_t>(output_attrs_[i].n_elems * sizeof(float));
        }
        require(rknn_outputs_get(context_, 2, outputs, nullptr) == 0,
                "get YOLO26 scores and boxes");
        require(rknn_outputs_release(context_, 2, outputs) == 0, "release YOLO26 outputs");
    }
    else
    {
        rknn_output out{};
        out.want_float = 1;
        out.is_prealloc = 1;
        out.buf = output_.data();
        out.size = static_cast<uint32_t>(output_.size() * sizeof(float));
        require(rknn_outputs_get(context_, 1, &out, nullptr) == 0, "get RKNN output");
        require(rknn_outputs_release(context_, 1, &out) == 0, "release RKNN output");
    }
    last_output_ms_ = milliseconds(started);
}

Json Detector::infer(const Frame &f, std::vector<Detection> *unfiltered)
{
    auto pre = Clock::now();
    auto geometry = preprocessor_->process(f.image());
    double pre_ms = milliseconds(pre);
    auto start = Clock::now();
    run_output();
    double infer_ms = milliseconds(start);
    double run_ms = last_run_ms_, output_ms = last_output_ms_;
    auto post = Clock::now();
    Json result = {{"task", task_name(options_.task)}, {"detections", Json::array()}};
    double nms_ms = 0;
    if (dense_output_)
    {
        result = task_result_json(
            task_postprocessor_->decode_dense(geometry, dense_output_->view()), options_.task);
    }
    else if (task_output_)
    {
        result = task_result_json(
            task_postprocessor_->decode(geometry, task_outputs_, confidence_, nms_iou_),
            options_.task);
    }
    else
    {
        for (const auto &detection : postprocessor_->decode(geometry, output_, quant_outputs_, &nms_ms, unfiltered))
        {
            result["detections"].push_back({{"class_id", detection.class_id},
                                            {"confidence", detection.confidence},
                                            {"box", detection.box}});
        }
    }
    if (!task_output_) result["nms_ms"] = nms_ms;
    double post_ms = milliseconds(post);
    result.update(Json{{"frame", f.sequence},
                       {"source_width", geometry.width},
                       {"source_height", geometry.height},
                       {"preprocess_ms", pre_ms},
                       {"inference_ms", infer_ms},
                       {"rknn_run_ms", run_ms},
                       {"output_get_ms", output_ms},
                       {"postprocess_ms", post_ms}});
    return result;
}

} // namespace nvr
