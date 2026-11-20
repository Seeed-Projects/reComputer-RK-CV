#include "engine/task_json.hpp"
#include "engine/common.hpp"

namespace nvr
{
ModelOptions model_options(const Json &info)
{
    ModelOptions options;
    options.task = model_task(info.value("task", std::string("detect")));
    options.classes = info.value("classes", info.contains("names") ? int(info["names"].size()) : 0);
    if (info.contains("kpt_shape"))
    {
        const auto &shape = info.at("kpt_shape");
        require(shape.is_array() && shape.size() == 2, "kpt_shape must be [count,dimensions]");
        options.keypoints = shape[0].get<int>();
        options.keypoint_dimensions = shape[1].get<int>();
    }
    options.end_to_end = info.value("end2end", false);
    options.apply_nms = info.value("nms", true);
    options.normalized = info.value("coordinates", std::string("pixels")) == "normalized";
    options.center_crop = info.value("preprocess", options.task == ModelTask::Classify
                                                       ? std::string("center_crop")
                                                       : std::string("letterbox")) == "center_crop";
    const std::string resize = info.value("dense_resize", std::string("none"));
    require(resize == "none" || resize == "half_pixel" || resize == "align_corners",
            "Unsupported dense interpolation policy");
    options.dense_resize = resize == "half_pixel"      ? DenseResize::HalfPixel
                           : resize == "align_corners" ? DenseResize::AlignCorners
                                                       : DenseResize::None;
    return options;
}

Json task_result_json(const TaskResult &result, ModelTask task)
{
    auto raster_json = [](const ResultRaster &raster)
    {
        return Json{{"width", raster.width}, {"height", raster.height}, {"box", raster.box},
                    {"runs", raster.runs},   {"offset", raster.offset}, {"scale", raster.scale}};
    };
    Json output = {{"task", task_name(task)}, {"detections", Json::array()}};
    for (const auto &item : result.detections)
    {
        Json detection = {{"class_id", item.detection.class_id},
                          {"confidence", item.detection.confidence},
                          {"box", item.detection.box}};
        if (!item.keypoints.empty())
        {
            detection["keypoints"] = item.keypoints;
        }
        if (!item.polygon.empty())
        {
            detection["polygon"] = item.polygon;
        }
        if (item.mask)
        {
            detection["mask"] = raster_json(*item.mask);
        }
        output["detections"].push_back(std::move(detection));
    }
    if (task == ModelTask::Classify)
    {
        output["classification"] = Json::array();
        for (const auto &prediction : result.classification)
        {
            output["classification"].push_back(
                {{"class_id", prediction.class_id}, {"confidence", prediction.confidence}});
        }
    }
    if (result.dense)
    {
        output["dense"] = raster_json(*result.dense);
    }
    return output;
}
} // namespace nvr
