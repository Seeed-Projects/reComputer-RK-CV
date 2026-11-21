#pragma once

#include "engine/task_model.hpp"
#include "engine/json.hpp"

namespace nvr
{
ModelOptions model_options(const Json &info);
Json task_result_json(const TaskResult &result, ModelTask task);
} // namespace nvr
