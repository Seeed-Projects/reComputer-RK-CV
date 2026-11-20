#include "engine/detector_pool.hpp"
#include "engine/affinity.hpp"

#include <fstream>
#include <thread>

namespace nvr
{
DetectorPool::DetectorPool(const std::string &model,
                           float confidence, float iou, int workers, const ModelOptions &options)
{
    configure(confidence, iou);
    std::ifstream compatible("/proc/device-tree/compatible", std::ios::binary);
    std::string chip((std::istreambuf_iterator<char>(compatible)), {});
    const int cores = chip.find("rockchip,rk3588") != std::string::npos   ? 3
                      : chip.find("rockchip,rk3576") != std::string::npos ? 2
                                                                          : 1;
    require(workers == 0 || workers == 1, "inference_workers must be 0 (auto) or 1 (single context)");
    if (workers == 0)
    {
        workers = cores;
    }
    Json masks = Json::array();
    for (int i = 0; i < workers; ++i)
    {
        auto mask = workers == 1 ? RKNN_NPU_CORE_ALL : static_cast<rknn_core_mask>(1 << i);
        detectors_.push_back(
            std::make_unique<Detector>(model, confidence, iou, mask, options));
        masks.push_back(detectors_.back()->description.at("core_mask"));
    }
    description = detectors_.front()->description;
    description["inference_workers"] = workers;
    description["core_masks"] = masks;
    description["prefer_performance_cpus"] = cores > 1;
    description["core_mask"] = workers == 1 ? "all" : "per-worker";
}

void DetectorPool::run(FrameQueue &queue, const std::function<bool(const Frame &)> &active,
                       bool publish_candidates,
                       const std::function<void(const FramePtr &, Json, const std::vector<Detection> &)> &consume)
{
    std::mutex mutex, dequeue_mutex;
    std::condition_variable ready;
    uint64_t issued = 0, published = 0;
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::vector<std::thread> threads;
    auto fail = [&]
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error)
        {
            error = std::current_exception();
        }
        failed = true;
        queue.close();
        ready.notify_all();
    };
    try
    {
        for (auto &detector : detectors_)
        {
            threads.emplace_back(
                [&, worker = detector.get()]
                {
                    try
                    {
                        if (description.at("prefer_performance_cpus").get<bool>())
                        {
                            prefer_performance_cpus();
                        }
                        std::vector<Detection> unfiltered;
                        while (true)
                        {
                            FramePtr frame;
                            uint64_t ticket;
                            {
                                // Assign tickets in source order, including across idle workers.
                                std::lock_guard<std::mutex> lock(dequeue_mutex);
                                if (failed)
                                {
                                    break;
                                }
                                frame = queue.pop();
                                if (!frame)
                                {
                                    break;
                                }
                                ticket = issued++;
                            }
                            Json result;
                            if (active(*frame))
                            {
                                {
                                    std::lock_guard<std::mutex> lock(config_mutex_);
                                    worker->configure(confidence_, iou_);
                                }
                                result = worker->infer(*frame, publish_candidates ? &unfiltered : nullptr);
                            }
                            std::unique_lock<std::mutex> lock(mutex);
                            ready.wait(lock,
                                       [&]
                                       {
                                           return failed || ticket == published;
                                       });
                            if (failed)
                            {
                                break;
                            }
                            if (!result.is_null() && active(*frame))
                            {
                                consume(frame, std::move(result), unfiltered);
                            }
                            ++published;
                            ready.notify_all();
                        }
                    }
                    catch (...)
                    {
                        fail();
                    }
                });
        }
    }
    catch (...)
    {
        fail();
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
    if (error)
    {
        std::rethrow_exception(error);
    }
}
} // namespace nvr
