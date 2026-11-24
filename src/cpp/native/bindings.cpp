#include "engine/affinity.hpp"
#include "engine/task_json.hpp"
#include "engine/decoder.hpp"
#include "engine/detector_pool.hpp"
#include "engine/jpeg_preview.hpp"

#include <pybind11/pybind11.h>
#include <cstring>
#include <thread>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>

extern "C"
{
#include <libavdevice/avdevice.h>
}

namespace py = pybind11;
using namespace nvr;

// ARM64 Linux uses little-endian IEEE floats. Append fields individually to avoid padding.
static std::string candidate_packet(const Frame &frame, const Json &result,
                                    const std::vector<Detection> &candidates)
{
    const uint32_t size = candidates.size() * 6 * sizeof(float);
    std::string packet;
    packet.reserve(32 + size);
    auto append = [&](const auto &value) {
        packet.append(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    append(size);
    append(result.at("source_width").get<uint32_t>());
    append(result.at("source_height").get<uint32_t>());
    append(uint32_t(frame.cycle));
    append(result.at("media_time").get<double>());
    append(frame.sequence);
    for (const auto &candidate : candidates)
    {
        packet.append(reinterpret_cast<const char *>(candidate.box.data()), 4 * sizeof(float));
        append(candidate.confidence);
        append(float(candidate.class_id));
    }
    return packet;
}

// A request context is serialized; live workers have their own independent contexts.
class Predictor
{
    std::mutex mutex_;
    std::unique_ptr<DmaBuffer> image_;
    Detector detector_;

public:
    Predictor(const std::string &model, const std::string &options)
        : detector_(model, .25f, .45f, RKNN_NPU_CORE_ALL, model_options(Json::parse(options))) {}

    Json frame(const Frame &frame, float confidence, float iou)
    {
        prefer_performance_cpus();
        std::lock_guard<std::mutex> lock(mutex_);
        // Request buffers and live decoder pools can reuse file descriptors.
        detector_.clear_sources();
        detector_.configure(confidence, iou);
        try
        {
            auto result = detector_.infer(frame);
            detector_.clear_sources();
            return result;
        }
        catch (...)
        {
            detector_.clear_sources();
            throw;
        }
    }

    std::string image(const std::string &pixels, int width, int height, float confidence, float iou)
    {
        prefer_performance_cpus();
        auto entered = Clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        double wait_ms = milliseconds(entered);
        auto upload_started = Clock::now();
        require(width > 0 && height > 0 && width <= 8192 && height <= 8192 &&
                    width % 2 == 0 && height % 2 == 0 &&
                    pixels.size() == size_t(width) * height * 3 / 2, "invalid NV12 image");
        detector_.clear_sources();
        int hs = (width + 15) & ~15, vs = (height + 15) & ~15;
        size_t size = size_t(hs) * vs * 3 / 2;
        if (!image_ || image_->size < size)
        {
            image_ = std::make_unique<DmaBuffer>(size);
        }
        dma_buf_sync sync{};
        sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
        require(ioctl(image_->fd, DMA_BUF_IOCTL_SYNC, &sync) == 0, "sync uploaded image");
        auto *base = static_cast<uint8_t *>(image_->data);
        for (int row = 0; row < height; ++row)
            std::memcpy(base + size_t(row) * hs, pixels.data() + size_t(row) * width, width);
        for (int row = 0; row < height / 2; ++row)
            std::memcpy(base + size_t(hs) * vs + size_t(row) * hs,
                        pixels.data() + size_t(width) * height + size_t(row) * width, width);
        sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
        require(ioctl(image_->fd, DMA_BUF_IOCTL_SYNC, &sync) == 0, "finish uploaded image sync");
        MppBufferInfo info{};
        info.type = MPP_BUFFER_TYPE_EXT_DMA;
        info.fd = image_->fd;
        info.size = image_->size;
        MppBuffer buffer = nullptr;
        require(mpp_buffer_import(&buffer, &info) == MPP_OK, "import uploaded image");
        MppFrame raw = nullptr;
        auto rc = mpp_frame_init(&raw);
        if (rc != MPP_OK) { mpp_buffer_put(buffer); throw std::runtime_error("create image frame"); }
        Frame frame(raw);
        mpp_frame_set_width(raw, width);
        mpp_frame_set_height(raw, height);
        mpp_frame_set_hor_stride(raw, hs);
        mpp_frame_set_ver_stride(raw, vs);
        mpp_frame_set_fmt(raw, MPP_FMT_YUV420SP);
        mpp_frame_set_buffer(raw, buffer);
        mpp_buffer_put(buffer);
        detector_.configure(confidence, iou);
        double upload_ms = milliseconds(upload_started);
        auto result = detector_.infer(frame);
        result["context_wait_ms"] = wait_ms;
        result["dma_upload_ms"] = upload_ms;
        return result.dump();
    }

    std::string description() const { return detector_.description.dump(); }
    bool supports_candidates() const { return detector_.supports_candidates(); }
};

class Live
{
    std::atomic<bool> stop_{false};
    FrameQueue queue_{6};
    // Destruction order: retained frames and consumers before the decoder.
    std::unique_ptr<Decoder> decoder_;
    std::unique_ptr<DetectorPool> pool_;
    std::unique_ptr<JpegPreview> preview_;
    std::mutex mutex_;
    std::mutex preview_mutex_;
    FramePtr latest_;
    Json result_;
    uint64_t candidate_sequence_ = 0;
    std::string candidates_;
    FramePtr preview_frame_;
    Json preview_result_;
    uint64_t jpeg_frame_ = 0;
    std::vector<uint8_t> jpeg_;
    std::string error_;
    bool ended_ = false;
    std::thread thread_;

    Json status() const
    {
        return {{"error", error_}, {"ended", ended_}, {"dropped", queue_.dropped.load()}};
    }

public:
    Live(const std::string &source, const std::string &model, int workers,
         const std::string &options, bool loop_file)
    {
        avdevice_register_all();
        decoder_ = std::make_unique<Decoder>(source, stop_, source.rfind("/dev/video", 0) == 0);
        pool_ = std::make_unique<DetectorPool>(model, .25f, .45f, workers, model_options(Json::parse(options)));
        const bool publish_candidates = loop_file && pool_->supports_candidates();
        pool_->description["candidate_stream"] = publish_candidates;
        preview_ = std::make_unique<JpegPreview>();
        thread_ = std::thread([this, source, loop_file, publish_candidates]
        {
            std::thread producer;
            try
            {
                producer = std::thread([this, source, loop_file]
                {
                    try
                    {
                        const bool file = source.find("://") == std::string::npos &&
                                          source.rfind("/dev/", 0) != 0;
                        for (uint64_t cycle = 0; !stop_; ++cycle)
                        {
                            auto start = Clock::now();
                            int64_t origin = loop_file ? decoder_->start_us() : AV_NOPTS_VALUE;
                            auto pace = [&](int64_t due_us) {
                                while (!stop_ && milliseconds(start) * 1000 < due_us)
                                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            };
                            bool received = false;
                            decoder_->run([&](FramePtr frame)
                            {
                                received = true;
                                if (file && frame->pts_us != AV_NOPTS_VALUE)
                                {
                                    if (origin == AV_NOPTS_VALUE) origin = frame->pts_us;
                                    pace(frame->pts_us - origin);
                                }
                                frame->cycle = cycle;
                                queue_.push(std::move(frame), stop_);
                            });
                            if (!loop_file || stop_) break;
                            require(received, "video contains no decoded frames");
                            pace(decoder_->duration_us());
                            if (!stop_) decoder_->rewind();
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        error_ = e.what();
                    }
                    queue_.close();
                });
                auto last_preview = Clock::now() - std::chrono::seconds(1);
                pool_->run(queue_, [this](const Frame &) { return !stop_; }, publish_candidates,
                    [&](const FramePtr &frame, Json result, const std::vector<Detection> &unfiltered)
                    {
                        std::string packet;
                        if (loop_file && frame->pts_us != AV_NOPTS_VALUE)
                        {
                            result["media_time"] = (frame->pts_us - decoder_->start_us()) / 1000000.0;
                            result["cycle"] = frame->cycle;
                            if (publish_candidates)
                                packet = candidate_packet(*frame, result, unfiltered);
                        }
                        std::lock_guard<std::mutex> lock(mutex_);
                        latest_ = frame;
                        result_ = std::move(result);
                        if (!packet.empty())
                        {
                            candidate_sequence_ = frame->sequence;
                            candidates_ = std::move(packet);
                        }
                        // Keep the legacy preview at 10 FPS and encode only when it is read.
                        if (milliseconds(last_preview) >= 100)
                        {
                            preview_frame_ = frame;
                            preview_result_ = result_;
                            last_preview = Clock::now();
                        }
                    });
            }
            catch (const std::exception &e)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = e.what();
                stop_ = true;
                queue_.close();
            }
            if (producer.joinable()) producer.join();
            std::lock_guard<std::mutex> lock(mutex_);
            ended_ = true;
        });
    }

    ~Live() { close(); }
    void close()
    {
        stop_ = true;
        queue_.close();
        if (thread_.joinable()) thread_.join();
        queue_.clear();
        {
            std::lock_guard<std::mutex> preview_lock(preview_mutex_);
            std::lock_guard<std::mutex> lock(mutex_);
            latest_.reset();
            preview_frame_.reset();
            preview_.reset();
        }
        pool_.reset();
        decoder_.reset();
    }
    void configure(float confidence, float iou) { pool_->configure(confidence, iou); }
    std::string description() const { return pool_->description.dump(); }
    std::string predict(Predictor &predictor, float confidence, float iou)
    {
        FramePtr frame;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            frame = latest_;
        }
        require(bool(frame), "live source has no frame yet");
        return predictor.frame(*frame, confidence, iou).dump();
    }
    py::tuple snapshot()
    {
        std::lock_guard<std::mutex> preview_lock(preview_mutex_);
        FramePtr frame;
        Json state;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            frame = preview_frame_;
            state = status();
            state["result"] = preview_result_;
        }
        if (frame && frame->sequence != jpeg_frame_)
        {
            jpeg_ = preview_->encode(*frame);
            jpeg_frame_ = frame->sequence;
        }
        return py::make_tuple(state.dump(), py::bytes(reinterpret_cast<const char *>(jpeg_.data()), jpeg_.size()));
    }

    std::string state(bool include_result, uint64_t after)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto current = status();
        if (include_result)
            current["result"] = latest_ && latest_->sequence > after ? result_ : Json();
        return current.dump();
    }

    py::object candidates(uint64_t after)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        require(error_.empty(), error_);
        if (candidate_sequence_ <= after) return py::none();
        return py::make_tuple(candidate_sequence_, py::bytes(candidates_));
    }
};

PYBIND11_MODULE(_native, m)
{
    py::class_<Predictor>(m, "Predictor")
        .def(py::init<const std::string &, const std::string &>(), py::arg("model"), py::arg("options") = "{}")
        .def("predict", &Predictor::image, py::call_guard<py::gil_scoped_release>())
        .def("description", &Predictor::description)
        .def("supports_candidates", &Predictor::supports_candidates);
    py::class_<Live>(m, "Live")
        .def(py::init<const std::string &, const std::string &, int, const std::string &, bool>(),
             py::arg("source"), py::arg("model"), py::arg("workers") = 0, py::arg("options") = "{}",
             py::arg("loop_file") = false)
        .def("configure", &Live::configure)
        .def("description", &Live::description)
        .def("predict", &Live::predict, py::call_guard<py::gil_scoped_release>())
        .def("snapshot", &Live::snapshot)
        .def("state", &Live::state, py::arg("include_result") = false, py::arg("after") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("candidates", &Live::candidates)
        .def("close", &Live::close, py::call_guard<py::gil_scoped_release>());
}
