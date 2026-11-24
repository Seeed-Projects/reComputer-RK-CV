#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C"
{
#include <libavcodec/defs.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace py = pybind11;

namespace
{
void check(int result, const char *operation)
{
    if (result >= 0)
        return;
    char detail[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(result, detail, sizeof(detail));
    throw std::runtime_error(std::string(operation) + ": " + detail);
}

struct CloseInput
{
    void operator()(AVFormatContext *input) const { avformat_close_input(&input); }
};

struct FreePacket
{
    void operator()(AVPacket *packet) const { av_packet_free(&packet); }
};

class HdmiCapture
{
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<AVFormatContext, CloseInput> input_;
    std::unique_ptr<AVPacket, FreePacket> packet_{av_packet_alloc()};
    std::unique_ptr<SwsContext, decltype(&sws_freeContext)> converter_{nullptr, sws_freeContext};
    AVPixelFormat format_;
    int source_width_, source_height_, width_, height_;
    int frame_size_;
    std::vector<char> bgr_;

public:
    explicit HdmiCapture(const std::string &device)
    {
        avdevice_register_all();
        auto *driver = av_find_input_format("video4linux2");
        if (!driver || !packet_)
            throw std::runtime_error("V4L2 capture is unavailable");
        auto *input = avformat_alloc_context();
        if (!input)
            throw std::bad_alloc();
        input->flags |= AVFMT_FLAG_NONBLOCK;
        check(avformat_open_input(&input, device.c_str(), driver, nullptr), "Open HDMI input");
        input_.reset(input);
        if (input_->nb_streams == 0)
            throw std::runtime_error("HDMI input has no video stream");
        const auto *video = input_->streams[0]->codecpar;
        if (video->codec_id != AV_CODEC_ID_RAWVIDEO || video->width <= 0 || video->height <= 0)
            throw std::runtime_error("HDMI input requires an active raw video signal");
        source_width_ = video->width;
        source_height_ = video->height;
        format_ = static_cast<AVPixelFormat>(video->format);
        frame_size_ = av_image_get_buffer_size(format_, source_width_, source_height_, 1);
        check(frame_size_, "Get HDMI frame size");
        const double scale = std::min({1280.0 / source_width_, 720.0 / source_height_, 1.0});
        width_ = std::max(1, int(std::lround(source_width_ * scale)));
        height_ = std::max(1, int(std::lround(source_height_ * scale)));
        converter_.reset(sws_getContext(source_width_, source_height_, format_,
                                        width_, height_, AV_PIX_FMT_BGR24, SWS_BILINEAR,
                                        nullptr, nullptr, nullptr));
        if (!converter_)
            throw std::runtime_error("Unsupported HDMI pixel format");
        const int matrix = video->color_space == AVCOL_SPC_BT709 ||
                                   (video->color_space == AVCOL_SPC_UNSPECIFIED && source_height_ >= 720)
                               ? SWS_CS_ITU709 : SWS_CS_DEFAULT;
        auto *coefficients = sws_getCoefficients(matrix);
        check(sws_setColorspaceDetails(converter_.get(), coefficients,
                                       video->color_range == AVCOL_RANGE_JPEG,
                                       coefficients, 1, 0, 1 << 16, 1 << 16), "Set HDMI color range");
        bgr_.resize(size_t(width_) * height_ * 3 + AV_INPUT_BUFFER_PADDING_SIZE);
    }

    py::tuple read()
    {
        {
            py::gil_scoped_release unlocked;
            const auto deadline = Clock::now() + std::chrono::seconds(2);
            int result;
            while ((result = av_read_frame(input_.get(), packet_.get())) == AVERROR(EAGAIN) &&
                   Clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (result == AVERROR(EAGAIN))
                throw std::runtime_error("No HDMI frames received. Check the HDMI IN signal.");
            check(result, "Read HDMI frame");
            // Prefer the latest queued frame when the browser consumes below the HDMI refresh rate.
            std::unique_ptr<AVPacket, FreePacket> next(av_packet_alloc());
            if (next)
            {
                for (int i = 0; i < 8 && av_read_frame(input_.get(), next.get()) >= 0; ++i)
                {
                    av_packet_unref(packet_.get());
                    av_packet_move_ref(packet_.get(), next.get());
                }
            }
            if (packet_->size != frame_size_)
                throw std::runtime_error("HDMI frame layout changed. Restart the camera preview.");

            uint8_t *planes[4]{};
            int strides[4]{};
            check(av_image_fill_arrays(planes, strides, packet_->data, format_,
                                       source_width_, source_height_, 1), "Read HDMI plane layout");
            uint8_t *output[] = {reinterpret_cast<uint8_t *>(bgr_.data()), nullptr, nullptr, nullptr};
            const int output_strides[] = {width_ * 3, 0, 0, 0};
            check(sws_scale(converter_.get(), planes, strides, 0, source_height_,
                            output, output_strides), "Convert HDMI frame");
            av_packet_unref(packet_.get());
        }
        return py::make_tuple(width_, height_, py::bytes(bgr_.data(), size_t(width_) * height_ * 3));
    }
};
}

PYBIND11_MODULE(_hdmi, module)
{
    py::class_<HdmiCapture>(module, "HdmiCapture")
        .def(py::init<const std::string &>(), py::call_guard<py::gil_scoped_release>())
        .def("read", &HdmiCapture::read);
}
