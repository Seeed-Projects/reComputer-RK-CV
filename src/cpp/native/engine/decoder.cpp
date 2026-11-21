#include "engine/decoder.hpp"

#include <algorithm>
#include <thread>


namespace nvr
{
int Decoder::interrupt(void *opaque)
{
    auto *self = static_cast<Decoder *>(opaque);
    const auto now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count();
    return self->stop_ || now > self->deadline_ns_;
}

void Decoder::deadline()
{
    deadline_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       (Clock::now() + std::chrono::seconds(8)).time_since_epoch())
                       .count();
}

// Decoder lifetime.
Decoder::Decoder(const std::string &uri, std::atomic<bool> &stop, bool camera) : stop_(stop)
{
    try
    {
        format_ = avformat_alloc_context();
        require(format_, "allocate demux context");
        format_->interrupt_callback = {interrupt, this};
        deadline();
        AVDictionary *options = nullptr;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        av_dict_set(&options, "rw_timeout", "5000000", 0);
        if (camera)
        {
            av_dict_set(&options, "input_format", "h264", 0);
        }
        int rc =
            avformat_open_input(&format_, uri.c_str(),
                                camera ? av_find_input_format("video4linux2") : nullptr, &options);
        av_dict_free(&options);
        require(rc >= 0, "open input: " + av_error(rc));
        deadline();
        rc = avformat_find_stream_info(format_, nullptr);
        require(rc >= 0, "read stream info: " + av_error(rc));
        stream_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        require(stream_ >= 0, "no video stream");
        auto codec = parameters()->codec_id;
        require(codec == AV_CODEC_ID_H264 || codec == AV_CODEC_ID_HEVC,
                "demo supports H.264/H.265 video");
        const auto *filter =
            av_bsf_get_by_name(codec == AV_CODEC_ID_H264 ? "h264_mp4toannexb" : "hevc_mp4toannexb");
        require(filter && av_bsf_alloc(filter, &bsf_) >= 0, "create Annex-B filter");
        require(avcodec_parameters_copy(bsf_->par_in, parameters()) >= 0,
                "copy decoder parameters");
        bsf_->time_base_in = time_base();
        require(av_bsf_init(bsf_) >= 0, "init Annex-B filter");
        require(mpp_create(&context_, &api_) == MPP_OK, "mpp_create");
        RK_U32 split = 1;
        require(api_->control(context_, MPP_DEC_SET_PARSER_SPLIT_MODE, &split) == MPP_OK,
                "MPP split parser");
        const auto init_result =
            mpp_init(context_, MPP_CTX_DEC,
                     codec == AV_CODEC_ID_H264 ? MPP_VIDEO_CodingAVC : MPP_VIDEO_CodingHEVC);
        require(init_result == MPP_OK,
                "MPP decoder initialization failed (" + std::to_string(init_result) +
                    "): check the board MPP driver and read/write permissions on /dev/mpp_service");
        MppFrameFormat fmt = MPP_FMT_YUV420SP;
        require(api_->control(context_, MPP_DEC_SET_OUTPUT_FORMAT, &fmt) == MPP_OK,
                "set linear NV12 output");
        RK_S64 timeout = MPP_POLL_NON_BLOCK;
        require(api_->control(context_, MPP_SET_OUTPUT_TIMEOUT, &timeout) == MPP_OK,
                "set nonblocking decode output");
    }
    catch (...)
    {
        if (context_)
        {
            mpp_destroy(context_);
        }
        context_ = nullptr;
        if (bsf_)
        {
            av_bsf_free(&bsf_);
        }
        if (format_)
        {
            avformat_close_input(&format_);
        }
        throw;
    }
}

Decoder::~Decoder()
{
    // A seek/stop may leave reordered pictures inside MPP. Flush those slots
    // after consumers release their frames, before destroying the decoder.
    if (context_)
    {
        api_->reset(context_);
        mpp_destroy(context_);
    }
    if (group_)
    {
        mpp_buffer_group_put(group_);
    }
    if (bsf_)
    {
        av_bsf_free(&bsf_);
    }
    if (format_)
    {
        avformat_close_input(&format_);
    }
}

// MPP packet submission and frame delivery.
bool Decoder::drain(const std::function<void(FramePtr)> &consume, bool &eos)
{
    bool progress = false;
    for (;;)
    {
        MppFrame raw = nullptr;
        MPP_RET rc = api_->decode_get_frame(context_, &raw);
        require(rc == MPP_OK || rc == MPP_ERR_TIMEOUT, "MPP get frame error " + std::to_string(rc));
        if (!raw)
        {
            break;
        }
        progress = true;
        progress_ = Clock::now();
        auto frame = std::make_shared<Frame>(raw);
        if (mpp_frame_get_info_change(raw))
        {
            // Retain old frame ownership in consumers until their RGA work completes.
            if (!group_)
            {
                require(mpp_buffer_group_get_internal(&group_, MPP_BUFFER_TYPE_DRM) == MPP_OK,
                        "create MPP frame pool");
            }
            require(mpp_buffer_group_limit_config(group_, mpp_frame_get_buf_size(raw), 24) ==
                        MPP_OK,
                    "size MPP frame pool");
            require(api_->control(context_, MPP_DEC_SET_EXT_BUF_GROUP, group_) == MPP_OK,
                    "bind MPP frame pool");
            require(api_->control(context_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr) == MPP_OK,
                    "acknowledge MPP format change");
            continue;
        }
        eos = mpp_frame_get_eos(raw) != 0;
        if (mpp_frame_get_errinfo(raw) || mpp_frame_get_discard(raw))
        {
            continue;
        }
        if (!mpp_frame_get_buffer(raw))
        {
            continue;
        }
        frame->sequence = ++sequence_;
        frame->pts_us = mpp_frame_get_pts(raw);
        consume(std::move(frame));
    }
    return progress;
}

void Decoder::submit(AVPacket *avpacket, bool end, const std::function<void(FramePtr)> &consume,
                     bool &eos)
{
    MppPacket packet = nullptr;
    require(mpp_packet_init(&packet, avpacket ? avpacket->data : nullptr,
                            avpacket ? static_cast<size_t>(avpacket->size) : 0) == MPP_OK,
            "create MPP packet");
    if (end)
    {
        mpp_packet_set_eos(packet);
    }
    if (avpacket && avpacket->pts != AV_NOPTS_VALUE)
    {
        mpp_packet_set_pts(
            packet, av_rescale_q(avpacket->pts, bsf_->time_base_out, AVRational{1, 1000000}));
    }
    try
    {
        while (!stop_)
        {
            MPP_RET rc = api_->decode_put_packet(context_, packet);
            if (rc == MPP_OK)
            {
                progress_ = Clock::now();
                break;
            }
            require(rc == MPP_ERR_BUFFER_FULL || rc == MPP_NOK,
                    "MPP submit error " + std::to_string(rc));
            drain(consume, eos);
            require(milliseconds(progress_) < 8000, "MPP submit stalled");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        mpp_packet_deinit(&packet);
    }
    catch (...)
    {
        mpp_packet_deinit(&packet);
        throw;
    }
    drain(consume, eos);
}

int64_t Decoder::start_us() const
{
    return format_->start_time == AV_NOPTS_VALUE ? 0 : format_->start_time;
}

int64_t Decoder::duration_us() const
{
    return format_->duration == AV_NOPTS_VALUE ? 0 : format_->duration;
}

void Decoder::rewind()
{
    deadline();
    const auto start = av_rescale_q(start_us(), AV_TIME_BASE_Q, time_base());
    const int result = av_seek_frame(format_, stream_, start, AVSEEK_FLAG_BACKWARD);
    if (stop_) return;
    require(result >= 0, "rewind video: " + av_error(result));
    av_bsf_flush(bsf_);
    require(api_->reset(context_) == MPP_OK, "reset MPP decoder");
    progress_ = Clock::now();
}

void Decoder::run(const std::function<void(FramePtr)> &consume)
{
    auto free_packet = [](AVPacket *p)
    {
        av_packet_free(&p);
    };
    std::unique_ptr<AVPacket, decltype(free_packet)> packet(av_packet_alloc(), free_packet),
        filtered(av_packet_alloc(), free_packet);
    require(packet && filtered, "allocate demux packets");
    bool eos = false, awaiting_key = true;
    auto drain_filter = [&]
    {
        for (;;)
        {
            int rc = av_bsf_receive_packet(bsf_, filtered.get());
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF)
            {
                break;
            }
            require(rc >= 0, "Annex-B receive: " + av_error(rc));
            submit(filtered.get(), false, consume, eos);
            av_packet_unref(filtered.get());
        }
    };
    while (!stop_)
    {
        deadline();
        int rc = av_read_frame(format_, packet.get());
        if (rc == AVERROR_EOF)
        {
            break;
        }
        if (stop_)
        {
            break;
        }
        require(rc >= 0, "read input: " + av_error(rc));
        if (packet->stream_index == stream_)
        {
            if (awaiting_key)
            {
                if (!(packet->flags & AV_PKT_FLAG_KEY) ||
                    (packet->pts == AV_NOPTS_VALUE && packet->dts == AV_NOPTS_VALUE))
                {
                    av_packet_unref(packet.get());
                    continue;
                }
                awaiting_key = false;
            }
            rc = av_bsf_send_packet(bsf_, packet.get());
            require(rc >= 0, "Annex-B submit: " + av_error(rc));
            drain_filter();
        }
        av_packet_unref(packet.get());
    }
    if (!stop_)
    {
        require(av_bsf_send_packet(bsf_, nullptr) >= 0, "flush Annex-B");
        drain_filter();
        submit(nullptr, true, consume, eos);
        while (!eos && !stop_)
        {
            drain(consume, eos);
            require(milliseconds(progress_) < 8000, "MPP EOS stalled");
            if (!eos)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}

} // namespace nvr
