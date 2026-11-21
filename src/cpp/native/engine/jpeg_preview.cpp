#include "engine/jpeg_preview.hpp"

#include <rk_venc_cfg.h>
#include <algorithm>
#include <cstring>

namespace nvr
{
void JpegPreview::initialize(int width, int height)
{
    width_ = std::min(640, width) & ~1;
    height_ = std::max(2, (height * width_ / width) & ~1);
    hs_ = (width_ + 15) & ~15;
    vs_ = (height_ + 15) & ~15;
    require(mpp_check_support_format(MPP_CTX_ENC, MPP_VIDEO_CodingMJPEG) == MPP_OK,
            "MPP JPEG encoding is unavailable");
    image_ = std::make_unique<DmaBuffer>(size_t(hs_) * size_t(vs_) * 3 / 2);
    destination_ = importbuffer_fd(image_->fd, int(image_->size));
    require(destination_, "import preview destination");
    MppBufferInfo info{};
    info.type = MPP_BUFFER_TYPE_EXT_DMA;
    info.fd = image_->fd;
    info.size = image_->size;
    require(mpp_buffer_import(&input_buffer_, &info) == MPP_OK,
            "import preview DMA into MPP encoder");
    require(mpp_create(&context_, &api_) == MPP_OK, "create preview encoder");
    require(mpp_init(context_, MPP_CTX_ENC, MPP_VIDEO_CodingMJPEG) == MPP_OK,
            "initialize MPP JPEG encoder");
    require(mpp_enc_cfg_init(&cfg_) == MPP_OK, "allocate JPEG encoder configuration");
    require(api_->control(context_, MPP_ENC_GET_CFG, cfg_) == MPP_OK, "get JPEG encoder config");
    mpp_enc_cfg_set_s32(cfg_, "prep:width", width_);
    mpp_enc_cfg_set_s32(cfg_, "prep:height", height_);
    mpp_enc_cfg_set_s32(cfg_, "prep:hor_stride", hs_);
    mpp_enc_cfg_set_s32(cfg_, "prep:ver_stride", vs_);
    mpp_enc_cfg_set_s32(cfg_, "prep:format", MPP_FMT_YUV420SP);
    mpp_enc_cfg_set_s32(cfg_, "codec:type", MPP_VIDEO_CodingMJPEG);
    mpp_enc_cfg_set_s32(cfg_, "jpeg:q_factor", 70);
    mpp_enc_cfg_set_s32(cfg_, "jpeg:qf_min", 40);
    mpp_enc_cfg_set_s32(cfg_, "jpeg:qf_max", 80);
    require(api_->control(context_, MPP_ENC_SET_CFG, cfg_) == MPP_OK, "configure MPP JPEG encoder");
    RK_S64 timeout = 1000;
    require(api_->control(context_, MPP_SET_OUTPUT_TIMEOUT, &timeout) == MPP_OK,
            "set preview encoding timeout");
}

JpegPreview::~JpegPreview()
{
    if (context_)
    {
        mpp_destroy(context_);
    }
    if (cfg_)
    {
        mpp_enc_cfg_deinit(cfg_);
    }
    if (input_buffer_)
    {
        mpp_buffer_put(input_buffer_);
    }
    for (auto &entry : sources_)
    {
        releasebuffer_handle(entry.second);
    }
    if (destination_)
    {
        releasebuffer_handle(destination_);
    }
}

std::vector<uint8_t> JpegPreview::encode(const Frame &frame)
{
    auto source = frame.value;
    int w = int(mpp_frame_get_width(source)), h = int(mpp_frame_get_height(source));
    if (!context_)
    {
        initialize(w, h);
    }
    require(mpp_frame_get_fmt(source) == MPP_FMT_YUV420SP, "preview needs linear NV12");
    auto buffer = mpp_frame_get_buffer(source);
    int fd = mpp_buffer_get_fd(buffer);
    auto found = sources_.find(fd);
    if (found == sources_.end())
    {
        require(sources_.size() < 96, "preview import cache limit");
        auto handle = importbuffer_fd(fd, int(mpp_buffer_get_size(buffer)));
        require(handle, "import preview source");
        found = sources_.emplace(fd, handle).first;
    }
    auto src = wrapbuffer_handle_t(found->second, w, h, int(mpp_frame_get_hor_stride(source)),
                                   int(mpp_frame_get_ver_stride(source)), RK_FORMAT_YCbCr_420_SP);
    auto dst = wrapbuffer_handle_t(destination_, width_, height_, hs_, vs_, RK_FORMAT_YCbCr_420_SP);
    auto status = imresize(src, dst, 0, 0, INTER_LINEAR, 1);
    require(status == IM_STATUS_SUCCESS,
            std::string("RGA preview resize: ") + imStrError_t(status));
    MppFrame input = nullptr;
    require(mpp_frame_init(&input) == MPP_OK, "allocate preview frame");
    mpp_frame_set_width(input, uint32_t(width_));
    mpp_frame_set_height(input, uint32_t(height_));
    mpp_frame_set_hor_stride(input, uint32_t(hs_));
    mpp_frame_set_ver_stride(input, uint32_t(vs_));
    mpp_frame_set_fmt(input, MPP_FMT_YUV420SP);
    mpp_frame_set_buffer(input, input_buffer_);
    mpp_frame_set_pts(input, frame.pts_us);
    auto result = api_->encode_put_frame(context_, input);
    mpp_frame_deinit(&input);
    require(result == MPP_OK, "submit preview frame");
    MppPacket packet = nullptr;
    result = api_->encode_get_packet(context_, &packet);
    require(result == MPP_OK && packet, "MPP JPEG encode timeout");
    auto *bytes = static_cast<uint8_t *>(mpp_packet_get_pos(packet));
    size_t size = mpp_packet_get_length(packet);
    std::vector<uint8_t> encoded(bytes, bytes + size);
    mpp_packet_deinit(&packet);
    require(encoded.size() > 2 && encoded[0] == 0xff && encoded[1] == 0xd8,
            "invalid hardware JPEG");
    return encoded;
}
} // namespace nvr
