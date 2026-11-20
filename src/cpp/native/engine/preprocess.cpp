#include "engine/preprocess.hpp"
#include "engine/common.hpp"

#include <algorithm>
#include <cstring>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>

namespace nvr
{
// Model input resizing and color conversion.
Preprocessor::Preprocessor(DmaBuffer &input, int w, int h, int stride,
                           bool center_crop)
    : width_(w), height_(h), stride_(stride), input_(input), center_crop_(center_crop)
{
    require(w > 0 && h > 0 && w % 16 == 0 && h % 2 == 0 && stride >= w && stride % 16 == 0,
            "invalid RGB destination dimensions or stride");
    require(input.fd >= 0 && input.data && input.size >= size_t(stride) * size_t(h) * 3,
            "invalid RGB destination DMA buffer");
    destination_ = importbuffer_fd(input_.fd, static_cast<int>(input_.size));
    require(destination_ != 0, "RGA import RKNN input DMA-BUF");
}

Preprocessor::~Preprocessor()
{
    if (destination_)
    {
        releasebuffer_handle(destination_);
    }
    clear_sources();
    for (auto handle : scaled_handles_)
    {
        if (handle)
        {
            releasebuffer_handle(handle);
        }
    }
}

void Preprocessor::clear_sources()
{
    for (auto &item : sources_)
    {
        releasebuffer_handle(item.second);
    }
    sources_.clear();
}

rga_buffer_t Preprocessor::resize_buffer(size_t slot, int width, int height)
{
    const int stride = (width + 15) & ~15;
    const size_t bytes = size_t(stride) * size_t(height) * 3 / 2;
    if (!scaled_[slot] || scaled_[slot]->size < bytes)
    {
        if (scaled_handles_[slot])
        {
            releasebuffer_handle(scaled_handles_[slot]);
            scaled_handles_[slot] = 0;
        }
        scaled_[slot] = std::make_unique<DmaBuffer>(bytes);
    }
    if (!scaled_handles_[slot])
    {
        scaled_handles_[slot] = importbuffer_fd(scaled_[slot]->fd, int(scaled_[slot]->size));
        require(scaled_handles_[slot] != 0, "RGA import resize buffer");
    }
    return wrapbuffer_handle_t(scaled_handles_[slot], width, height, stride, height,
                               RK_FORMAT_YCbCr_420_SP);
}

Geometry Preprocessor::process(const Nv12ImageView &image)
{
    const int hs = image.horizontal_stride, vs = image.vertical_stride;
    require(image.width > 0 && image.height > 0 && hs >= image.width && vs >= image.height,
            "invalid NV12 image strides");
    require(image.size >= size_t(hs) * size_t(vs) * 3 / 2, "invalid NV12 image buffer size");
    auto g = letterbox_geometry(image.width, image.height, width_, height_);
    int crop_width = image.width, crop_height = image.height, crop_x = 0, crop_y = 0;
    if (center_crop_)
    {
        float ratio = float(width_) / float(height_);
        crop_width = std::min(image.width, int(float(image.height) * ratio)) & ~1;
        crop_height = std::min(image.height, int(float(image.width) / ratio)) & ~1;
        require(crop_width >= 2 && crop_height >= 2, "Invalid classification crop");
        crop_x = ((image.width - crop_width) / 2) & ~1;
        crop_y = ((image.height - crop_height) / 2) & ~1;
        g.rw = width_;
        g.rh = height_;
        g.px = 0;
        g.py = 0;
        g.scale = float(width_) / float(crop_width);
    }

    int fd = image.fd;
    require(fd >= 0, "RGA preprocessing needs a DMA-BUF file descriptor");
    auto found = sources_.find(fd);
    if (found == sources_.end())
    {
        require(sources_.size() < 96,
                "source buffer cache limit exceeded (reconnect required)");
        auto handle = importbuffer_fd(fd, static_cast<int>(image.size));
        // The import owns its DMA-BUF attachment. The caller retains the source
        // during processing; no decoder-specific buffer reference is stored.
        require(handle != 0, "RGA import NV12 DMA-BUF");
        found = sources_.emplace(fd, handle).first;
    }
    auto dst =
        wrapbuffer_handle_t(destination_, width_, height_, stride_, height_, RK_FORMAT_RGB_888);
    if (!painted_ || previous_.rw != g.rw || previous_.rh != g.rh || previous_.px != g.px ||
        previous_.py != g.py)
    {
        // RK3588 color fill needs RGA2 and below-4-GiB memory. Initialize the
        // constant background once instead, so ordinary DMA heaps also work.
        // Every image is still resized/converted by RGA directly into this buffer.
        dma_buf_sync sync{};
        sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
        require(ioctl(input_.fd, DMA_BUF_IOCTL_SYNC, &sync) == 0,
                "sync model input for letterbox initialization");
        std::memset(input_.data, 114, input_.size);
        sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
        require(ioctl(input_.fd, DMA_BUF_IOCTL_SYNC, &sync) == 0,
                "finish letterbox initialization");
        painted_ = true;
        previous_ = g;
    }
    auto src =
        wrapbuffer_handle_t(found->second, g.width, g.height, hs, vs, RK_FORMAT_YCbCr_420_SP);
    im_rect source_rect{crop_x, crop_y, crop_width, crop_height};
    size_t slot = 0;
    // Large reductions can exceed an available core's scale range. Keep each
    // pass within 8x and let librga select a compatible engine and DMA mapping.
    // Two reusable NV12 buffers also allow reductions requiring several passes.
    while (source_rect.width > g.rw * 8 || source_rect.height > g.rh * 8)
    {
        const int next_width = std::max(g.rw, ((source_rect.width + 15) / 16) * 2);
        const int next_height = std::max(g.rh, ((source_rect.height + 15) / 16) * 2);
        auto intermediate = resize_buffer(slot, next_width, next_height);
        const im_rect intermediate_rect{0, 0, next_width, next_height};
        auto status = improcess(src, intermediate, {}, source_rect, intermediate_rect, {}, -1,
                                nullptr, nullptr, IM_SYNC);
        require(status == IM_STATUS_SUCCESS,
                std::string("RGA staged resize: ") + imStrError_t(status));
        src = intermediate;
        source_rect = intermediate_rect;
        slot ^= 1;
    }
    // The final job converts NV12 to RGB directly in RKNN-bound DMA memory.
    auto status = improcess(src, dst, {}, source_rect, im_rect{g.px, g.py, g.rw, g.rh}, {}, -1,
                            nullptr, nullptr, IM_SYNC);
    require(status == IM_STATUS_SUCCESS,
            std::string("RGA preprocess: ") + imStrError_t(status));
    return g;
}
} // namespace nvr
