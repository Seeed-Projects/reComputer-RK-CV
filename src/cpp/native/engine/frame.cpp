#include "engine/frame.hpp"

namespace nvr
{
Frame::Frame(MppFrame frame) : value(frame)
{
}

Frame::~Frame()
{
    if (value)
    {
        mpp_frame_deinit(&value);
    }
}

Nv12ImageView Frame::image() const
{
    require(value && mpp_frame_get_fmt(value) == MPP_FMT_YUV420SP, "expected linear NV12 from MPP");
    MppBuffer buffer = mpp_frame_get_buffer(value);
    require(buffer, "missing MPP image buffer");
    return {int(mpp_frame_get_width(value)),
            int(mpp_frame_get_height(value)),
            int(mpp_frame_get_hor_stride(value)),
            int(mpp_frame_get_ver_stride(value)),
            mpp_buffer_get_fd(buffer),
            mpp_buffer_get_size(buffer)};
}
} // namespace nvr
