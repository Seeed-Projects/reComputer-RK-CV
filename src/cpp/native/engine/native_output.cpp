#include "engine/native_output.hpp"
#include "engine/common.hpp"

namespace nvr
{
NativeOutput::NativeOutput(rknn_context context, const rknn_tensor_attr &logical)
    : context_(context)
{
    if (logical.n_dims != 4 || logical.dims[0] != 1 || logical.fmt != RKNN_TENSOR_NCHW)
    {
        return;
    }

    rknn_tensor_attr native{};
    native.index = logical.index;
    if (rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native, sizeof(native)) != 0)
    {
        return;
    }
    size_t scalar_bytes = 0;
    switch (native.type)
    {
    case RKNN_TENSOR_FLOAT16:
        view_.type = TensorView::Type::Float16;
        scalar_bytes = 2;
        break;
    case RKNN_TENSOR_FLOAT32:
        view_.type = TensorView::Type::Float32;
        scalar_bytes = 4;
        break;
    case RKNN_TENSOR_INT8:
        view_.type = TensorView::Type::Int8;
        scalar_bytes = 1;
        break;
    case RKNN_TENSOR_UINT8:
        view_.type = TensorView::Type::Uint8;
        scalar_bytes = 1;
        break;
    case RKNN_TENSOR_INT32:
        view_.type = TensorView::Type::Int32;
        scalar_bytes = 4;
        break;
    default:
        return;
    }
    const size_t channels = logical.dims[1];
    view_.height = logical.dims[2];
    view_.width = logical.dims[3];
    view_.count = logical.n_elems;
    if (!channels || !view_.height || !view_.width || native.dims[0] != 1)
    {
        return;
    }
    size_t native_width = 0;
    if (native.fmt == RKNN_TENSOR_NC1HWC2 && native.n_dims == 5 && native.dims[2] == view_.height &&
        native.dims[1] * native.dims[4] >= channels)
    {
        view_.channel_block = native.dims[4];
        native_width = native.dims[3];
    }
    else if (native.fmt == RKNN_TENSOR_NHWC && native.n_dims == 4 &&
             native.dims[1] == view_.height && native.dims[3] >= channels)
    {
        view_.channel_block = native.dims[3];
        native_width = native.dims[2];
    }
    else if (native.fmt == RKNN_TENSOR_NCHW && native.n_dims == 4 &&
             native.dims[2] == view_.height && native.dims[1] >= channels)
    {
        view_.channel_block = 1;
        native_width = native.dims[3];
    }
    else
    {
        return;
    }
    view_.row_stride = native.w_stride ? native.w_stride : native_width;
    if (!view_.channel_block || view_.row_stride < view_.width)
    {
        return;
    }
    const size_t last_channel = channels - 1;
    const size_t last =
        ((last_channel / view_.channel_block * view_.height + view_.height - 1) * view_.row_stride +
         view_.width - 1) *
            view_.channel_block +
        last_channel % view_.channel_block;
    if (native.size_with_stride / scalar_bytes <= last)
    {
        return;
    }
    if (native.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        if (!(native.scale > 0))
        {
            return;
        }
        view_.scale = native.scale;
        view_.zero_point = native.zp;
    }
    memory_ = rknn_create_mem(context_, native.size_with_stride);
    require(memory_, "Allocate native output memory");
    if (rknn_set_io_mem(context_, memory_, &native) != 0)
    {
        rknn_destroy_mem(context_, memory_);
        memory_ = nullptr;
    }
}

NativeOutput::~NativeOutput()
{
    if (memory_)
    {
        rknn_destroy_mem(context_, memory_);
    }
}

TensorView NativeOutput::view() const
{
    require(bound(), "Native output is not bound");
    auto result = view_;
    result.data = memory_->virt_addr;
    return result;
}

void NativeOutput::sync() const
{
    require(bound() && rknn_mem_sync(context_, memory_, RKNN_MEMORY_SYNC_FROM_DEVICE) == 0, "Sync native output memory");
}
} // namespace nvr
