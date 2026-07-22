#include "rk3576_demo/rga_processor.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "im2d.hpp"
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_demo {

namespace {

int ClampToByte(int value) {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return value;
}

int BlendChannel(int src, int dst, int alpha) {
  return (src * alpha + dst * (255 - alpha) + 127) / 255;
}

void RgbToYuv(std::uint8_t r, std::uint8_t g, std::uint8_t b, int* y, int* u, int* v) {
  *y = ClampToByte(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
  *u = ClampToByte(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
  *v = ClampToByte(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
}

}  // namespace

bool RgaProcessor::ProcessFrame(const DecodedFrame& frame, const WatermarkImage& watermark, MppEncoder* encoder,
                                rk3576_yolo_demo::ResizeMode resize_mode,
                                rk3576_yolo_demo::FrameTransformInfo* transform) {
  if (encoder == nullptr || encoder->input_fd() < 0 || frame.fd < 0) {
    return false;
  }

  const int src_format = ToRgaFormat(frame.format);
  if (src_format < 0) {
    RKLOG_ERROR("APP") << "Unsupported decoder frame format: " << frame.format << "\n";
    return false;
  }

  rga_buffer_handle_t src_handle = importbuffer_fd(frame.fd, frame.width, frame.height, src_format);
  if (src_handle == 0) {
    RKLOG_ERROR("APP") << "importbuffer_fd for decoder frame failed: " << imStrError() << "\n";
    return false;
  }
  rga_buffer_handle_t dst_handle =
      importbuffer_fd(encoder->input_fd(), encoder->width(), encoder->height(), RK_FORMAT_YCbCr_420_SP);
  if (dst_handle == 0) {
    RKLOG_ERROR("APP") << "importbuffer_fd for encoder frame failed: " << imStrError() << "\n";
    releasebuffer_handle(src_handle);
    return false;
  }

  rga_buffer_t src = wrapbuffer_handle(src_handle, frame.width, frame.height, src_format,
                                       frame.hor_stride, frame.ver_stride);
  rga_buffer_t dst = wrapbuffer_handle(dst_handle, encoder->width(), encoder->height(),
                                       RK_FORMAT_YCbCr_420_SP, encoder->hor_stride(), encoder->ver_stride());

  const rk3576_yolo_demo::FrameTransformInfo transform_info =
      rk3576_yolo_demo::ComputeFrameTransform(resize_mode, frame.width, frame.height, encoder->width(), encoder->height());
  if (transform_info.valid && transform != nullptr) {
    *transform = transform_info;
  }

  if (resize_mode == rk3576_yolo_demo::ResizeMode::kLetterbox || resize_mode == rk3576_yolo_demo::ResizeMode::kCenterCrop) {
    auto* dst_base = static_cast<std::uint8_t*>(encoder->input_addr());
    if (dst_base != nullptr) {
      std::memset(dst_base, 0, static_cast<std::size_t>(encoder->hor_stride() * encoder->ver_stride()));
      std::memset(dst_base + static_cast<std::size_t>(encoder->hor_stride() * encoder->ver_stride()), 128,
                  static_cast<std::size_t>(encoder->hor_stride() * encoder->ver_stride() / 2));
    }
  }

  IM_STATUS status = IM_STATUS_SUCCESS;
  if (resize_mode == rk3576_yolo_demo::ResizeMode::kStretch &&
      frame.width == encoder->width() && frame.height == encoder->height() &&
      src_format == RK_FORMAT_YCbCr_420_SP) {
    status = imcopy(src, dst);
  } else {
    im_rect src_rect = {transform_info.src_x, transform_info.src_y, transform_info.src_width, transform_info.src_height};
    im_rect dst_rect = {transform_info.dst_x, transform_info.dst_y, transform_info.dst_width, transform_info.dst_height};
    if (transform_info.valid) {
      status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
    } else {
      status = imresize(src, dst);
    }
  }
  if (status != IM_STATUS_SUCCESS) {
    RKLOG_ERROR("APP") << "RGA resize/copy failed: " << imStrError(status) << "\n";
    releasebuffer_handle(dst_handle);
    releasebuffer_handle(src_handle);
    return false;
  }

  releasebuffer_handle(dst_handle);
  releasebuffer_handle(src_handle);

  if (watermark.rgba.empty()) {
    return true;
  }

  if (!BlendWatermarkOnNv12(watermark, encoder)) {
    RKLOG_ERROR("APP") << "Software watermark blend on NV12 failed\n";
    return false;
  }
  return true;
}

int RgaProcessor::ToRgaFormat(int mpp_format) const {
  switch (mpp_format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:
      return RK_FORMAT_YCbCr_420_SP;
    case MPP_FMT_YUV420SP_VU:
      return RK_FORMAT_YCrCb_420_SP;
    case MPP_FMT_YUV422SP:
      return RK_FORMAT_YCbCr_422_SP;
    case MPP_FMT_YUV422SP_VU:
      return RK_FORMAT_YCrCb_422_SP;
    default:
      return -1;
  }
}

bool RgaProcessor::BlendWatermarkOnNv12(const WatermarkImage& watermark, MppEncoder* encoder) const {
  if (encoder == nullptr || encoder->input_addr() == nullptr) {
    return false;
  }
  if (watermark.width <= 0 || watermark.height <= 0 || watermark.rgba.empty()) {
    return true;
  }

  auto* base = static_cast<std::uint8_t*>(encoder->input_addr());
  const int frame_width = encoder->width();
  const int frame_height = encoder->height();
  const int y_stride = encoder->hor_stride();
  const int uv_stride = encoder->hor_stride();
  const int ver_stride = encoder->ver_stride();

  auto* y_plane = base;
  auto* uv_plane = base + static_cast<std::size_t>(y_stride * ver_stride);

  const int overlay_x0 = std::max(0, watermark.x);
  const int overlay_y0 = std::max(0, watermark.y);
  const int overlay_x1 = std::min(frame_width, watermark.x + watermark.width);
  const int overlay_y1 = std::min(frame_height, watermark.y + watermark.height);
  if (overlay_x0 >= overlay_x1 || overlay_y0 >= overlay_y1) {
    return true;
  }

  for (int y = overlay_y0; y < overlay_y1; ++y) {
    const int wm_y = y - watermark.y;
    for (int x = overlay_x0; x < overlay_x1; ++x) {
      const int wm_x = x - watermark.x;
      const std::size_t rgba_index =
          static_cast<std::size_t>(wm_y * watermark.width + wm_x) * 4;
      const int r = watermark.rgba[rgba_index + 0];
      const int g = watermark.rgba[rgba_index + 1];
      const int b = watermark.rgba[rgba_index + 2];
      const int a = watermark.rgba[rgba_index + 3];
      if (a == 0) {
        continue;
      }

      int y_value = 0;
      int u_value = 0;
      int v_value = 0;
      RgbToYuv(static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b),
               &y_value, &u_value, &v_value);

      const std::size_t y_index = static_cast<std::size_t>(y * y_stride + x);
      y_plane[y_index] = static_cast<std::uint8_t>(BlendChannel(y_value, y_plane[y_index], a));
    }
  }

  const int uv_x0 = overlay_x0 & ~1;
  const int uv_y0 = overlay_y0 & ~1;
  const int uv_x1 = overlay_x1 & ~1;
  const int uv_y1 = overlay_y1 & ~1;
  for (int y = uv_y0; y < uv_y1; y += 2) {
    const int uv_row = y / 2;
    for (int x = uv_x0; x < uv_x1; x += 2) {
      int r_sum = 0;
      int g_sum = 0;
      int b_sum = 0;
      int a_sum = 0;
      int sample_count = 0;

      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const int px = x + dx;
          const int py = y + dy;
          if (px < overlay_x0 || px >= overlay_x1 || py < overlay_y0 || py >= overlay_y1) {
            continue;
          }
          const int wm_x = px - watermark.x;
          const int wm_y = py - watermark.y;
          const std::size_t rgba_index =
              static_cast<std::size_t>(wm_y * watermark.width + wm_x) * 4;
          r_sum += watermark.rgba[rgba_index + 0];
          g_sum += watermark.rgba[rgba_index + 1];
          b_sum += watermark.rgba[rgba_index + 2];
          a_sum += watermark.rgba[rgba_index + 3];
          ++sample_count;
        }
      }

      if (sample_count == 0 || a_sum == 0) {
        continue;
      }

      const int r_avg = r_sum / sample_count;
      const int g_avg = g_sum / sample_count;
      const int b_avg = b_sum / sample_count;
      const int a_avg = a_sum / sample_count;
      int y_value = 0;
      int u_value = 0;
      int v_value = 0;
      RgbToYuv(static_cast<std::uint8_t>(r_avg), static_cast<std::uint8_t>(g_avg),
               static_cast<std::uint8_t>(b_avg), &y_value, &u_value, &v_value);

      const std::size_t uv_index = static_cast<std::size_t>(uv_row * uv_stride + x);
      uv_plane[uv_index + 0] = static_cast<std::uint8_t>(BlendChannel(u_value, uv_plane[uv_index + 0], a_avg));
      uv_plane[uv_index + 1] = static_cast<std::uint8_t>(BlendChannel(v_value, uv_plane[uv_index + 1], a_avg));
    }
  }

  return true;
}

}  // namespace rk3576_demo
