#include "rk3576_demo/mpp_decoder.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_demo {

MppDecoder::~MppDecoder() {
  Close();
}

bool MppDecoder::Open(InputCodec codec, int fps, int width, int height, const FrameCallback& callback) {
  codec_ = codec;
  fps_ = fps;
  width_ = width;
  height_ = height;
  callback_ = callback;
  packet_count_ = 0;
  frame_count_ = 0;
  use_external_group_ = (codec != InputCodec::kMjpeg);

  MPP_RET ret = mpp_create(&ctx_, &mpi_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_create failed: " << ret << "\n";
    return false;
  }

  const MppCodingType coding_type = ToMppCodingType(codec);
  ret = mpp_init(ctx_, MPP_CTX_DEC, coding_type);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_init decoder failed: " << ret << "\n";
    Close();
    return false;
  }

  MppDecCfg cfg = nullptr;
  ret = mpp_dec_cfg_init(&cfg);
  if (ret == MPP_OK) {
    ret = mpi_->control(ctx_, MPP_DEC_GET_CFG, cfg);
    if (ret == MPP_OK) {
      const RK_U32 need_split = (codec == InputCodec::kMjpeg) ? 0 : 1;
      mpp_dec_cfg_set_u32(cfg, "base:split_parse", need_split);
      ret = mpi_->control(ctx_, MPP_DEC_SET_CFG, cfg);
    }
    mpp_dec_cfg_deinit(cfg);
  }

  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "Failed to configure decoder: " << ret << "\n";
    Close();
    return false;
  }

  if (codec == InputCodec::kMjpeg) {
    if (!SetupMjpegResources(0)) {
      Close();
      return false;
    }
  }

  RKLOG_INFO("APP") << "MPP decoder initialized, codec="
            << (codec == InputCodec::kMjpeg ? "MJPEG" : codec == InputCodec::kH264 ? "H264" : "H265")
            << ", split_parse=" << (codec == InputCodec::kMjpeg ? 0 : 1)
            << ", external_group=" << (use_external_group_ ? "on" : "off") << "\n";
  return true;
}

bool MppDecoder::SetupMjpegResources(std::size_t min_output_buffer_size) {
  if (codec_ != InputCodec::kMjpeg) {
    return true;
  }

  if (mjpeg_input_group_ == nullptr) {
    MPP_RET ret = mpp_buffer_group_get_internal(&mjpeg_input_group_, MPP_BUFFER_TYPE_DRM);
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "mpp_buffer_group_get_internal for MJPEG input failed: " << ret << "\n";
      return false;
    }
  }

  const std::size_t aligned_width = static_cast<std::size_t>(AlignTo(std::max(width_, 16), 16));
  const std::size_t aligned_height = static_cast<std::size_t>(AlignTo(std::max(height_, 16), 16));
  const std::size_t output_buffer_size = std::max(min_output_buffer_size, aligned_width * aligned_height * 4);

  if (frame_group_ != nullptr && mjpeg_output_frame_ != nullptr &&
      output_buffer_size <= mjpeg_output_buffer_size_) {
    return true;
  }

  if (frame_group_ == nullptr) {
    MPP_RET ret = mpp_buffer_group_get_internal(&frame_group_, MPP_BUFFER_TYPE_DRM);
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "mpp_buffer_group_get_internal for MJPEG frame failed: " << ret << "\n";
      return false;
    }
  } else if (mjpeg_output_frame_ != nullptr) {
    mpp_frame_deinit(&mjpeg_output_frame_);
    mjpeg_output_frame_ = nullptr;
    mpp_buffer_group_clear(frame_group_);
  }

  MPP_RET ret = mpp_buffer_group_limit_config(frame_group_, output_buffer_size, 4);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_buffer_group_limit_config for MJPEG failed: " << ret << "\n";
    return false;
  }

  MppBuffer output_buffer = nullptr;
  ret = mpp_buffer_get(frame_group_, &output_buffer, output_buffer_size);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_buffer_get for MJPEG output failed: " << ret << "\n";
    return false;
  }

  ret = mpp_frame_init(&mjpeg_output_frame_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_frame_init for MJPEG output failed: " << ret << "\n";
    mpp_buffer_put(output_buffer);
    return false;
  }

  mpp_frame_set_buffer(mjpeg_output_frame_, output_buffer);
  mpp_buffer_put(output_buffer);
  mjpeg_output_buffer_size_ = output_buffer_size;
  return true;
}

bool MppDecoder::CreateInputPacket(const std::uint8_t* data, std::size_t size, bool eos,
                                   std::uint64_t pts_ms, MppPacket* packet) {
  if (packet == nullptr) {
    return false;
  }

  *packet = nullptr;

  if (codec_ == InputCodec::kMjpeg) {
    if (!SetupMjpegResources(0)) {
      return false;
    }

    MppBuffer input_buffer = nullptr;
    MPP_RET ret = mpp_buffer_get(mjpeg_input_group_, &input_buffer, size);
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "mpp_buffer_get for MJPEG input failed: " << ret << "\n";
      return false;
    }

    ret = mpp_buffer_write(input_buffer, 0, const_cast<std::uint8_t*>(data), size);
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "mpp_buffer_write for MJPEG input failed: " << ret << "\n";
      mpp_buffer_put(input_buffer);
      return false;
    }

    ret = mpp_packet_init_with_buffer(packet, input_buffer);
    mpp_buffer_put(input_buffer);
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "mpp_packet_init_with_buffer failed: " << ret << "\n";
      return false;
    }

    mpp_packet_set_length(*packet, size);
    mpp_packet_set_pts(*packet, static_cast<RK_S64>(pts_ms));
    if (eos) {
      mpp_packet_set_eos(*packet);
    }

    MppMeta meta = mpp_packet_get_meta(*packet);
    if (meta == nullptr) {
      RKLOG_ERROR("APP") << "Failed to get MJPEG packet metadata\n";
      mpp_packet_deinit(packet);
      return false;
    }

    ret = mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, mjpeg_output_frame_);
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "mpp_meta_set_frame(KEY_OUTPUT_FRAME) failed: " << ret << "\n";
      mpp_packet_deinit(packet);
      return false;
    }

    return true;
  }

  MPP_RET ret = mpp_packet_init(packet, const_cast<std::uint8_t*>(data), size);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_packet_init failed: " << ret << "\n";
    return false;
  }

  mpp_packet_set_pos(*packet, const_cast<std::uint8_t*>(data));
  mpp_packet_set_length(*packet, size);
  mpp_packet_set_pts(*packet, static_cast<RK_S64>(pts_ms));
  if (eos) {
    mpp_packet_set_eos(*packet);
  }
  return true;
}

void MppDecoder::RecycleMjpegInputPacket(MppFrame frame, MppPacket* fallback_packet) {
  if (codec_ != InputCodec::kMjpeg || frame == nullptr) {
    return;
  }

  MppMeta meta = mpp_frame_get_meta(frame);
  if (meta == nullptr) {
    return;
  }

  MppPacket input_packet = nullptr;
  if (mpp_meta_get_packet(meta, KEY_INPUT_PACKET, &input_packet) != MPP_OK || input_packet == nullptr) {
    return;
  }

  if (fallback_packet != nullptr && *fallback_packet == input_packet) {
    *fallback_packet = nullptr;
  }
  mpp_packet_deinit(&input_packet);
}

bool MppDecoder::Decode(const std::uint8_t* data, std::size_t size, bool eos, std::uint64_t pts_ms) {
  if (ctx_ == nullptr || mpi_ == nullptr) {
    return false;
  }

  const auto decode_begin = std::chrono::steady_clock::now();
  MppPacket packet = nullptr;
  if (!CreateInputPacket(data, size, eos, pts_ms, &packet)) {
    return false;
  }
  ++packet_count_;
  if (packet_count_ <= 5 || (packet_count_ % 30) == 0) {
    RKLOG_INFO("APP") << "MPP decode input packet #" << packet_count_
              << " size=" << size
              << " pts_ms=" << pts_ms
              << " eos=" << (eos ? 1 : 0) << "\n";
  }

  bool packet_done = false;
  bool got_output_frame = false;
  MPP_RET ret = MPP_OK;
  std::uint64_t decode_put_us = 0;
  while (!packet_done) {
    const auto put_begin = std::chrono::steady_clock::now();
    ret = mpi_->decode_put_packet(ctx_, packet);
    decode_put_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - put_begin).count());
    if (ret == MPP_OK) {
      packet_done = true;
      if (packet_count_ <= 5 || (packet_count_ % 30) == 0) {
        RKLOG_INFO("APP") << "MPP decode_put_packet accepted packet #" << packet_count_ << "\n";
      }
    } else if (ret == MPP_ERR_BUFFER_FULL) {
      RKLOG_INFO("APP") << "MPP decode_put_packet buffer full for packet #" << packet_count_ << ", retrying\n";
      usleep(2000);
    } else {
      RKLOG_ERROR("APP") << "decode_put_packet failed: " << ret << "\n";
      mpp_packet_deinit(&packet);
      return false;
    }
  }

  int empty_retry_count = 0;
  const int max_empty_retry = (codec_ == InputCodec::kMjpeg) ? 20 : 1;
  std::uint64_t decode_wait_us = 0;
  while (true) {
    MppFrame frame = nullptr;
    const auto wait_begin = std::chrono::steady_clock::now();
    ret = mpi_->decode_get_frame(ctx_, &frame);
    decode_wait_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_begin).count());
    if (ret == MPP_ERR_TIMEOUT) {
      usleep(2000);
      continue;
    }
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "decode_get_frame failed: " << ret << "\n";
      break;
    }
    if (frame == nullptr) {
      if (empty_retry_count++ < max_empty_retry) {
        usleep(2000);
        continue;
      }
      if (packet_count_ <= 5 || (packet_count_ % 30) == 0) {
        RKLOG_INFO("APP") << "MPP decode_get_frame returned no frame for packet #" << packet_count_
                  << " after " << empty_retry_count << " retries\n";
      }
      break;
    }
    empty_retry_count = 0;

    if (mpp_frame_get_info_change(frame)) {
      const RK_U32 buf_size = mpp_frame_get_buf_size(frame);
      RKLOG_INFO("APP") << "MPP decoder info change: width=" << mpp_frame_get_width(frame)
                << " height=" << mpp_frame_get_height(frame)
                << " hor_stride=" << mpp_frame_get_hor_stride(frame)
                << " ver_stride=" << mpp_frame_get_ver_stride(frame)
                << " buf_size=" << buf_size
                << " fmt=" << mpp_frame_get_fmt(frame) << "\n";

      if (use_external_group_ && frame_group_ == nullptr) {
        ret = mpp_buffer_group_get_internal(&frame_group_, MPP_BUFFER_TYPE_DRM);
        if (ret != MPP_OK) {
          RKLOG_ERROR("APP") << "mpp_buffer_group_get_internal failed: " << ret << "\n";
          mpp_frame_deinit(&frame);
          mpp_packet_deinit(&packet);
          return false;
        }
        ret = mpi_->control(ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frame_group_);
        if (ret != MPP_OK) {
          RKLOG_ERROR("APP") << "MPP_DEC_SET_EXT_BUF_GROUP failed: " << ret << "\n";
          mpp_frame_deinit(&frame);
          mpp_packet_deinit(&packet);
          return false;
        }
      } else if (use_external_group_ && frame_group_ != nullptr) {
        mpp_buffer_group_clear(frame_group_);
      }

      if (use_external_group_ && frame_group_ != nullptr) {
        ret = mpp_buffer_group_limit_config(frame_group_, buf_size, 12);
        if (ret != MPP_OK) {
          RKLOG_ERROR("APP") << "mpp_buffer_group_limit_config failed: " << ret << "\n";
          mpp_frame_deinit(&frame);
          mpp_packet_deinit(&packet);
          return false;
        }
      }

      if (codec_ == InputCodec::kMjpeg) {
        width_ = static_cast<int>(mpp_frame_get_width(frame));
        height_ = static_cast<int>(mpp_frame_get_height(frame));
        if (!SetupMjpegResources(buf_size)) {
          RecycleMjpegInputPacket(frame, &packet);
          mpp_frame_deinit(&frame);
          mpp_packet_deinit(&packet);
          return false;
        }
      }

      ret = mpi_->control(ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
      if (ret != MPP_OK) {
        RKLOG_ERROR("APP") << "MPP_DEC_SET_INFO_CHANGE_READY failed: " << ret << "\n";
        RecycleMjpegInputPacket(frame, &packet);
        mpp_frame_deinit(&frame);
        mpp_packet_deinit(&packet);
        return false;
      }
      RKLOG_INFO("APP") << "MPP decoder info change acknowledged"
                << (use_external_group_ ? " with external DRM buffers" : " using internal buffers")
                << "\n";
      RecycleMjpegInputPacket(frame, &packet);
      mpp_frame_deinit(&frame);
      continue;
    }

    const RK_U32 errinfo = mpp_frame_get_errinfo(frame);
    const RK_U32 discard = mpp_frame_get_discard(frame);
    MppBuffer buffer = mpp_frame_get_buffer(frame);
    if (buffer == nullptr) {
      RKLOG_ERROR("APP") << "MPP decoder returned frame with null buffer, errinfo=" << errinfo
                << " discard=" << discard << "\n";
      RecycleMjpegInputPacket(frame, &packet);
      mpp_frame_deinit(&frame);
      continue;
    }

    ++frame_count_;
    got_output_frame = true;
    RKLOG_INFO("APP") << "MPP decoded frame #" << frame_count_
              << " width=" << mpp_frame_get_width(frame)
              << " height=" << mpp_frame_get_height(frame)
              << " hor_stride=" << mpp_frame_get_hor_stride(frame)
              << " ver_stride=" << mpp_frame_get_ver_stride(frame)
              << " fmt=" << mpp_frame_get_fmt(frame)
              << " fd=" << mpp_buffer_get_fd(buffer)
              << " errinfo=" << errinfo
              << " discard=" << discard << "\n";

    if (callback_ && errinfo == 0 && discard == 0) {
      DecodedFrame decoded {};
      decoded.width = static_cast<int>(mpp_frame_get_width(frame));
      decoded.height = static_cast<int>(mpp_frame_get_height(frame));
      decoded.hor_stride = static_cast<int>(mpp_frame_get_hor_stride(frame));
      decoded.ver_stride = static_cast<int>(mpp_frame_get_ver_stride(frame));
      decoded.format = static_cast<int>(mpp_frame_get_fmt(frame));
      decoded.fd = mpp_buffer_get_fd(buffer);
      decoded.virt_addr = mpp_buffer_get_ptr(buffer);
      const auto pts = mpp_frame_get_pts(frame);
      decoded.pts_ms = pts >= 0 ? static_cast<std::uint64_t>(pts) : pts_ms;
      decoded.decode_put_us = decode_put_us;
      decoded.decode_wait_us = decode_wait_us;
      decoded.decode_total_us = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - decode_begin)
              .count());
      callback_(decoded);
    } else if (errinfo != 0 || discard != 0) {
      RKLOG_ERROR("APP") << "Skipping decoded frame #" << frame_count_
                << " due to errinfo/discard flags\n";
    }

    const bool frame_eos = mpp_frame_get_eos(frame) != 0;
    RecycleMjpegInputPacket(frame, &packet);
    if (!(codec_ == InputCodec::kMjpeg && frame == mjpeg_output_frame_)) {
      mpp_frame_deinit(&frame);
    } else {
      frame = nullptr;
    }
    if (frame_eos) {
      break;
    }
  }

  if (packet != nullptr) {
    mpp_packet_deinit(&packet);
  }

  if (codec_ == InputCodec::kMjpeg && !got_output_frame && empty_retry_count > max_empty_retry) {
    RKLOG_ERROR("APP") << "MJPEG decoder accepted packet but produced no frame\n";
    return false;
  }

  return true;
}

void MppDecoder::Reset() {
  if (mpi_ != nullptr && ctx_ != nullptr) {
    mpi_->reset(ctx_);
  }
}

void MppDecoder::Close() {
  if (mjpeg_output_frame_ != nullptr) {
    mpp_frame_deinit(&mjpeg_output_frame_);
  }
  mjpeg_output_buffer_size_ = 0;
  if (mjpeg_input_group_ != nullptr) {
    mpp_buffer_group_put(mjpeg_input_group_);
    mjpeg_input_group_ = nullptr;
  }
  if (frame_group_ != nullptr) {
    mpp_buffer_group_put(frame_group_);
    frame_group_ = nullptr;
  }
  if (ctx_ != nullptr) {
    mpp_destroy(ctx_);
    ctx_ = nullptr;
  }
  mpi_ = nullptr;
}

MppCodingType MppDecoder::ToMppCodingType(InputCodec codec) const {
  switch (codec) {
    case InputCodec::kH264:
      return MPP_VIDEO_CodingAVC;
    case InputCodec::kH265:
      return MPP_VIDEO_CodingHEVC;
    case InputCodec::kMjpeg:
    default:
      return MPP_VIDEO_CodingMJPEG;
  }
}

}  // namespace rk3576_demo
