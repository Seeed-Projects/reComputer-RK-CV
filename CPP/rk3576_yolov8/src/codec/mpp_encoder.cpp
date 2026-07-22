#include "rk3576_demo/mpp_encoder.hpp"

#include <cstring>
#include <iostream>
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_demo {

namespace {

constexpr std::size_t kSz4K = 4096;

std::size_t GetMdInfoSize(int hor_stride, int ver_stride) {
  const auto aligned_w = AlignTo(hor_stride, 64);
  const auto aligned_h = AlignTo(ver_stride, 16);
  return (aligned_w >> 6) * (aligned_h >> 4) * 16;
}

}  // namespace

MppEncoder::~MppEncoder() {
  Close();
}

bool MppEncoder::Open(int width, int height, int fps, int bitrate) {
  width_ = width;
  height_ = height;
  hor_stride_ = AlignTo(width_, 16);
  ver_stride_ = AlignTo(height_, 16);
  frame_size_ = AlignTo(hor_stride_, 64) * AlignTo(ver_stride_, 64) * 3 / 2;
  mdinfo_size_ = GetMdInfoSize(hor_stride_, ver_stride_);

  MPP_RET ret = mpp_buffer_group_get_internal(&buffer_group_, MPP_BUFFER_TYPE_DRM);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_buffer_group_get_internal failed: " << ret << "\n";
    Close();
    return false;
  }

  ret = mpp_buffer_get(buffer_group_, &frame_buffer_, frame_size_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_buffer_get(frame) failed: " << ret << "\n";
    Close();
    return false;
  }
  ret = mpp_buffer_get(buffer_group_, &packet_buffer_, frame_size_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_buffer_get(packet) failed: " << ret << "\n";
    Close();
    return false;
  }
  ret = mpp_buffer_get(buffer_group_, &motion_buffer_, mdinfo_size_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_buffer_get(motion) failed: " << ret << "\n";
    Close();
    return false;
  }

  ret = mpp_create(&ctx_, &mpi_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_create encoder failed: " << ret << "\n";
    Close();
    return false;
  }

  MppPollType timeout = MPP_POLL_BLOCK;
  mpi_->control(ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout);

  ret = mpp_init(ctx_, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_init encoder failed: " << ret << "\n";
    Close();
    return false;
  }

  if (!SetupConfig(fps, bitrate)) {
    Close();
    return false;
  }

  return true;
}

bool MppEncoder::SetupConfig(int fps, int bitrate) {
  MPP_RET ret = mpp_enc_cfg_init(&cfg_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "mpp_enc_cfg_init failed: " << ret << "\n";
    return false;
  }

  mpp_enc_cfg_set_s32(cfg_, "prep:width", width_);
  mpp_enc_cfg_set_s32(cfg_, "prep:height", height_);
  mpp_enc_cfg_set_s32(cfg_, "prep:hor_stride", hor_stride_);
  mpp_enc_cfg_set_s32(cfg_, "prep:ver_stride", ver_stride_);
  mpp_enc_cfg_set_s32(cfg_, "prep:format", MPP_FMT_YUV420SP);

  mpp_enc_cfg_set_s32(cfg_, "rc:mode", MPP_ENC_RC_MODE_CBR);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_flex", 0);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_num", fps);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_denorm", 1);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_flex", 0);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_num", fps);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_denorm", 1);
  mpp_enc_cfg_set_s32(cfg_, "rc:gop", fps * 2);
  mpp_enc_cfg_set_s32(cfg_, "rc:bps_target", bitrate);
  mpp_enc_cfg_set_s32(cfg_, "rc:bps_max", bitrate * 17 / 16);
  mpp_enc_cfg_set_s32(cfg_, "rc:bps_min", bitrate * 15 / 16);
  mpp_enc_cfg_set_u32(cfg_, "rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED);
  mpp_enc_cfg_set_s32(cfg_, "rc:qp_init", -1);
  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max", 51);
  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min", 10);
  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max_i", 51);
  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min_i", 10);
  mpp_enc_cfg_set_s32(cfg_, "rc:qp_ip", 2);

  mpp_enc_cfg_set_s32(cfg_, "codec:type", MPP_VIDEO_CodingAVC);
  mpp_enc_cfg_set_s32(cfg_, "h264:profile", 100);
  mpp_enc_cfg_set_s32(cfg_, "h264:level", 40);
  mpp_enc_cfg_set_s32(cfg_, "h264:cabac_en", 1);
  mpp_enc_cfg_set_s32(cfg_, "h264:cabac_idc", 0);
  mpp_enc_cfg_set_s32(cfg_, "h264:trans8x8", 1);

  ret = mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg_);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "MPP_ENC_SET_CFG failed: " << ret << "\n";
    return false;
  }

  MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
  ret = mpi_->control(ctx_, MPP_ENC_SET_HEADER_MODE, &header_mode);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "MPP_ENC_SET_HEADER_MODE failed: " << ret << "\n";
    return false;
  }
  return true;
}

bool MppEncoder::GetHeader(std::vector<std::uint8_t>* packet_out) {
  if (packet_out == nullptr) {
    return false;
  }

  packet_out->clear();
  MppPacket packet = nullptr;
  mpp_packet_init_with_buffer(&packet, packet_buffer_);
  mpp_packet_set_length(packet, 0);

  const MPP_RET ret = mpi_->control(ctx_, MPP_ENC_GET_HDR_SYNC, packet);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "MPP_ENC_GET_HDR_SYNC failed: " << ret << "\n";
    mpp_packet_deinit(&packet);
    return false;
  }

  const void* ptr = mpp_packet_get_pos(packet);
  const auto len = mpp_packet_get_length(packet);
  packet_out->assign(static_cast<const std::uint8_t*>(ptr),
                     static_cast<const std::uint8_t*>(ptr) + len);
  mpp_packet_deinit(&packet);
  return !packet_out->empty();
}

bool MppEncoder::EncodeCurrentFrame(std::uint64_t pts_ms, std::vector<std::uint8_t>* packet_out) {
  if (packet_out == nullptr) {
    return false;
  }

  packet_out->clear();

  MppFrame frame = nullptr;
  if (mpp_frame_init(&frame) != MPP_OK) {
    return false;
  }

  mpp_frame_set_width(frame, width_);
  mpp_frame_set_height(frame, height_);
  mpp_frame_set_hor_stride(frame, hor_stride_);
  mpp_frame_set_ver_stride(frame, ver_stride_);
  mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
  mpp_frame_set_buffer(frame, frame_buffer_);
  mpp_frame_set_pts(frame, static_cast<RK_S64>(pts_ms));

  MppMeta meta = mpp_frame_get_meta(frame);
  MppPacket packet = nullptr;
  mpp_packet_init_with_buffer(&packet, packet_buffer_);
  mpp_packet_set_length(packet, 0);
  mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);
  mpp_meta_set_buffer(meta, KEY_MOTION_INFO, motion_buffer_);

  MPP_RET ret = mpi_->encode_put_frame(ctx_, frame);
  mpp_frame_deinit(&frame);
  if (ret != MPP_OK) {
    RKLOG_ERROR("APP") << "encode_put_frame failed: " << ret << "\n";
    mpp_packet_deinit(&packet);
    return false;
  }

  bool eoi = false;
  while (!eoi) {
    ret = mpi_->encode_get_packet(ctx_, &packet);
    if (ret != MPP_OK) {
      RKLOG_ERROR("APP") << "encode_get_packet failed: " << ret << "\n";
      return false;
    }
    if (packet == nullptr) {
      continue;
    }

    const auto* ptr = static_cast<const std::uint8_t*>(mpp_packet_get_pos(packet));
    const auto len = mpp_packet_get_length(packet);
    packet_out->insert(packet_out->end(), ptr, ptr + len);
    if (mpp_packet_is_partition(packet)) {
      eoi = mpp_packet_is_eoi(packet);
    } else {
      eoi = true;
    }
    mpp_packet_deinit(&packet);
  }

  return !packet_out->empty();
}

int MppEncoder::input_fd() const {
  return frame_buffer_ ? mpp_buffer_get_fd(frame_buffer_) : -1;
}

void* MppEncoder::input_addr() const {
  return frame_buffer_ ? mpp_buffer_get_ptr(frame_buffer_) : nullptr;
}

void MppEncoder::Close() {
  if (ctx_ != nullptr) {
    mpp_destroy(ctx_);
    ctx_ = nullptr;
  }
  mpi_ = nullptr;

  if (cfg_ != nullptr) {
    mpp_enc_cfg_deinit(cfg_);
    cfg_ = nullptr;
  }
  if (frame_buffer_ != nullptr) {
    mpp_buffer_put(frame_buffer_);
    frame_buffer_ = nullptr;
  }
  if (packet_buffer_ != nullptr) {
    mpp_buffer_put(packet_buffer_);
    packet_buffer_ = nullptr;
  }
  if (motion_buffer_ != nullptr) {
    mpp_buffer_put(motion_buffer_);
    motion_buffer_ = nullptr;
  }
  if (buffer_group_ != nullptr) {
    mpp_buffer_group_put(buffer_group_);
    buffer_group_ = nullptr;
  }

  width_ = 0;
  height_ = 0;
  hor_stride_ = 0;
  ver_stride_ = 0;
  frame_size_ = 0;
  mdinfo_size_ = 0;
}

}  // namespace rk3576_demo
