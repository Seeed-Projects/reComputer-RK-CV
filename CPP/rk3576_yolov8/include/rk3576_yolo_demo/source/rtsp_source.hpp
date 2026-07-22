#pragma once

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "rk3576_yolo_demo/app/app_config_v2.hpp"
#include "rk3576_yolo_demo/source/i_source.hpp"

namespace rk3576_yolo_demo {

class RtspSource : public IInputSource {
 public:
  explicit RtspSource(const AppConfigV2& config);
  ~RtspSource() override;

  const char* Name() const override { return config_.source == SourceKind::kLocalVideo ? "LocalVideoSource" : "RtspSource"; }
  bool Open() override;
  void Close() override;
  SourceDescriptor Describe() const override;
  std::string LastError() const override { return last_error_; }
  bool SupportsPacketRead() const override { return true; }
  bool ReadPacket(CompressedPacket* packet) override;
  rk3576_demo::InputCodec OutputCodec() const override { return codec_; }
  int OutputWidth() const override { return width_; }
  int OutputHeight() const override { return height_; }

 private:
  struct LibavBackend;

  enum class BackendKind {
    kNone,
    kLibavformat,
    kFfmpegPipe,
  };

  bool ProbeStreamInfo();
  bool StartLibavInput();
  void StopLibavInput();
  bool StartFfmpegPipe();
  bool RestartPipe(bool silent_loop = false);
  bool ReadMjpegPacket(CompressedPacket* packet);
  bool ReadElementaryPacket(CompressedPacket* packet);
  bool ReadLibavPacket(CompressedPacket* packet);
  bool IsLocalVideoEof() const;
  bool IsRtspSource() const { return config_.source == SourceKind::kRtsp; }
  std::string EffectiveInputLocation() const;
  std::string DisplayInputLocation() const;
  std::string PrepareStderrPath(const char* stage);
  std::string ReadAndClearStderrFile();

  AppConfigV2 config_;
  BackendKind backend_ = BackendKind::kNone;
  std::unique_ptr<LibavBackend> libav_;
  std::FILE* pipe_ = nullptr;
  rk3576_demo::InputCodec codec_ = rk3576_demo::InputCodec::kH264;
  int width_ = 0;
  int height_ = 0;
  bool opened_ = false;
  std::vector<std::uint8_t> mjpeg_buffer_;
  std::string stderr_path_;
  std::string last_error_;
};

}  // namespace rk3576_yolo_demo
