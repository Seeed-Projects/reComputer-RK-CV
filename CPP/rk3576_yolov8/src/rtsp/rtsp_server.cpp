#include "rk3576_demo/rtsp_server.hpp"

#include <cstring>
#include <iostream>

#include "mk_media.h"
#include "mk_common.h"
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_demo {

namespace {

bool g_env_initialized = false;
int g_server_port = 0;
int g_media_instance_count = 0;

}  // namespace

RtspServer::~RtspServer() {
  Stop();
}

bool RtspServer::Start(const AppConfig& config, int width, int height) {
  if (!g_env_initialized) {
    mk_config mk_cfg {};
    mk_cfg.thread_num = 0;
    mk_cfg.log_level = 2;
    mk_cfg.log_mask = LOG_CONSOLE;
    mk_env_init(&mk_cfg);
    g_env_initialized = true;
  }

  if (g_server_port == 0) {
    g_server_port = static_cast<int>(mk_rtsp_server_start(static_cast<std::uint16_t>(config.rtsp_port), 0));
    if (g_server_port == 0) {
      RKLOG_ERROR("APP") << "mk_rtsp_server_start failed\n";
      return false;
    }
  }
  port_ = g_server_port;

  app_ = config.rtsp_app;
  stream_ = config.rtsp_stream;
  media_ = mk_media_create("__defaultVhost__", app_.c_str(), stream_.c_str(), 0.0f, 0, 0);
  if (media_ == nullptr) {
    RKLOG_ERROR("APP") << "mk_media_create failed\n";
    return false;
  }

  if (!mk_media_init_video(media_, 0, width, height, static_cast<float>(config.fps), config.bitrate)) {
    RKLOG_ERROR("APP") << "mk_media_init_video failed\n";
    Stop();
    return false;
  }
  mk_media_init_complete(media_);
  ++g_media_instance_count;
  return true;
}

bool RtspServer::PushH264(const std::uint8_t* data, std::size_t size, std::uint64_t pts_ms) {
  if (media_ == nullptr || data == nullptr || size == 0) {
    return false;
  }
  return mk_media_input_h264(media_, data, static_cast<int>(size), pts_ms, pts_ms) != 0;
}

void RtspServer::Stop() {
  if (media_ != nullptr) {
    mk_media_release(media_);
    media_ = nullptr;
    if (g_media_instance_count > 0) {
      --g_media_instance_count;
    }
  }
  if (port_ != 0 && g_media_instance_count == 0) {
    mk_stop_all_server();
    g_server_port = 0;
  }
  port_ = 0;
}

std::string RtspServer::RtspUrl() const {
  if (port_ == 0) {
    return {};
  }
  return "rtsp://<board-ip>:" + std::to_string(port_) + "/" + app_ + "/" + stream_;
}

}  // namespace rk3576_demo
