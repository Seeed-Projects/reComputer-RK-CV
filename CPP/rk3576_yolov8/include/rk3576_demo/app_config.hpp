#pragma once

#include <getopt.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "rk3576_demo/media_types.hpp"

namespace rk3576_demo {

struct AppConfig {
  std::string device = "/dev/video0";
  InputCodec input_codec = InputCodec::kMjpeg;
  std::string rtsp_app = "live";
  std::string rtsp_stream = "camera";
  std::string dump_h264_path;
  std::string dump_mjpg_path;
  int camera_width = 1920;
  int camera_height = 1080;
  int output_width = 1280;
  int output_height = 720;
  int fps = 30;
  int rtsp_port = 8554;
  int bitrate = 4 * 1000 * 1000;
  int v4l2_buffer_count = 4;
  int frame_limit = 0;
  int perf_log_interval_ms = 1000;
  bool decode_only = false;
  bool dump_h264 = false;
  bool dump_mjpg = false;

  static void PrintUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --device <path>          V4L2 device path, default /dev/video0\n"
        << "  --camera-width <value>   Camera capture width, default 1920\n"
        << "  --camera-height <value>  Camera capture height, default 1080\n"
        << "  --output-width <value>   Encoded output width, default 1280\n"
        << "  --output-height <value>  Encoded output height, default 720\n"
        << "  --fps <value>            Capture/output fps, default 30\n"
        << "  --bitrate <value>        H.264 bitrate in bps, default 4000000\n"
        << "  --rtsp-port <value>      RTSP server port, default 8554\n"
        << "  --rtsp-app <value>       RTSP app name, default live\n"
        << "  --rtsp-stream <value>    RTSP stream name, default camera\n"
        << "  --dump-mjpg <path>       Dump captured MJPG packets to a local file\n"
        << "  --dump-h264 <path>       Dump encoded H.264 to a local file\n"
        << "  --frame-limit <value>    Stop after N frames, default 0 means unlimited\n"
        << "  --perf-interval-ms <v>   Performance/resource log interval, default 1000\n"
        << "  --help                   Show this help message\n";
  }

  static bool Parse(int argc, char** argv, AppConfig* config) {
    static option long_options[] = {
        {"device", required_argument, nullptr, 'd'},
        {"camera-width", required_argument, nullptr, 'w'},
        {"camera-height", required_argument, nullptr, 'h'},
        {"output-width", required_argument, nullptr, 'W'},
        {"output-height", required_argument, nullptr, 'H'},
        {"fps", required_argument, nullptr, 'f'},
        {"bitrate", required_argument, nullptr, 'b'},
        {"rtsp-port", required_argument, nullptr, 'p'},
        {"rtsp-app", required_argument, nullptr, 'a'},
        {"rtsp-stream", required_argument, nullptr, 's'},
        {"dump-mjpg", required_argument, nullptr, 'm'},
        {"dump-h264", required_argument, nullptr, 'o'},
        {"frame-limit", required_argument, nullptr, 'n'},
        {"perf-interval-ms", required_argument, nullptr, 'P'},
        {"help", no_argument, nullptr, '?'},
        {nullptr, 0, nullptr, 0},
    };

    optind = 1;
    while (true) {
      const int opt = getopt_long(argc, argv, "d:w:h:W:H:f:b:p:a:s:m:o:n:P:?", long_options, nullptr);
      if (opt == -1) {
        break;
      }
      switch (opt) {
        case 'd':
          config->device = optarg;
          break;
        case 'w':
          config->camera_width = std::atoi(optarg);
          break;
        case 'h':
          config->camera_height = std::atoi(optarg);
          break;
        case 'W':
          config->output_width = std::atoi(optarg);
          break;
        case 'H':
          config->output_height = std::atoi(optarg);
          break;
        case 'f':
          config->fps = std::atoi(optarg);
          break;
        case 'b':
          config->bitrate = std::atoi(optarg);
          break;
        case 'p':
          config->rtsp_port = std::atoi(optarg);
          break;
        case 'a':
          config->rtsp_app = optarg;
          break;
        case 's':
          config->rtsp_stream = optarg;
          break;
        case 'm':
          config->dump_mjpg = true;
          config->dump_mjpg_path = optarg;
          break;
        case 'o':
          config->dump_h264 = true;
          config->dump_h264_path = optarg;
          break;
        case 'n':
          config->frame_limit = std::atoi(optarg);
          break;
        case 'P':
          config->perf_log_interval_ms = std::atoi(optarg);
          break;
        case '?':
        default:
          PrintUsage(argv[0]);
          return false;
      }
    }

    return true;
  }
};

}  // namespace rk3576_demo
