#pragma once

#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "rk3576_yolo_demo/common/runtime_types.hpp"

namespace rk3576_yolo_demo {

struct AppConfigV2 {
  SourceKind source = SourceKind::kV4L2;
  std::string device = "/dev/video0";
  std::string input_rtsp_url;
  std::string local_video_path;
  std::string rtsp_username;
  std::string rtsp_password;
  std::string model_path;
  std::string rtsp_app = "live";
  std::string rtsp_stream = "camera";
  std::string sub_stream = "camera_sub";
  std::string ai_stream = "camera_ai";
  std::string dump_ai_input_dir;
  std::string enabled_streams = "main,sub,ai";
  int camera_width = 1920;
  int camera_height = 1080;
  int output_width = 1280;
  int output_height = 720;
  int sub_width = 640;
  int sub_height = 480;
  int ai_width = 0;
  int ai_height = 0;
  int fps = 30;
  int bitrate = 4 * 1000 * 1000;
  int rtsp_port = 8554;
  int perf_log_interval_ms = 1000;
  int rtsp_timeout_ms = 5000;
  int rtsp_reconnect_interval_ms = 1000;
  int frame_limit = 0;
  int dump_ai_input_every = 0;
  bool dump_mjpg = false;
  bool dump_h264 = false;
  std::string dump_mjpg_path;
  std::string dump_h264_path;
  bool enable_main_stream = true;
  bool enable_sub_stream = true;
  bool enable_ai_stream = true;
  bool detail_info = false;
  bool show_help = false;

  static void PrintUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --source <v4l2|rtsp|localvideo>  输入源类型，默认 v4l2\n"
        << "  --device <path>                  V4L2 设备节点，默认 /dev/video0\n"
        << "  --rtsp-input-url <url>           RTSP 输入地址\n"
        << "  --rtsp-username <value>          RTSP 用户名，可选\n"
        << "  --rtsp-password <value>          RTSP 密码，可选\n"
        << "  --localvideo <path>              本地视频文件路径，处理完毕后循环播放\n"
        << "  --model <path>                   RKNN 模型路径\n"
        << "  --camera-width <value>           输入宽度，默认 1920\n"
        << "  --camera-height <value>          输入高度，默认 1080\n"
        << "  --output-width <value>           主码流宽度，默认 1280\n"
        << "  --output-height <value>          主码流高度，默认 720\n"
        << "  --sub-width <value>              副码流宽度，默认 640\n"
        << "  --sub-height <value>             副码流高度，默认 480\n"
        << "  --ai-width <value>               AI 调试流宽度，默认跟随输入宽度\n"
        << "  --ai-height <value>              AI 调试流高度，默认跟随输入高度\n"
        << "  --fps <value>                    编码目标帧率，默认 30\n"
        << "  --bitrate <value>                主码流码率，默认 4000000\n"
        << "  --rtsp-port <value>              本地 RTSP 服务端口，默认 8554\n"
        << "  --rtsp-app <value>               RTSP app 名称，默认 live\n"
        << "  --rtsp-stream <value>            主码流名称，默认 camera\n"
        << "  --sub-stream <value>             副码流名称，默认 camera_sub\n"
        << "  --ai-stream <value>              AI 调试流名称，默认 camera_ai\n"
        << "  --streams <items>                启用的输出流组合，逗号分隔: main,sub,ai\n"
        << "  --dump-ai-input-dir <path>       导出送入 NPU 的 640x640 RGB 输入图目录\n"
        << "  --dump-ai-input-every <value>    每隔 N 帧导出一张 AI 输入图，默认 0(关闭)\n"
        << "  --rtsp-timeout-ms <value>        RTSP 读超时，默认 5000\n"
        << "  --rtsp-reconnect-ms <value>      RTSP 重连间隔，默认 1000\n"
        << "  --perf-interval-ms <value>       PERF 输出周期，默认 1000\n"
        << "  --frame-limit <value>            处理 N 帧后退出，0 为不限制\n"
        << "  --detail-info                    打印详细日志\n"
        << "  --help                           显示帮助\n"
        << "\nExamples:\n"
        << "  " << argv0 << " --source v4l2 --device /dev/video0 --model model/yolov8n_rk3576.rknn\n"
        << "  " << argv0 << " --source rtsp --rtsp-input-url rtsp://192.168.1.10/live --rtsp-username admin --rtsp-password 123456\n"
        << "  " << argv0 << " --source localvideo --localvideo ./videos/demo.mp4 --model model/yolov8n_rk3576.rknn\n"
        << "  " << argv0 << " --source rtsp --rtsp-input-url rtsp://192.168.1.10/live --streams ai\n"
        << "  " << argv0 << " --source rtsp --rtsp-input-url rtsp://192.168.1.10/live --streams main,ai\n";
  }

  static bool ParseEnabledStreams(const std::string& value, AppConfigV2* config) {
    if (config == nullptr) {
      return false;
    }

    config->enable_main_stream = false;
    config->enable_sub_stream = false;
    config->enable_ai_stream = false;

    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
      token.erase(std::remove_if(token.begin(), token.end(),
                                 [](unsigned char ch) { return std::isspace(ch) != 0; }),
                  token.end());
      std::transform(token.begin(), token.end(), token.begin(),
                     [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (token.empty()) {
        continue;
      }
      if (token == "main") {
        config->enable_main_stream = true;
      } else if (token == "sub") {
        config->enable_sub_stream = true;
      } else if (token == "ai") {
        config->enable_ai_stream = true;
      } else {
        std::cerr << "Unsupported stream role in --streams: " << token
                  << " (allowed: main,sub,ai)\n";
        return false;
      }
    }

    config->enabled_streams = value;
    return config->enable_main_stream || config->enable_sub_stream || config->enable_ai_stream;
  }

  std::string EnabledStreamsSummary() const {
    std::string result;
    if (enable_main_stream) {
      result += "main";
    }
    if (enable_sub_stream) {
      if (!result.empty()) {
        result += ",";
      }
      result += "sub";
    }
    if (enable_ai_stream) {
      if (!result.empty()) {
        result += ",";
      }
      result += "ai";
    }
    return result;
  }

  static bool Parse(int argc, char** argv, AppConfigV2* config) {
    if (config == nullptr) {
      return false;
    }

    static option long_options[] = {
        {"source", required_argument, nullptr, 'S'},
        {"device", required_argument, nullptr, 'd'},
        {"rtsp-input-url", required_argument, nullptr, 'u'},
        {"localvideo", required_argument, nullptr, 'l'},
        {"rtsp-username", required_argument, nullptr, 'U'},
        {"rtsp-password", required_argument, nullptr, 'A'},
        {"model", required_argument, nullptr, 'm'},
        {"camera-width", required_argument, nullptr, 'w'},
        {"camera-height", required_argument, nullptr, 'h'},
        {"output-width", required_argument, nullptr, 'W'},
        {"output-height", required_argument, nullptr, 'H'},
        {"sub-width", required_argument, nullptr, 'x'},
        {"sub-height", required_argument, nullptr, 'y'},
        {"ai-width", required_argument, nullptr, 'X'},
        {"ai-height", required_argument, nullptr, 'Y'},
        {"fps", required_argument, nullptr, 'f'},
        {"bitrate", required_argument, nullptr, 'b'},
        {"rtsp-port", required_argument, nullptr, 'p'},
        {"rtsp-app", required_argument, nullptr, 'a'},
        {"rtsp-stream", required_argument, nullptr, 's'},
        {"sub-stream", required_argument, nullptr, 'q'},
        {"ai-stream", required_argument, nullptr, 'i'},
        {"streams", required_argument, nullptr, 'k'},
        {"dump-ai-input-dir", required_argument, nullptr, 'I'},
        {"dump-ai-input-every", required_argument, nullptr, 'e'},
        {"rtsp-timeout-ms", required_argument, nullptr, 't'},
        {"rtsp-reconnect-ms", required_argument, nullptr, 'r'},
        {"perf-interval-ms", required_argument, nullptr, 'P'},
        {"frame-limit", required_argument, nullptr, 'n'},
        {"dump-mjpg", required_argument, nullptr, 'j'},
        {"dump-h264", required_argument, nullptr, 'o'},
        {"detail-info", no_argument, nullptr, 'D'},
        {"help", no_argument, nullptr, 'Z'},
        {nullptr, 0, nullptr, 0},
    };

    optind = 1;
    while (true) {
      const int opt = getopt_long(argc, argv, "S:d:u:l:U:A:m:w:h:W:H:x:y:X:Y:f:b:p:a:s:q:i:k:I:e:t:r:P:n:j:o:DZ",
                                  long_options, nullptr);
      if (opt == -1) {
        break;
      }
      switch (opt) {
        case 'S':
          if (std::string(optarg) == "v4l2") {
            config->source = SourceKind::kV4L2;
          } else if (std::string(optarg) == "rtsp") {
            config->source = SourceKind::kRtsp;
          } else if (std::string(optarg) == "localvideo") {
            config->source = SourceKind::kLocalVideo;
          } else {
            std::cerr << "Unsupported source type: " << optarg << "\n";
            return false;
          }
          break;
        case 'd':
          config->device = optarg;
          break;
        case 'u':
          config->input_rtsp_url = optarg;
          break;
        case 'l':
          config->source = SourceKind::kLocalVideo;
          config->local_video_path = optarg;
          break;
        case 'U':
          config->rtsp_username = optarg;
          break;
        case 'A':
          config->rtsp_password = optarg;
          break;
        case 'm':
          config->model_path = optarg;
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
        case 'x':
          config->sub_width = std::atoi(optarg);
          break;
        case 'y':
          config->sub_height = std::atoi(optarg);
          break;
        case 'X':
          config->ai_width = std::atoi(optarg);
          break;
        case 'Y':
          config->ai_height = std::atoi(optarg);
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
        case 'q':
          config->sub_stream = optarg;
          break;
        case 'i':
          config->ai_stream = optarg;
          break;
        case 'k':
          if (!ParseEnabledStreams(optarg, config)) {
            std::cerr << "--streams requires at least one valid role from: main,sub,ai\n";
            return false;
          }
          break;
        case 'I':
          config->dump_ai_input_dir = optarg;
          break;
        case 'e':
          config->dump_ai_input_every = std::atoi(optarg);
          break;
        case 't':
          config->rtsp_timeout_ms = std::atoi(optarg);
          break;
        case 'r':
          config->rtsp_reconnect_interval_ms = std::atoi(optarg);
          break;
        case 'P':
          config->perf_log_interval_ms = std::atoi(optarg);
          break;
        case 'n':
          config->frame_limit = std::atoi(optarg);
          break;
        case 'j':
          config->dump_mjpg = true;
          config->dump_mjpg_path = optarg;
          break;
        case 'o':
          config->dump_h264 = true;
          config->dump_h264_path = optarg;
          break;
        case 'D':
          config->detail_info = true;
          break;
        case 'Z':
          config->show_help = true;
          PrintUsage(argv[0]);
          return true;
        case '?':
        default:
          return false;
      }
    }

    if (config->source == SourceKind::kRtsp && config->input_rtsp_url.empty()) {
      std::cerr << "--source rtsp requires --rtsp-input-url\n";
      return false;
    }
    if (config->source == SourceKind::kLocalVideo && config->local_video_path.empty()) {
      std::cerr << "--source localvideo requires --localvideo\n";
      return false;
    }
    if (!config->enable_main_stream && !config->enable_sub_stream && !config->enable_ai_stream) {
      std::cerr << "--streams requires at least one enabled role: main,sub,ai\n";
      return false;
    }
    if (!config->dump_ai_input_dir.empty() && config->dump_ai_input_every <= 0) {
      config->dump_ai_input_every = 1;
    }
    if (config->dump_ai_input_every < 0) {
      std::cerr << "--dump-ai-input-every must be >= 0\n";
      return false;
    }

    return true;
  }

  std::string ResolvedRtspInputUrl() const {
    if (input_rtsp_url.empty()) {
      return std::string();
    }
    if (rtsp_username.empty()) {
      return input_rtsp_url;
    }

    static const std::string kScheme = "rtsp://";
    if (input_rtsp_url.rfind(kScheme, 0) != 0) {
      return input_rtsp_url;
    }

    const std::string rest = input_rtsp_url.substr(kScheme.size());
    const std::size_t slash_pos = rest.find('/');
    const std::size_t at_pos = rest.find('@');
    if (at_pos != std::string::npos && (slash_pos == std::string::npos || at_pos < slash_pos)) {
      return input_rtsp_url;
    }

    std::string auth = rtsp_username;
    if (!rtsp_password.empty()) {
      auth += ":" + rtsp_password;
    }
    return kScheme + auth + "@" + rest;
  }

  std::string DisplayRtspInputUrl() const {
    std::string url = ResolvedRtspInputUrl();
    static const std::string kScheme = "rtsp://";
    if (url.rfind(kScheme, 0) != 0) {
      return url;
    }

    const std::size_t auth_begin = kScheme.size();
    const std::size_t slash_pos = url.find('/', auth_begin);
    const std::size_t at_pos = url.find('@', auth_begin);
    if (at_pos == std::string::npos || (slash_pos != std::string::npos && at_pos > slash_pos)) {
      return url;
    }

    const std::size_t colon_pos = url.find(':', auth_begin);
    if (colon_pos != std::string::npos && colon_pos < at_pos) {
      return url.substr(0, colon_pos + 1) + "***" + url.substr(at_pos);
    }
    return url.substr(0, auth_begin) + "***@" + url.substr(at_pos + 1);
  }

  std::string DisplayInputLocation() const {
    switch (source) {
      case SourceKind::kV4L2:
        return device;
      case SourceKind::kRtsp:
        return DisplayRtspInputUrl();
      case SourceKind::kLocalVideo:
        return local_video_path;
    }
    return std::string();
  }
};

}  // namespace rk3576_yolo_demo
