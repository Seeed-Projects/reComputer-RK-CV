#include "rk3576_yolo_demo/source/rtsp_source.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_yolo_demo {

namespace {

struct AVDictionary;
struct AVFormatContext;
struct AVInputFormat;
struct AVCodec;

enum AVMediaType {
  AVMEDIA_TYPE_VIDEO = 0,
};

struct AVPacket {
  void* buf;
  std::int64_t pts;
  std::int64_t dts;
  std::uint8_t* data;
  int size;
  int stream_index;
  int flags;
  void* side_data;
  int side_data_elems;
  std::int64_t duration;
  std::int64_t pos;
  std::int64_t convergence_duration;
};

constexpr std::int64_t kAvNoPtsValue = std::numeric_limits<std::int64_t>::min();

std::string ShellEscape(const std::string& input) {
  std::string escaped = "'";
  for (char ch : input) {
    if (ch == '\'') {
      escaped += "'\\''";
    } else {
      escaped.push_back(ch);
    }
  }
  escaped.push_back('\'');
  return escaped;
}

std::uint64_t MonotonicNowMs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

const char* CodecName(rk3576_demo::InputCodec codec) {
  switch (codec) {
    case rk3576_demo::InputCodec::kMjpeg:
      return "mjpeg";
    case rk3576_demo::InputCodec::kH264:
      return "h264";
    case rk3576_demo::InputCodec::kH265:
      return "hevc";
  }
  return "h264";
}

class LibavApi {
 public:
  bool EnsureLoaded() {
    if (loaded_) {
      return true;
    }

    avformat_handle_ = OpenLibrary({"libavformat.so", "libavformat.so.59", "libavformat.so.58"});
    if (avformat_handle_ == nullptr) {
      load_error_ = "Unable to load libavformat shared library";
      return false;
    }
    avcodec_handle_ = OpenLibrary({"libavcodec.so", "libavcodec.so.59", "libavcodec.so.58"});
    if (avcodec_handle_ == nullptr) {
      load_error_ = "Unable to load libavcodec shared library";
      return false;
    }
    avutil_handle_ = OpenLibrary({"libavutil.so", "libavutil.so.57", "libavutil.so.56"});
    if (avutil_handle_ == nullptr) {
      load_error_ = "Unable to load libavutil shared library";
      return false;
    }

    if (!LoadSymbols()) {
      return false;
    }
    loaded_ = true;
    return true;
  }

  const std::string& load_error() const { return load_error_; }

  int (*avformat_network_init)() = nullptr;
  int (*avformat_network_deinit)() = nullptr;
  int (*avformat_open_input)(AVFormatContext**, const char*, AVInputFormat*, AVDictionary**) = nullptr;
  int (*avformat_find_stream_info)(AVFormatContext*, AVDictionary**) = nullptr;
  void (*avformat_close_input)(AVFormatContext**) = nullptr;
  int (*av_read_frame)(AVFormatContext*, AVPacket*) = nullptr;
  int (*av_find_best_stream)(AVFormatContext*, AVMediaType, int, int, AVCodec**, int) = nullptr;
  AVPacket* (*av_packet_alloc)() = nullptr;
  int (*av_packet_ref)(AVPacket*, const AVPacket*) = nullptr;
  void (*av_packet_free)(AVPacket**) = nullptr;
  void (*av_packet_unref)(AVPacket*) = nullptr;
  int (*av_strerror)(int, char*, std::size_t) = nullptr;
  int (*av_dict_set)(AVDictionary**, const char*, const char*, int) = nullptr;
  void (*av_dict_free)(AVDictionary**) = nullptr;
  void (*av_log_set_level)(int) = nullptr;

 private:
  static void* OpenLibrary(std::initializer_list<const char*> names) {
    for (const char* name : names) {
      void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) {
        return handle;
      }
    }
    return nullptr;
  }

  template <typename T>
  bool LoadSymbol(void* handle, const char* symbol, T* out) {
    *out = reinterpret_cast<T>(dlsym(handle, symbol));
    if (*out == nullptr) {
      load_error_ = std::string("Missing symbol: ") + symbol;
      return false;
    }
    return true;
  }

  bool LoadSymbols() {
    const bool ok = LoadSymbol(avformat_handle_, "avformat_network_init", &avformat_network_init) &&
                    LoadSymbol(avformat_handle_, "avformat_network_deinit", &avformat_network_deinit) &&
                    LoadSymbol(avformat_handle_, "avformat_open_input", &avformat_open_input) &&
                    LoadSymbol(avformat_handle_, "avformat_find_stream_info", &avformat_find_stream_info) &&
                    LoadSymbol(avformat_handle_, "avformat_close_input", &avformat_close_input) &&
                    LoadSymbol(avformat_handle_, "av_read_frame", &av_read_frame) &&
                    LoadSymbol(avformat_handle_, "av_find_best_stream", &av_find_best_stream) &&
                    LoadSymbol(avutil_handle_, "av_strerror", &av_strerror) &&
                    LoadSymbol(avutil_handle_, "av_dict_set", &av_dict_set) &&
                    LoadSymbol(avutil_handle_, "av_dict_free", &av_dict_free) &&
                    LoadSymbol(avcodec_handle_, "av_packet_alloc", &av_packet_alloc) &&
                    LoadSymbol(avcodec_handle_, "av_packet_ref", &av_packet_ref) &&
                    LoadSymbol(avcodec_handle_, "av_packet_free", &av_packet_free) &&
                    LoadSymbol(avcodec_handle_, "av_packet_unref", &av_packet_unref);
    av_log_set_level = reinterpret_cast<void (*)(int)>(dlsym(avutil_handle_, "av_log_set_level"));
    return ok;
  }

  bool loaded_ = false;
  void* avformat_handle_ = nullptr;
  void* avcodec_handle_ = nullptr;
  void* avutil_handle_ = nullptr;
  std::string load_error_;
};

LibavApi& GetLibavApi() {
  static LibavApi api;
  return api;
}

std::string LibavErrorText(int error_code) {
  LibavApi& api = GetLibavApi();
  char buffer[256] = {};
  if (api.av_strerror != nullptr && api.av_strerror(error_code, buffer, sizeof(buffer)) == 0) {
    return std::string(buffer);
  }
  return "ffmpeg error " + std::to_string(error_code);
}

}  // namespace

struct RtspSource::LibavBackend {
  AVFormatContext* format_ctx = nullptr;
  AVPacket* packet = nullptr;
  int video_stream_index = -1;
};

RtspSource::RtspSource(const AppConfigV2& config) : config_(config) {}

RtspSource::~RtspSource() {
  Close();
}

bool RtspSource::Open() {
  if (opened_) {
    return true;
  }
  if (EffectiveInputLocation().empty()) {
    last_error_ = config_.source == SourceKind::kLocalVideo ? "Local video path is empty" : "RTSP input URL is empty";
    return false;
  }
  if (!ProbeStreamInfo()) {
    return false;
  }

  bool input_ok = false;
  if (config_.source == SourceKind::kLocalVideo) {
    // Local MP4/MOV commonly carry H264/H265 in AVCC/HVCC form. The ffmpeg pipe
    // path already applies mp4toannexb, which is more reliable for current MPP input.
    input_ok = StartFfmpegPipe();
  } else {
    input_ok = StartLibavInput();
    if (!input_ok) {
      const std::string libav_error = last_error_;
      RKLOG_WARN("APP") << "libavformat backend unavailable, fallback to ffmpeg pipe. detail="
                << libav_error << "\n";
      input_ok = StartFfmpegPipe();
    }
  }
  if (!input_ok) {
    return false;
  }

  opened_ = true;
  RKLOG_INFO("APP") << (config_.source == SourceKind::kLocalVideo ? "Local video source opened: path=" : "RTSP source opened: url=")
            << DisplayInputLocation()
            << " codec=" << CodecName(codec_)
            << " size=" << width_ << "x" << height_
            << " backend=" << (backend_ == BackendKind::kLibavformat ? "libavformat" : "ffmpeg-pipe")
            << "\n";
  return true;
}

void RtspSource::Close() {
  StopLibavInput();
  if (pipe_ != nullptr) {
    pclose(pipe_);
    pipe_ = nullptr;
  }
  backend_ = BackendKind::kNone;
  opened_ = false;
  mjpeg_buffer_.clear();
  if (!stderr_path_.empty()) {
    std::remove(stderr_path_.c_str());
    stderr_path_.clear();
  }
}

bool RtspSource::ReadPacket(CompressedPacket* packet) {
  if (packet == nullptr) {
    last_error_ = "packet pointer is null";
    return false;
  }
  packet->Clear();
  if (!opened_ && !Open()) {
    return false;
  }

  for (int attempt = 0; attempt < 3; ++attempt) {
    bool ok = false;
    if (backend_ == BackendKind::kLibavformat) {
      ok = ReadLibavPacket(packet);
    } else {
      ok = (codec_ == rk3576_demo::InputCodec::kMjpeg) ? ReadMjpegPacket(packet) : ReadElementaryPacket(packet);
    }
    if (ok) {
      packet->codec = codec_;
      return true;
    }
    const bool silent_loop = IsLocalVideoEof();
    if (!RestartPipe(silent_loop)) {
      return false;
    }
  }

  last_error_ = "RTSP reconnect exceeded retry budget";
  return false;
}

bool RtspSource::ProbeStreamInfo() {
  const std::string escaped_url = ShellEscape(EffectiveInputLocation());
  const std::string probe_base =
      "ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height -of csv=p=0 ";
  std::vector<std::string> commands;
  if (IsRtspSource()) {
    commands.push_back(probe_base + "-rtsp_transport tcp " + escaped_url);
  }
  commands.push_back(probe_base + escaped_url);

  std::string last_probe_output;
  for (const std::string& command_base : commands) {
    const std::string probe_stderr_path = PrepareStderrPath("ffprobe");
    const std::string cmd = command_base + " 2>" + ShellEscape(probe_stderr_path);
    std::FILE* probe = popen(cmd.c_str(), "r");
    if (probe == nullptr) {
      last_probe_output = "Failed to launch ffprobe";
      continue;
    }

    std::array<char, 512> buffer {};
    std::string output;
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), probe) != nullptr) {
      output += buffer.data();
    }
    const int status = pclose(probe);
    last_probe_output = ReadAndClearStderrFile();
    if (last_probe_output.empty()) {
      last_probe_output = output;
    }

    std::string line = output;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }

    if (status != 0 || line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string codec_name;
    std::string width_text;
    std::string height_text;
    std::getline(ss, codec_name, ',');
    std::getline(ss, width_text, ',');
    std::getline(ss, height_text, ',');

    if (codec_name == "h264") {
      codec_ = rk3576_demo::InputCodec::kH264;
    } else if (codec_name == "hevc" || codec_name == "h265") {
      codec_ = rk3576_demo::InputCodec::kH265;
    } else if (codec_name == "mjpeg") {
      codec_ = rk3576_demo::InputCodec::kMjpeg;
    } else {
      last_probe_output = "Unsupported codec from ffprobe: " + codec_name;
      continue;
    }

    width_ = width_text.empty() ? config_.camera_width : std::atoi(width_text.c_str());
    height_ = height_text.empty() ? config_.camera_height : std::atoi(height_text.c_str());
    if (width_ <= 0) {
      width_ = config_.camera_width;
    }
    if (height_ <= 0) {
      height_ = config_.camera_height;
    }
    return true;
  }

  codec_ = rk3576_demo::InputCodec::kH264;
  width_ = config_.camera_width;
  height_ = config_.camera_height;
  last_error_ = last_probe_output.empty() ? "ffprobe returned empty stream info" : last_probe_output;
  RKLOG_WARN("APP") << (IsRtspSource()
                            ? "RTSP probe failed, fallback to configured geometry and default H264. detail="
                            : "Local video probe failed, fallback to configured geometry and default H264. detail=")
            << last_error_
            << " fallback_size=" << width_ << "x" << height_ << "\n";
  return true;
}

bool RtspSource::StartLibavInput() {
  StopLibavInput();

  LibavApi& api = GetLibavApi();
  if (!api.EnsureLoaded()) {
    last_error_ = api.load_error();
    return false;
  }

  if (api.avformat_network_init != nullptr) {
    api.avformat_network_init();
  }
  if (api.av_log_set_level != nullptr) {
    api.av_log_set_level(16);
  }

  auto open_once = [&](bool use_tcp) -> bool {
    std::unique_ptr<LibavBackend> backend(new LibavBackend());
    AVDictionary* options = nullptr;
    if (api.av_dict_set != nullptr) {
      if (IsRtspSource() && use_tcp) {
        api.av_dict_set(&options, "rtsp_transport", "tcp", 0);
      }
      if (IsRtspSource()) {
        api.av_dict_set(&options, "fflags", "nobuffer", 0);
        api.av_dict_set(&options, "flags", "low_delay", 0);
        api.av_dict_set(&options, "reorder_queue_size", "0", 0);
        api.av_dict_set(&options, "max_delay", "0", 0);
        api.av_dict_set(&options, "buffer_size", "102400", 0);
      }
    }

    AVFormatContext* context = nullptr;
    const int open_rc = api.avformat_open_input(&context, EffectiveInputLocation().c_str(), nullptr, &options);
    if (api.av_dict_free != nullptr) {
      api.av_dict_free(&options);
    }
    if (open_rc < 0 || context == nullptr) {
      last_error_ = "avformat_open_input failed: " + LibavErrorText(open_rc);
      if (context != nullptr && api.avformat_close_input != nullptr) {
        api.avformat_close_input(&context);
      }
      return false;
    }

    const int info_rc = api.avformat_find_stream_info(context, nullptr);
    if (info_rc < 0) {
      last_error_ = "avformat_find_stream_info failed: " + LibavErrorText(info_rc);
      api.avformat_close_input(&context);
      return false;
    }

    const int video_stream_index = api.av_find_best_stream(context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index < 0) {
      last_error_ = "av_find_best_stream failed: " + LibavErrorText(video_stream_index);
      api.avformat_close_input(&context);
      return false;
    }

    AVPacket* packet = api.av_packet_alloc();
    if (packet == nullptr) {
      last_error_ = "av_packet_alloc failed";
      api.avformat_close_input(&context);
      return false;
    }

    backend->format_ctx = context;
    backend->packet = packet;
    backend->video_stream_index = video_stream_index;
    libav_ = std::move(backend);
    backend_ = BackendKind::kLibavformat;
    return true;
  };

  if (IsRtspSource()) {
    if (open_once(true)) {
      return true;
    }
    const std::string tcp_error = last_error_;
    if (open_once(false)) {
      return true;
    }
    last_error_ = tcp_error + " | fallback_open: " + last_error_;
    return false;
  }
  return open_once(false);
}

void RtspSource::StopLibavInput() {
  if (!libav_) {
    return;
  }

  LibavApi& api = GetLibavApi();
  if (api.av_packet_free != nullptr && libav_->packet != nullptr) {
    api.av_packet_free(&libav_->packet);
  }
  if (api.avformat_close_input != nullptr && libav_->format_ctx != nullptr) {
    api.avformat_close_input(&libav_->format_ctx);
  }
  libav_.reset();
}

bool RtspSource::StartFfmpegPipe() {
  if (pipe_ != nullptr) {
    pclose(pipe_);
    pipe_ = nullptr;
  }

  const std::string escaped_url = ShellEscape(EffectiveInputLocation());
  const std::string ffmpeg_base = "ffmpeg -nostdin -hide_banner -loglevel error ";
  const std::string input_prefix = (config_.source == SourceKind::kLocalVideo) ? "-re -i " : "-i ";
  std::string ffmpeg_sink;
  if (codec_ == rk3576_demo::InputCodec::kH264) {
    ffmpeg_sink = "-bsf:v h264_mp4toannexb -f h264 -";
  } else if (codec_ == rk3576_demo::InputCodec::kH265) {
    ffmpeg_sink = "-bsf:v hevc_mp4toannexb -f hevc -";
  } else {
    ffmpeg_sink = "-f mjpeg -";
  }

  std::vector<std::string> commands;
  if (IsRtspSource()) {
    commands.push_back(ffmpeg_base + "-rtsp_transport tcp -i " + escaped_url + " -map 0:v:0 -an -c:v copy " + ffmpeg_sink);
  }
  commands.push_back(ffmpeg_base + input_prefix + escaped_url + " -map 0:v:0 -an -c:v copy " + ffmpeg_sink);

  for (const std::string& command_base : commands) {
    stderr_path_ = PrepareStderrPath("ffmpeg");
    const std::string cmd = command_base + " 2>" + ShellEscape(stderr_path_);
    pipe_ = popen(cmd.c_str(), "r");
    if (pipe_ != nullptr) {
      backend_ = BackendKind::kFfmpegPipe;
      mjpeg_buffer_.clear();
      return true;
    }
  }

  last_error_ = IsRtspSource() ? "Failed to launch ffmpeg for RTSP input" : "Failed to launch ffmpeg for local video input";
  return false;
}

bool RtspSource::RestartPipe(bool silent_loop) {
  StopLibavInput();
  if (pipe_ != nullptr) {
    pclose(pipe_);
    pipe_ = nullptr;
  }
  backend_ = BackendKind::kNone;
  opened_ = false;

  if (!silent_loop) {
    RKLOG_WARN("APP") << (IsRtspSource() ? "RTSP source reconnecting after read failure, wait_ms="
                                         : "Local video source reopening after read failure, wait_ms=")
              << config_.rtsp_reconnect_interval_ms << "\n";
  } else {
    RKLOG_INFO("APP") << "Local video reached EOF, restart from beginning\n";
  }
  const int wait_ms = silent_loop ? 0 : config_.rtsp_reconnect_interval_ms;
  if (wait_ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
  }

  if (!ProbeStreamInfo()) {
    return false;
  }
  if (config_.source == SourceKind::kLocalVideo) {
    if (!StartFfmpegPipe()) {
      return false;
    }
  } else if (!StartLibavInput() && !StartFfmpegPipe()) {
    return false;
  }
  opened_ = true;
  return true;
}

bool RtspSource::ReadMjpegPacket(CompressedPacket* packet) {
  static constexpr std::size_t kReadSize = 64 * 1024;
  std::array<std::uint8_t, kReadSize> chunk {};

  while (true) {
    for (std::size_t i = 0; i + 1 < mjpeg_buffer_.size(); ++i) {
      if (mjpeg_buffer_[i] != 0xFF || mjpeg_buffer_[i + 1] != 0xD8) {
        continue;
      }
      for (std::size_t j = i + 2; j + 1 < mjpeg_buffer_.size(); ++j) {
        if (mjpeg_buffer_[j] == 0xFF && mjpeg_buffer_[j + 1] == 0xD9) {
          std::vector<std::uint8_t> owned_data(mjpeg_buffer_.begin() + static_cast<long>(i),
                                               mjpeg_buffer_.begin() + static_cast<long>(j + 2));
          packet->AssignOwned(std::move(owned_data));
          packet->pts_ms = MonotonicNowMs();
          packet->eos = false;
          mjpeg_buffer_.erase(mjpeg_buffer_.begin(), mjpeg_buffer_.begin() + static_cast<long>(j + 2));
          return true;
        }
      }
      break;
    }

    const std::size_t bytes = std::fread(chunk.data(), 1, chunk.size(), pipe_);
    if (bytes == 0) {
      const std::string detail = ReadAndClearStderrFile();
      last_error_ = detail.empty() ? "ffmpeg mjpeg pipe reached EOF" : detail;
      return false;
    }
    mjpeg_buffer_.insert(mjpeg_buffer_.end(), chunk.begin(), chunk.begin() + static_cast<long>(bytes));
  }
}

bool RtspSource::ReadElementaryPacket(CompressedPacket* packet) {
  static constexpr std::size_t kReadSize = 64 * 1024;
  std::array<std::uint8_t, kReadSize> chunk {};
  const std::size_t bytes = std::fread(chunk.data(), 1, chunk.size(), pipe_);
  if (bytes == 0) {
    const std::string detail = ReadAndClearStderrFile();
    last_error_ = detail.empty() ? "ffmpeg elementary pipe reached EOF" : detail;
    return false;
  }

  std::vector<std::uint8_t> owned_data(chunk.begin(), chunk.begin() + static_cast<long>(bytes));
  packet->AssignOwned(std::move(owned_data));
  packet->pts_ms = MonotonicNowMs();
  packet->eos = false;
  return true;
}

bool RtspSource::ReadLibavPacket(CompressedPacket* packet) {
  if (!libav_) {
    last_error_ = "libavformat backend is not initialized";
    return false;
  }

  LibavApi& api = GetLibavApi();
  while (true) {
    const int rc = api.av_read_frame(libav_->format_ctx, libav_->packet);
    if (rc < 0) {
      last_error_ = (rc == -541478725)
                        ? (config_.source == SourceKind::kLocalVideo ? "local video reached EOF" : "libavformat reached EOF")
                        : ("av_read_frame failed: " + LibavErrorText(rc));
      return false;
    }

    if (libav_->packet->stream_index != libav_->video_stream_index) {
      api.av_packet_unref(libav_->packet);
      continue;
    }

    if (libav_->packet->data == nullptr || libav_->packet->size <= 0) {
      api.av_packet_unref(libav_->packet);
      continue;
    }

    AVPacket* packet_ref = api.av_packet_alloc();
    if (packet_ref == nullptr) {
      last_error_ = "av_packet_alloc for packet_ref failed";
      api.av_packet_unref(libav_->packet);
      return false;
    }
    const int ref_rc = api.av_packet_ref(packet_ref, libav_->packet);
    if (ref_rc < 0) {
      last_error_ = "av_packet_ref failed: " + LibavErrorText(ref_rc);
      api.av_packet_free(&packet_ref);
      api.av_packet_unref(libav_->packet);
      return false;
    }
    std::shared_ptr<void> packet_owner(packet_ref, [](void* opaque) {
      LibavApi& api = GetLibavApi();
      AVPacket* owned_packet = static_cast<AVPacket*>(opaque);
      if (api.av_packet_free != nullptr && owned_packet != nullptr) {
        api.av_packet_free(&owned_packet);
      }
    });
    packet->AssignBorrowed(packet_ref->data, static_cast<std::size_t>(packet_ref->size), std::move(packet_owner));
    packet->pts_ms = MonotonicNowMs();
    packet->eos = false;
    api.av_packet_unref(libav_->packet);
    return true;
  }
}

SourceDescriptor RtspSource::Describe() const {
  SourceDescriptor descriptor;
  descriptor.kind = config_.source;
  descriptor.name = config_.source == SourceKind::kLocalVideo ? "local-video-input" : "rtsp-network-input";
  descriptor.location = DisplayInputLocation();
  descriptor.compressed_input = true;
  descriptor.raw_input = false;
  return descriptor;
}

std::string RtspSource::EffectiveInputLocation() const {
  return config_.source == SourceKind::kLocalVideo ? config_.local_video_path : config_.ResolvedRtspInputUrl();
}

std::string RtspSource::DisplayInputLocation() const {
  return config_.source == SourceKind::kLocalVideo ? config_.local_video_path : config_.DisplayRtspInputUrl();
}

bool RtspSource::IsLocalVideoEof() const {
  if (config_.source != SourceKind::kLocalVideo) {
    return false;
  }
  return last_error_ == "local video reached EOF" ||
         last_error_ == "ffmpeg elementary pipe reached EOF" ||
         last_error_ == "ffmpeg mjpeg pipe reached EOF";
}

std::string RtspSource::PrepareStderrPath(const char* stage) {
  if (!stderr_path_.empty()) {
    std::remove(stderr_path_.c_str());
    stderr_path_.clear();
  }
  std::ostringstream oss;
  oss << "/tmp/rk3576_rtsp_" << stage << "_" << getpid() << "_" << MonotonicNowMs() << ".log";
  stderr_path_ = oss.str();
  return stderr_path_;
}

std::string RtspSource::ReadAndClearStderrFile() {
  if (stderr_path_.empty()) {
    return std::string();
  }

  std::ifstream input(stderr_path_.c_str(), std::ios::in);
  std::stringstream ss;
  ss << input.rdbuf();
  input.close();
  std::remove(stderr_path_.c_str());
  stderr_path_.clear();

  std::string output = ss.str();
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

}  // namespace rk3576_yolo_demo
