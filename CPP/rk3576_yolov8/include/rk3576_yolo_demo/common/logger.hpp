#pragma once

#include <mutex>
#include <sstream>
#include <string>

namespace rk3576_yolo_demo {

class Logger {
 public:
  enum class Level {
    kDebug,
    kInfo,
    kWarn,
    kError,
  };

  static Logger& Instance();

  void SetInfoEnabled(bool enabled) { info_enabled_ = enabled; }
  bool info_enabled() const { return info_enabled_; }
  void Log(Level level, const std::string& tag, const std::string& message);

 private:
  Logger() = default;

  static const char* ToString(Level level);

  std::mutex mutex_;
  bool info_enabled_ = false;
};

class LogMessage {
 public:
  LogMessage(Logger::Level level, std::string tag, bool force = false);
  ~LogMessage();

  template <typename T>
  LogMessage& operator<<(const T& value) {
    stream_ << value;
    return *this;
  }

 private:
  Logger::Level level_;
  std::string tag_;
  bool enabled_ = true;
  std::ostringstream stream_;
};

}  // namespace rk3576_yolo_demo

#define RKLOG_DEBUG(tag) ::rk3576_yolo_demo::LogMessage(::rk3576_yolo_demo::Logger::Level::kDebug, tag)
#define RKLOG_INFO(tag) ::rk3576_yolo_demo::LogMessage(::rk3576_yolo_demo::Logger::Level::kInfo, tag)
#define RKLOG_INFO_ALWAYS(tag) ::rk3576_yolo_demo::LogMessage(::rk3576_yolo_demo::Logger::Level::kInfo, tag, true)
#define RKLOG_WARN(tag) ::rk3576_yolo_demo::LogMessage(::rk3576_yolo_demo::Logger::Level::kWarn, tag)
#define RKLOG_ERROR(tag) ::rk3576_yolo_demo::LogMessage(::rk3576_yolo_demo::Logger::Level::kError, tag)
