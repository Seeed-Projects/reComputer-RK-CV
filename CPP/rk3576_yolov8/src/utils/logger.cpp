#include "rk3576_yolo_demo/common/logger.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <thread>

namespace rk3576_yolo_demo {

Logger& Logger::Instance() {
  static Logger logger;
  return logger;
}

void Logger::Log(Level level, const std::string& tag, const std::string& message) {
  if (level == Level::kInfo && !info_enabled_) {
    return;
  }
  const auto now = std::chrono::system_clock::now();
  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

  std::tm local_tm {};
  localtime_r(&now_time, &local_tm);

  std::ostringstream oss;
  oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S")
      << '.' << std::setfill('0') << std::setw(3) << now_ms.count()
      << " [" << ToString(level) << "]"
      << "[tid=" << std::this_thread::get_id() << "]"
      << "[" << tag << "] "
      << message;

  std::lock_guard<std::mutex> lock(mutex_);
  std::ostream& out = level == Level::kError ? std::cerr : std::cout;
  out << oss.str() << '\n';
}

const char* Logger::ToString(Level level) {
  switch (level) {
    case Level::kDebug:
      return "DEBUG";
    case Level::kInfo:
      return "INFO";
    case Level::kWarn:
      return "WARN";
    case Level::kError:
      return "ERROR";
  }
  return "UNKNOWN";
}

LogMessage::LogMessage(Logger::Level level, std::string tag, bool force)
    : level_(level), tag_(std::move(tag)),
      enabled_(force || level != Logger::Level::kInfo || Logger::Instance().info_enabled()) {}

LogMessage::~LogMessage() {
  if (!enabled_) {
    return;
  }
  Logger::Instance().Log(level_, tag_, stream_.str());
}

}  // namespace rk3576_yolo_demo
