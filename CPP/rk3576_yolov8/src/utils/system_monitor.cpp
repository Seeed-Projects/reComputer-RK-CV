#include "rk3576_demo/system_monitor.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

namespace rk3576_demo {

namespace {

std::string Trim(std::string value) {
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(),
                           [](unsigned char ch) { return !std::isspace(ch); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [](unsigned char ch) { return !std::isspace(ch); })
                  .base(),
              value.end());
  return value;
}

bool ReadWholeFile(const std::string& path, std::string* content) {
  if (content == nullptr) {
    return false;
  }

  std::ifstream input(path);
  if (!input.is_open()) {
    return false;
  }

  std::ostringstream oss;
  oss << input.rdbuf();
  *content = oss.str();
  return true;
}

bool ParseUnsigned(const std::string& value, std::uint64_t* out) {
  if (out == nullptr) {
    return false;
  }
  try {
    *out = static_cast<std::uint64_t>(std::stoull(value));
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

SystemMonitor::~SystemMonitor() {
  Stop();
}

bool SystemMonitor::Start(std::chrono::milliseconds interval, SnapshotCallback callback) {
  if (running_) {
    return true;
  }

  interval_ = interval.count() > 0 ? interval : std::chrono::milliseconds(1000);
  callback_ = std::move(callback);

  gpu_percent_paths_ = {
      "/sys/class/devfreq/fb000000.gpu/load",
      "/sys/class/devfreq/fb000000.gpu/device/load",
      "/sys/devices/platform/fb000000.gpu/load",
      "/sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/load",
      "/sys/kernel/debug/gpu/utilisation",
  };
  gpu_counter_files_ = {
      {"/sys/class/devfreq/fb000000.gpu/device/busy_time", "/sys/class/devfreq/fb000000.gpu/device/total_time",
       "fb000000.gpu busy/total"},
      {"/sys/class/devfreq/fb000000.gpu/busy_time", "/sys/class/devfreq/fb000000.gpu/total_time",
       "fb000000.gpu busy/total"},
  };
  vpu_percent_paths_ = {
      "/sys/kernel/debug/mpp_service/load",
      "/sys/kernel/debug/vcodec_service/load",
      "/sys/kernel/debug/rkvdec/load",
      "/sys/kernel/debug/rkvenc/load",
  };
  vpu_counter_files_ = {
      {"/sys/kernel/debug/mpp_service/busy_time", "/sys/kernel/debug/mpp_service/total_time", "mpp_service busy/total"},
      {"/sys/kernel/debug/vcodec_service/busy_time", "/sys/kernel/debug/vcodec_service/total_time",
       "vcodec_service busy/total"},
  };

  has_prev_cpu_times_ = false;
  prev_gpu_busy_ = 0;
  prev_gpu_total_ = 0;
  prev_vpu_busy_ = 0;
  prev_vpu_total_ = 0;

  running_ = true;
  worker_ = std::thread(&SystemMonitor::ThreadMain, this);
  return true;
}

void SystemMonitor::Stop() {
  running_ = false;
  if (worker_.joinable()) {
    worker_.join();
  }
}

void SystemMonitor::ThreadMain() {
  while (running_) {
    if (callback_) {
      ResourceSnapshot snapshot;
      if (Sample(&snapshot)) {
        callback_(snapshot);
      }
    }
    std::this_thread::sleep_for(interval_);
  }
}

bool SystemMonitor::Sample(ResourceSnapshot* snapshot) {
  if (snapshot == nullptr) {
    return false;
  }
  *snapshot = ResourceSnapshot {};
  snapshot->has_cpu = ReadCpuPercent(&snapshot->cpu_percent);
  snapshot->has_gpu = ReadGpuPercent(&snapshot->gpu_percent, &snapshot->gpu_source);
  snapshot->has_vpu = ReadVpuPercent(&snapshot->vpu_percent, &snapshot->vpu_source);
  return true;
}

bool SystemMonitor::ReadCpuTimes(CpuTimes* times) const {
  if (times == nullptr) {
    return false;
  }
  std::ifstream input("/proc/stat");
  if (!input.is_open()) {
    return false;
  }

  std::string cpu_label;
  std::uint64_t user = 0;
  std::uint64_t nice = 0;
  std::uint64_t system = 0;
  std::uint64_t idle = 0;
  std::uint64_t iowait = 0;
  std::uint64_t irq = 0;
  std::uint64_t softirq = 0;
  std::uint64_t steal = 0;
  input >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
  if (!input.good() || cpu_label != "cpu") {
    return false;
  }

  times->idle = idle + iowait;
  times->total = user + nice + system + idle + iowait + irq + softirq + steal;
  return true;
}

bool SystemMonitor::ReadCpuPercent(double* percent) {
  if (percent == nullptr) {
    return false;
  }
  CpuTimes current;
  if (!ReadCpuTimes(&current)) {
    return false;
  }
  if (!has_prev_cpu_times_) {
    prev_cpu_times_ = current;
    has_prev_cpu_times_ = true;
    return false;
  }

  const auto idle_delta = current.idle - prev_cpu_times_.idle;
  const auto total_delta = current.total - prev_cpu_times_.total;
  prev_cpu_times_ = current;
  if (total_delta == 0) {
    return false;
  }

  *percent = 100.0 * static_cast<double>(total_delta - idle_delta) / static_cast<double>(total_delta);
  return true;
}

bool SystemMonitor::ReadGpuPercent(double* percent, std::string* source) {
  if (ReadPercentFromPaths(gpu_percent_paths_, percent, source)) {
    return true;
  }
  return ReadCounterFromFiles(gpu_counter_files_, &prev_gpu_busy_, &prev_gpu_total_, percent, source);
}

bool SystemMonitor::ReadVpuPercent(double* percent, std::string* source) {
  if (ReadPercentFromPaths(vpu_percent_paths_, percent, source)) {
    return true;
  }
  return ReadCounterFromFiles(vpu_counter_files_, &prev_vpu_busy_, &prev_vpu_total_, percent, source);
}

bool SystemMonitor::ReadPercentFromPaths(const std::vector<std::string>& paths, double* percent, std::string* source) {
  if (source != nullptr) {
    source->clear();
  }
  for (const auto& path : paths) {
    if (ParsePercentFile(path, percent)) {
      if (source != nullptr) {
        *source = path;
      }
      return true;
    }
  }
  return false;
}

bool SystemMonitor::ReadCounterFromFiles(const std::vector<CounterFile>& files,
                                         std::uint64_t* prev_busy,
                                         std::uint64_t* prev_total,
                                         double* percent,
                                         std::string* source) {
  if (source != nullptr) {
    source->clear();
  }
  for (const auto& file : files) {
    if (ParseCounterFile(file, prev_busy, prev_total, percent)) {
      if (source != nullptr) {
        *source = file.label;
      }
      return true;
    }
  }
  return false;
}

bool SystemMonitor::ParsePercentFile(const std::string& path, double* percent) const {
  if (percent == nullptr) {
    return false;
  }
  std::string content;
  if (!ReadWholeFile(path, &content)) {
    return false;
  }

  content = Trim(content);
  if (content.empty()) {
    return false;
  }

  for (char& ch : content) {
    if (ch == '%' || ch == '@' || ch == ',') {
      ch = ' ';
    }
  }

  std::istringstream iss(content);
  double first = 0.0;
  double second = 0.0;
  if (!(iss >> first)) {
    return false;
  }
  if (iss >> second) {
    if (second <= 0.0) {
      return false;
    }
    *percent = 100.0 * first / second;
    return true;
  }

  if (first >= 0.0 && first <= 100.0) {
    *percent = first;
    return true;
  }
  if (first > 100.0 && first <= 1000.0) {
    *percent = first / 10.0;
    return true;
  }
  return false;
}

bool SystemMonitor::ParseCounterFile(const CounterFile& file,
                                     std::uint64_t* prev_busy,
                                     std::uint64_t* prev_total,
                                     double* percent) const {
  if (prev_busy == nullptr || prev_total == nullptr || percent == nullptr) {
    return false;
  }
  std::string busy_content;
  std::string total_content;
  if (!ReadWholeFile(file.busy_path, &busy_content) || !ReadWholeFile(file.total_path, &total_content)) {
    return false;
  }

  std::uint64_t busy = 0;
  std::uint64_t total = 0;
  if (!ParseUnsigned(Trim(busy_content), &busy) || !ParseUnsigned(Trim(total_content), &total)) {
    return false;
  }
  if (*prev_total == 0) {
    *prev_busy = busy;
    *prev_total = total;
    return false;
  }

  const auto busy_delta = busy - *prev_busy;
  const auto total_delta = total - *prev_total;
  *prev_busy = busy;
  *prev_total = total;
  if (total_delta == 0) {
    return false;
  }

  *percent = 100.0 * static_cast<double>(busy_delta) / static_cast<double>(total_delta);
  return true;
}

}  // namespace rk3576_demo
