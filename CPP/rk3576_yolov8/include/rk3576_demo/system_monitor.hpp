#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace rk3576_demo {

struct ResourceSnapshot {
  bool has_cpu = false;
  bool has_gpu = false;
  bool has_vpu = false;
  double cpu_percent = 0.0;
  double gpu_percent = 0.0;
  double vpu_percent = 0.0;
  std::string gpu_source;
  std::string vpu_source;
};

class SystemMonitor {
 public:
  using SnapshotCallback = std::function<void(const ResourceSnapshot&)>;

  SystemMonitor() = default;
  ~SystemMonitor();

  bool Start(std::chrono::milliseconds interval, SnapshotCallback callback);
  void Stop();

 private:
  struct CpuTimes {
    std::uint64_t idle = 0;
    std::uint64_t total = 0;
  };

  struct CounterFile {
    std::string busy_path;
    std::string total_path;
    std::string label;
  };

  void ThreadMain();
  bool Sample(ResourceSnapshot* snapshot);
  bool ReadCpuTimes(CpuTimes* times) const;
  bool ReadCpuPercent(double* percent);
  bool ReadGpuPercent(double* percent, std::string* source);
  bool ReadVpuPercent(double* percent, std::string* source);
  bool ReadPercentFromPaths(const std::vector<std::string>& paths, double* percent, std::string* source);
  bool ReadCounterFromFiles(const std::vector<CounterFile>& files, std::uint64_t* prev_busy,
                            std::uint64_t* prev_total, double* percent, std::string* source);
  bool ParsePercentFile(const std::string& path, double* percent) const;
  bool ParseCounterFile(const CounterFile& file, std::uint64_t* prev_busy,
                        std::uint64_t* prev_total, double* percent) const;

  std::chrono::milliseconds interval_ {1000};
  SnapshotCallback callback_;
  std::thread worker_;
  std::atomic<bool> running_ {false};
  CpuTimes prev_cpu_times_ {};
  bool has_prev_cpu_times_ = false;
  std::vector<std::string> gpu_percent_paths_;
  std::vector<CounterFile> gpu_counter_files_;
  std::vector<std::string> vpu_percent_paths_;
  std::vector<CounterFile> vpu_counter_files_;
  std::uint64_t prev_gpu_busy_ = 0;
  std::uint64_t prev_gpu_total_ = 0;
  std::uint64_t prev_vpu_busy_ = 0;
  std::uint64_t prev_vpu_total_ = 0;
};

}  // namespace rk3576_demo
