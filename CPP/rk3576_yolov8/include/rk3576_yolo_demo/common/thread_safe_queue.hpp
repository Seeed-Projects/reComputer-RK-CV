#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace rk3576_yolo_demo {

template <typename T>
class ThreadSafeQueue {
 public:
  explicit ThreadSafeQueue(std::size_t capacity = 4) : capacity_(capacity) {}

  bool Push(const T& value, bool* dropped_oldest = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    if (dropped_oldest != nullptr) {
      *dropped_oldest = false;
    }
    if (capacity_ > 0 && queue_.size() >= capacity_) {
      queue_.pop();
      if (dropped_oldest != nullptr) {
        *dropped_oldest = true;
      }
    }
    queue_.push(value);
    cond_.notify_one();
    return true;
  }

  bool TryPop(T* value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty() || value == nullptr) {
      return false;
    }
    *value = queue_.front();
    queue_.pop();
    return true;
  }

  bool WaitPop(T* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]() { return closed_ || !queue_.empty(); });
    if (queue_.empty() || value == nullptr) {
      return false;
    }
    *value = queue_.front();
    queue_.pop();
    return true;
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    cond_.notify_all();
  }

  bool closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  std::size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::size_t capacity_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  std::queue<T> queue_;
  bool closed_ = false;
};

}  // namespace rk3576_yolo_demo
