#pragma once

#include <string>

namespace rk3576_yolo_demo {

class PublisherChannel {
 public:
  virtual ~PublisherChannel() = default;
  virtual const char* Name() const = 0;
  virtual std::string Describe() const = 0;
};

}  // namespace rk3576_yolo_demo
