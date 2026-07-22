#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rk3576_demo/media_types.hpp"
#include "rk3576_yolo_demo/common/runtime_types.hpp"

namespace rk3576_yolo_demo {

struct CompressedPacket {
  std::vector<std::uint8_t> data;
  const std::uint8_t* payload = nullptr;
  std::size_t payload_size = 0;
  std::shared_ptr<void> payload_owner;
  std::uint64_t pts_ms = 0;
  rk3576_demo::InputCodec codec = rk3576_demo::InputCodec::kMjpeg;
  bool eos = false;

  void Clear() {
    data.clear();
    payload = nullptr;
    payload_size = 0;
    payload_owner.reset();
    pts_ms = 0;
    codec = rk3576_demo::InputCodec::kMjpeg;
    eos = false;
  }

  void AssignOwned(std::vector<std::uint8_t>&& owned_data) {
    data = std::move(owned_data);
    payload = data.empty() ? nullptr : data.data();
    payload_size = data.size();
    payload_owner.reset();
  }

  void AssignBorrowed(const std::uint8_t* borrowed_payload, std::size_t borrowed_size, std::shared_ptr<void> owner) {
    data.clear();
    payload = borrowed_payload;
    payload_size = borrowed_size;
    payload_owner = std::move(owner);
  }

  bool Empty() const {
    return payload == nullptr || payload_size == 0;
  }

  const std::uint8_t* Data() const {
    return payload;
  }

  std::size_t Size() const {
    return payload_size;
  }
};

class IInputSource {
 public:
  virtual ~IInputSource() = default;

  virtual const char* Name() const = 0;
  virtual bool Open() = 0;
  virtual void Close() = 0;
  virtual SourceDescriptor Describe() const = 0;
  virtual std::string LastError() const = 0;
  virtual bool SupportsPacketRead() const { return false; }
  virtual bool ReadPacket(CompressedPacket* packet) {
    (void)packet;
    return false;
  }
  virtual rk3576_demo::InputCodec OutputCodec() const { return rk3576_demo::InputCodec::kMjpeg; }
  virtual int OutputWidth() const { return 0; }
  virtual int OutputHeight() const { return 0; }
};

}  // namespace rk3576_yolo_demo
