#include "rk3576_demo/v4l2_camera.hpp"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_demo {

namespace {

bool IoctlRetry(int fd, unsigned long request, void* arg) {
  int ret = 0;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && errno == EINTR);
  return ret != -1;
}

std::uint64_t ToMilliseconds(const timeval& tv) {
  return static_cast<std::uint64_t>(tv.tv_sec) * 1000ULL +
         static_cast<std::uint64_t>(tv.tv_usec) / 1000ULL;
}

}  // namespace

V4L2Camera::~V4L2Camera() {
  Close();
}

bool V4L2Camera::Open(const AppConfig& config) {
  device_ = config.device;
  fd_ = open(device_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC, 0);
  if (fd_ < 0) {
    RKLOG_ERROR("APP") << "Failed to open camera " << device_ << ": " << std::strerror(errno) << "\n";
    return false;
  }

  if (!InitDevice(config) || !InitMmap(config.v4l2_buffer_count) || !StartStreaming()) {
    Close();
    return false;
  }
  return true;
}

bool V4L2Camera::InitDevice(const AppConfig& config) {
  v4l2_capability cap {};
  if (!IoctlRetry(fd_, VIDIOC_QUERYCAP, &cap)) {
    RKLOG_ERROR("APP") << "VIDIOC_QUERYCAP failed: " << std::strerror(errno) << "\n";
    return false;
  }
  if ((cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
      (cap.capabilities & V4L2_CAP_STREAMING) == 0) {
    RKLOG_ERROR("APP") << "Device does not support capture + streaming\n";
    return false;
  }

  v4l2_format fmt {};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = config.camera_width;
  fmt.fmt.pix.height = config.camera_height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;

  if (!IoctlRetry(fd_, VIDIOC_S_FMT, &fmt)) {
    RKLOG_ERROR("APP") << "VIDIOC_S_FMT failed: " << std::strerror(errno) << "\n";
    return false;
  }

  v4l2_streamparm parm {};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = config.fps;
  if (!IoctlRetry(fd_, VIDIOC_S_PARM, &parm)) {
    RKLOG_ERROR("APP") << "VIDIOC_S_PARM failed: " << std::strerror(errno) << "\n";
  }

  RKLOG_INFO("APP") << "Camera configured: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height
            << " format=MJPG fps=" << config.fps << "\n";
  return true;
}

bool V4L2Camera::InitMmap(std::uint32_t buffer_count) {
  v4l2_requestbuffers req {};
  req.count = buffer_count;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (!IoctlRetry(fd_, VIDIOC_REQBUFS, &req)) {
    RKLOG_ERROR("APP") << "VIDIOC_REQBUFS failed: " << std::strerror(errno) << "\n";
    return false;
  }
  if (req.count < 2) {
    RKLOG_ERROR("APP") << "Camera returned too few buffers: " << req.count << "\n";
    return false;
  }

  buffers_.resize(req.count);
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    v4l2_buffer buf {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = static_cast<std::uint32_t>(i);

    if (!IoctlRetry(fd_, VIDIOC_QUERYBUF, &buf)) {
      RKLOG_ERROR("APP") << "VIDIOC_QUERYBUF failed: " << std::strerror(errno) << "\n";
      return false;
    }

    buffers_[i].length = buf.length;
    buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (buffers_[i].start == MAP_FAILED) {
      buffers_[i].start = nullptr;
      RKLOG_ERROR("APP") << "mmap failed: " << std::strerror(errno) << "\n";
      return false;
    }
  }

  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    v4l2_buffer buf {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = static_cast<std::uint32_t>(i);
    if (!IoctlRetry(fd_, VIDIOC_QBUF, &buf)) {
      RKLOG_ERROR("APP") << "VIDIOC_QBUF failed: " << std::strerror(errno) << "\n";
      return false;
    }
  }

  return true;
}

bool V4L2Camera::StartStreaming() {
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (!IoctlRetry(fd_, VIDIOC_STREAMON, &type)) {
    RKLOG_ERROR("APP") << "VIDIOC_STREAMON failed: " << std::strerror(errno) << "\n";
    return false;
  }
  streaming_ = true;
  return true;
}

void V4L2Camera::StopStreaming() {
  if (!streaming_ || fd_ < 0) {
    return;
  }
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  IoctlRetry(fd_, VIDIOC_STREAMOFF, &type);
  streaming_ = false;
}

bool V4L2Camera::CaptureLoop(const FrameHandler& handler, int frame_limit) {
  int frame_count = 0;
  while (true) {
    pollfd pfd {};
    pfd.fd = fd_;
    pfd.events = POLLIN;

    const int poll_ret = poll(&pfd, 1, 2000);
    if (poll_ret < 0) {
      if (errno == EINTR) {
        continue;
      }
      RKLOG_ERROR("APP") << "poll failed: " << std::strerror(errno) << "\n";
      return false;
    }
    if (poll_ret == 0) {
      RKLOG_ERROR("APP") << "poll timeout while waiting for camera frame\n";
      continue;
    }

    v4l2_buffer buf {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (!IoctlRetry(fd_, VIDIOC_DQBUF, &buf)) {
      if (errno == EAGAIN) {
        continue;
      }
      RKLOG_ERROR("APP") << "VIDIOC_DQBUF failed: " << std::strerror(errno) << "\n";
      return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(buffers_[buf.index].start);
    const bool keep_running = handler(bytes, buf.bytesused, ToMilliseconds(buf.timestamp));

    if (!IoctlRetry(fd_, VIDIOC_QBUF, &buf)) {
      RKLOG_ERROR("APP") << "VIDIOC_QBUF failed after dequeue: " << std::strerror(errno) << "\n";
      return false;
    }

    ++frame_count;
    if (!keep_running || (frame_limit > 0 && frame_count >= frame_limit)) {
      break;
    }
  }
  return true;
}

void V4L2Camera::Close() {
  StopStreaming();

  for (auto& buffer : buffers_) {
    if (buffer.start != nullptr) {
      munmap(buffer.start, buffer.length);
      buffer.start = nullptr;
      buffer.length = 0;
    }
  }
  buffers_.clear();

  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

}  // namespace rk3576_demo
