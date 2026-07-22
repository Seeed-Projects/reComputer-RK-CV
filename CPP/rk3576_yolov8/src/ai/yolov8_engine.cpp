#include "rk3576_yolo_demo/ai/yolov8_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "im2d.hpp"
#include "rockchip/mpp_frame.h"
#include "rk3576_yolo_demo/common/logger.hpp"

namespace rk3576_yolo_demo {

namespace {

constexpr int kClassCount = 80;
constexpr const char* kCocoLabels[kClassCount] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

int ClampToByte(int value) {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return value;
}

void YuvToRgb(int y, int u, int v, std::uint8_t* r, std::uint8_t* g, std::uint8_t* b) {
  const int c = y - 16;
  const int d = u - 128;
  const int e = v - 128;
  *r = static_cast<std::uint8_t>(ClampToByte((298 * c + 409 * e + 128) >> 8));
  *g = static_cast<std::uint8_t>(ClampToByte((298 * c - 100 * d - 208 * e + 128) >> 8));
  *b = static_cast<std::uint8_t>(ClampToByte((298 * c + 516 * d + 128) >> 8));
}

std::uint64_t ToMicroseconds(std::chrono::steady_clock::duration duration) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
}

float ClampFloat(float value, float low, float high) {
  return std::max(low, std::min(high, value));
}

bool EnsureDirectoryExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  if (path == ".") {
    return true;
  }

  std::string current;
  if (path[0] == '/') {
    current = "/";
  }

  std::stringstream ss(path);
  std::string segment;
  while (std::getline(ss, segment, '/')) {
    if (segment.empty() || segment == ".") {
      continue;
    }
    if (!current.empty() && current[current.size() - 1] != '/') {
      current += "/";
    }
    current += segment;

    struct stat st;
    if (stat(current.c_str(), &st) == 0) {
      if (!S_ISDIR(st.st_mode)) {
        return false;
      }
      continue;
    }
    if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

}  // namespace

Yolov8Engine::~Yolov8Engine() {
  UnloadModel();
}

void Yolov8Engine::Configure(const AppConfigV2& config) {
  config_ = config;
}

bool Yolov8Engine::LoadModel(const std::string& model_path) {
  UnloadModel();
  if (model_path.empty()) {
    SetError("Model path is empty");
    return false;
  }

  std::ifstream input(model_path.c_str(), std::ios::binary);
  if (!input.good()) {
    SetError("Model file does not exist yet: " + model_path);
    return false;
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  if (model_data.empty()) {
    SetError("Model file is empty: " + model_path);
    return false;
  }

  int ret = rknn_init(&ctx_, model_data.data(), static_cast<std::uint32_t>(model_data.size()), 0, nullptr);
  if (ret < 0) {
    std::ostringstream oss;
    oss << "rknn_init failed, ret=" << ret;
    SetError(oss.str());
    ctx_ = 0;
    return false;
  }

  model_path_ = model_path;
  if (!QueryTensorAttributes()) {
    UnloadModel();
    return false;
  }

  loaded_ = true;
  last_error_.clear();
  return true;
}

void Yolov8Engine::UnloadModel() {
  ReleaseRgaPreprocessResources();
  if (ctx_ != 0) {
    rknn_destroy(ctx_);
    ctx_ = 0;
  }
  loaded_ = false;
  io_num_ = rknn_input_output_num();
  input_attrs_.clear();
  output_attrs_.clear();
  is_quant_ = false;
  input_width_ = 640;
  input_height_ = 640;
  input_channel_ = 3;
  reusable_input_rgb_.clear();
  model_path_.clear();
  last_error_.clear();
  tensor_summary_.clear();
}

bool Yolov8Engine::Infer(const rk3576_demo::DecodedFrame& frame, DetectionFrame* detection) {
  if (!loaded_) {
    SetError("RKNN model is not loaded");
    return false;
  }
  if (detection == nullptr) {
    SetError("Detection output pointer is null");
    return false;
  }
  if (frame.virt_addr == nullptr && frame.fd < 0) {
    SetError("Decoded frame has neither CPU address nor dmabuf fd");
    return false;
  }

  std::vector<std::uint8_t>& input_rgb = reusable_input_rgb_;
  LetterboxInfo letterbox;
  const auto preprocess_begin = std::chrono::steady_clock::now();
  if (!PrepareInputImage(frame, &input_rgb, &letterbox)) {
    return false;
  }
  const std::uint64_t preprocess_us = ToMicroseconds(std::chrono::steady_clock::now() - preprocess_begin);
  if (!MaybeDumpAiInput(input_rgb, frame.pts_ms)) {
    return false;
  }

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].buf = input_rgb.data();
  inputs[0].size = static_cast<std::uint32_t>(input_rgb.size());
  inputs[0].pass_through = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;

  int ret = rknn_inputs_set(ctx_, 1, inputs);
  if (ret < 0) {
    std::ostringstream oss;
    oss << "rknn_inputs_set failed, ret=" << ret;
    SetError(oss.str());
    return false;
  }

  const auto run_begin = std::chrono::steady_clock::now();
  ret = rknn_run(ctx_, nullptr);
  if (ret < 0) {
    std::ostringstream oss;
    oss << "rknn_run failed, ret=" << ret;
    SetError(oss.str());
    return false;
  }

  std::vector<rknn_output> outputs(io_num_.n_output);
  std::memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].index = static_cast<std::uint32_t>(i);
    outputs[i].want_float = is_quant_ ? 0 : 1;
  }
  ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
  if (ret < 0) {
    std::ostringstream oss;
    oss << "rknn_outputs_get failed, ret=" << ret;
    SetError(oss.str());
    return false;
  }
  if (!PostProcessOutputs(outputs.data(), letterbox, detection)) {
    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
    return false;
  }
  rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

  rknn_perf_run perf_run;
  std::memset(&perf_run, 0, sizeof(perf_run));
  std::uint64_t npu_us = ToMicroseconds(std::chrono::steady_clock::now() - run_begin);
  if (rknn_query(ctx_, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run)) == RKNN_SUCC &&
      perf_run.run_duration > 0) {
    npu_us = static_cast<std::uint64_t>(perf_run.run_duration);
  }

  detection->pts_ms = frame.pts_ms;
  detection->source_width = frame.width;
  detection->source_height = frame.height;
  detection->input_width = input_width_;
  detection->input_height = input_height_;
  detection->preprocess_us = preprocess_us;
  detection->npu_us = npu_us;
  last_error_.clear();
  return true;
}

bool Yolov8Engine::QueryTensorAttributes() {
  int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    std::ostringstream oss;
    oss << "rknn_query io num failed, ret=" << ret;
    SetError(oss.str());
    return false;
  }

  input_attrs_.resize(io_num_.n_input);
  output_attrs_.resize(io_num_.n_output);
  for (std::size_t i = 0; i < input_attrs_.size(); ++i) {
    std::memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
    input_attrs_[i].index = static_cast<std::uint32_t>(i);
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      std::ostringstream oss;
      oss << "rknn_query input attr failed, ret=" << ret << ", index=" << i;
      SetError(oss.str());
      return false;
    }
  }
  for (std::size_t i = 0; i < output_attrs_.size(); ++i) {
    std::memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
    output_attrs_[i].index = static_cast<std::uint32_t>(i);
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      std::ostringstream oss;
      oss << "rknn_query output attr failed, ret=" << ret << ", index=" << i;
      SetError(oss.str());
      return false;
    }
  }

  const rknn_tensor_attr& input = input_attrs_.front();
  is_quant_ = !output_attrs_.empty() &&
              output_attrs_[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
              (output_attrs_[0].type == RKNN_TENSOR_INT8 || output_attrs_[0].type == RKNN_TENSOR_UINT8);
  if (input.fmt == RKNN_TENSOR_NCHW) {
    input_channel_ = static_cast<int>(input.dims[1]);
    input_height_ = static_cast<int>(input.dims[2]);
    input_width_ = static_cast<int>(input.dims[3]);
  } else {
    input_height_ = static_cast<int>(input.dims[1]);
    input_width_ = static_cast<int>(input.dims[2]);
    input_channel_ = static_cast<int>(input.dims[3]);
  }

  std::ostringstream oss;
  oss << "inputs=" << io_num_.n_input << " outputs=" << io_num_.n_output
      << " input_fmt=" << get_format_string(input.fmt)
      << " input_type=" << get_type_string(input.type)
      << " input_dims=" << input_width_ << "x" << input_height_ << "x" << input_channel_;
  tensor_summary_ = oss.str();
  return true;
}

bool Yolov8Engine::PrepareInputImage(const rk3576_demo::DecodedFrame& frame,
                                     std::vector<std::uint8_t>* rgb_buffer,
                                     LetterboxInfo* letterbox) {
  if (prefer_rga_preprocess_ && rga_preprocess_available_ && frame.fd >= 0) {
    if (PrepareInputImageWithRga(frame, rgb_buffer, letterbox)) {
      return true;
    }
    rga_preprocess_available_ = false;
    RKLOG_ERROR("APP") << "[AI] RGA preprocess unavailable, fallback to CPU: " << LastError() << "\n";
  }
  return PrepareInputImageWithCpu(frame, rgb_buffer, letterbox);
}

bool Yolov8Engine::PrepareInputImageWithRga(const rk3576_demo::DecodedFrame& frame,
                                            std::vector<std::uint8_t>* rgb_buffer,
                                            LetterboxInfo* letterbox) {
  if (rgb_buffer == nullptr || letterbox == nullptr) {
    return false;
  }
  if (input_width_ <= 0 || input_height_ <= 0 || input_channel_ != 3) {
    SetError("Invalid RKNN input shape for preprocess");
    return false;
  }
  const int src_format = ToRgaFormat(frame.format);
  if (src_format < 0) {
    SetError("Unsupported decoded format for RGA preprocess: " + std::to_string(frame.format));
    return false;
  }

  if (!EnsureReusableInputBuffer()) {
    return false;
  }
  rgb_buffer->resize(static_cast<std::size_t>(input_width_ * input_height_ * 3));
  std::fill(rgb_buffer->begin(), rgb_buffer->end(), static_cast<std::uint8_t>(114));
  letterbox->scale = std::min(static_cast<float>(input_width_) / frame.width,
                              static_cast<float>(input_height_) / frame.height);
  letterbox->resized_width = std::max(1, static_cast<int>(frame.width * letterbox->scale + 0.5f));
  letterbox->resized_height = std::max(1, static_cast<int>(frame.height * letterbox->scale + 0.5f));
  letterbox->pad_x = (input_width_ - letterbox->resized_width) / 2;
  letterbox->pad_y = (input_height_ - letterbox->resized_height) / 2;

  if (!EnsureRgaSourceHandle(frame, src_format)) {
    return false;
  }
  if (!EnsureRgaDestinationHandle(rgb_buffer)) {
    return false;
  }

  rga_buffer_t src = wrapbuffer_handle(cached_rga_src_handle_, frame.width, frame.height, src_format,
                                       frame.hor_stride, frame.ver_stride);
  rga_buffer_t dst = wrapbuffer_handle(cached_rga_dst_handle_, input_width_, input_height_, RK_FORMAT_RGB_888,
                                       input_width_, input_height_);
  im_rect src_rect {};
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.width = frame.width;
  src_rect.height = frame.height;

  im_rect dst_rect {};
  dst_rect.x = letterbox->pad_x;
  dst_rect.y = letterbox->pad_y;
  dst_rect.width = letterbox->resized_width;
  dst_rect.height = letterbox->resized_height;

  IM_STATUS check_status = imcheck(src, dst, src_rect, dst_rect);
  IM_STATUS process_status = IM_STATUS_SUCCESS;
  if (check_status == IM_STATUS_NOERROR) {
    process_status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
  }
  if (check_status != IM_STATUS_NOERROR) {
    std::ostringstream oss;
    oss << "RGA imcheck failed: " << imStrError(check_status)
        << " src=" << frame.width << "x" << frame.height
        << " stride=" << frame.hor_stride << "x" << frame.ver_stride
        << " fmt=" << src_format
        << " dst=" << input_width_ << "x" << input_height_
        << " rect=(" << dst_rect.x << "," << dst_rect.y
        << "," << dst_rect.width << "," << dst_rect.height << ")";
    SetError(oss.str());
    return false;
  }
  if (process_status != IM_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << "RGA improcess failed: " << imStrError(process_status)
        << " src=" << frame.width << "x" << frame.height
        << " stride=" << frame.hor_stride << "x" << frame.ver_stride
        << " fmt=" << src_format
        << " dst=" << input_width_ << "x" << input_height_
        << " rect=(" << dst_rect.x << "," << dst_rect.y
        << "," << dst_rect.width << "," << dst_rect.height << ")";
    SetError(oss.str());
    return false;
  }
  last_error_.clear();
  return true;
}

bool Yolov8Engine::PrepareInputImageWithCpu(const rk3576_demo::DecodedFrame& frame,
                                            std::vector<std::uint8_t>* rgb_buffer,
                                            LetterboxInfo* letterbox) {
  if (rgb_buffer == nullptr || letterbox == nullptr) {
    return false;
  }
  if (input_width_ <= 0 || input_height_ <= 0 || input_channel_ != 3) {
    SetError("Invalid RKNN input shape for preprocess");
    return false;
  }

  if (!EnsureReusableInputBuffer()) {
    return false;
  }
  rgb_buffer->resize(static_cast<std::size_t>(input_width_ * input_height_ * 3));
  std::fill(rgb_buffer->begin(), rgb_buffer->end(), static_cast<std::uint8_t>(114));
  letterbox->scale = std::min(static_cast<float>(input_width_) / frame.width,
                              static_cast<float>(input_height_) / frame.height);
  letterbox->resized_width = std::max(1, static_cast<int>(frame.width * letterbox->scale + 0.5f));
  letterbox->resized_height = std::max(1, static_cast<int>(frame.height * letterbox->scale + 0.5f));
  letterbox->pad_x = (input_width_ - letterbox->resized_width) / 2;
  letterbox->pad_y = (input_height_ - letterbox->resized_height) / 2;

  for (int y = 0; y < letterbox->resized_height; ++y) {
    const int src_y = std::min(frame.height - 1, static_cast<int>(y / letterbox->scale));
    for (int x = 0; x < letterbox->resized_width; ++x) {
      const int src_x = std::min(frame.width - 1, static_cast<int>(x / letterbox->scale));
      std::uint8_t r = 0;
      std::uint8_t g = 0;
      std::uint8_t b = 0;
      if (!SampleRgbPixel(frame, src_x, src_y, &r, &g, &b)) {
        return false;
      }
      const int dst_x = x + letterbox->pad_x;
      const int dst_y = y + letterbox->pad_y;
      const std::size_t dst_index = static_cast<std::size_t>((dst_y * input_width_ + dst_x) * 3);
      (*rgb_buffer)[dst_index + 0] = r;
      (*rgb_buffer)[dst_index + 1] = g;
      (*rgb_buffer)[dst_index + 2] = b;
    }
  }
  return true;
}

bool Yolov8Engine::EnsureReusableInputBuffer() {
  if (input_width_ <= 0 || input_height_ <= 0 || input_channel_ != 3) {
    SetError("Invalid RKNN input shape for preprocess");
    return false;
  }
  const std::size_t required_size = static_cast<std::size_t>(input_width_ * input_height_ * 3);
  if (reusable_input_rgb_.size() != required_size) {
    reusable_input_rgb_.resize(required_size);
    if (cached_rga_dst_handle_ != 0) {
      releasebuffer_handle(cached_rga_dst_handle_);
      cached_rga_dst_handle_ = 0;
      cached_rga_dst_addr_ = nullptr;
    }
  }
  return true;
}

bool Yolov8Engine::EnsureRgaSourceHandle(const rk3576_demo::DecodedFrame& frame, int src_format) {
  const bool unchanged = cached_rga_src_handle_ != 0 && cached_rga_src_fd_ == frame.fd &&
                         cached_rga_src_width_ == frame.width && cached_rga_src_height_ == frame.height &&
                         cached_rga_src_hor_stride_ == frame.hor_stride &&
                         cached_rga_src_ver_stride_ == frame.ver_stride &&
                         cached_rga_src_format_ == src_format;
  if (unchanged) {
    return true;
  }

  if (cached_rga_src_handle_ != 0) {
    releasebuffer_handle(cached_rga_src_handle_);
    cached_rga_src_handle_ = 0;
  }
  cached_rga_src_handle_ = importbuffer_fd(frame.fd, frame.hor_stride, frame.ver_stride, src_format);
  if (cached_rga_src_handle_ == 0) {
    SetError(std::string("importbuffer_fd for AI source failed: ") + imStrError());
    return false;
  }
  cached_rga_src_fd_ = frame.fd;
  cached_rga_src_width_ = frame.width;
  cached_rga_src_height_ = frame.height;
  cached_rga_src_hor_stride_ = frame.hor_stride;
  cached_rga_src_ver_stride_ = frame.ver_stride;
  cached_rga_src_format_ = src_format;
  return true;
}

bool Yolov8Engine::EnsureRgaDestinationHandle(std::vector<std::uint8_t>* rgb_buffer) {
  if (rgb_buffer == nullptr || rgb_buffer->empty()) {
    SetError("AI preprocess destination buffer is empty");
    return false;
  }
  if (cached_rga_dst_handle_ != 0 && cached_rga_dst_addr_ == rgb_buffer->data()) {
    return true;
  }

  if (cached_rga_dst_handle_ != 0) {
    releasebuffer_handle(cached_rga_dst_handle_);
    cached_rga_dst_handle_ = 0;
    cached_rga_dst_addr_ = nullptr;
  }
  cached_rga_dst_handle_ = importbuffer_virtualaddr(rgb_buffer->data(), input_width_, input_height_, RK_FORMAT_RGB_888);
  if (cached_rga_dst_handle_ == 0) {
    SetError(std::string("importbuffer_virtualaddr for AI dst failed: ") + imStrError());
    return false;
  }
  cached_rga_dst_addr_ = rgb_buffer->data();
  return true;
}

void Yolov8Engine::ReleaseRgaPreprocessResources() {
  if (cached_rga_src_handle_ != 0) {
    releasebuffer_handle(cached_rga_src_handle_);
    cached_rga_src_handle_ = 0;
  }
  if (cached_rga_dst_handle_ != 0) {
    releasebuffer_handle(cached_rga_dst_handle_);
    cached_rga_dst_handle_ = 0;
  }
  cached_rga_src_fd_ = -1;
  cached_rga_src_width_ = 0;
  cached_rga_src_height_ = 0;
  cached_rga_src_hor_stride_ = 0;
  cached_rga_src_ver_stride_ = 0;
  cached_rga_src_format_ = -1;
  cached_rga_dst_addr_ = nullptr;
}

int Yolov8Engine::ToRgaFormat(int mpp_format) const {
  const int base_format = mpp_format & MPP_FRAME_FMT_MASK;
  switch (base_format) {
    case MPP_FMT_YUV420SP:
      return RK_FORMAT_YCbCr_420_SP;
    case MPP_FMT_YUV420SP_VU:
      return RK_FORMAT_YCrCb_420_SP;
    case MPP_FMT_YUV422SP:
      return RK_FORMAT_YCbCr_422_SP;
    case MPP_FMT_YUV422SP_VU:
      return RK_FORMAT_YCrCb_422_SP;
    default:
      return -1;
  }
}

bool Yolov8Engine::SampleRgbPixel(const rk3576_demo::DecodedFrame& frame, int x, int y,
                                  std::uint8_t* r, std::uint8_t* g, std::uint8_t* b) {
  if (frame.virt_addr == nullptr || r == nullptr || g == nullptr || b == nullptr) {
    if (frame.virt_addr == nullptr) {
      SetError("CPU AI preprocess source frame has null virt_addr");
    } else {
      SetError("CPU AI preprocess received null RGB output pointer");
    }
    return false;
  }
  const int src_fmt = frame.format & MPP_FRAME_FMT_MASK;
  if (src_fmt != MPP_FMT_YUV420SP && src_fmt != MPP_FMT_YUV420SP_VU &&
      src_fmt != MPP_FMT_YUV422SP && src_fmt != MPP_FMT_YUV422SP_VU) {
    SetError("Unsupported decoded format for AI preprocess: " + std::to_string(frame.format));
    return false;
  }

  const auto* y_plane = static_cast<const std::uint8_t*>(frame.virt_addr);
  const auto* uv_plane = y_plane + static_cast<std::size_t>(frame.hor_stride * frame.ver_stride);
  const int y_value = y_plane[static_cast<std::size_t>(y * frame.hor_stride + x)];
  const int uv_row = (src_fmt == MPP_FMT_YUV420SP || src_fmt == MPP_FMT_YUV420SP_VU) ? (y / 2) : y;
  const int uv_col = x & ~1;
  const std::size_t uv_index = static_cast<std::size_t>(uv_row * frame.hor_stride + uv_col);
  const bool vu_order = (src_fmt == MPP_FMT_YUV420SP_VU || src_fmt == MPP_FMT_YUV422SP_VU);
  const int u_value = vu_order ? uv_plane[uv_index + 1] : uv_plane[uv_index + 0];
  const int v_value = vu_order ? uv_plane[uv_index + 0] : uv_plane[uv_index + 1];
  YuvToRgb(y_value, u_value, v_value, r, g, b);
  return true;
}

void Yolov8Engine::SetError(const std::string& error) {
  last_error_ = error;
}

bool Yolov8Engine::PostProcessOutputs(rknn_output* outputs, const LetterboxInfo& letterbox, DetectionFrame* detection) {
  if (outputs == nullptr || detection == nullptr) {
    SetError("Post process input is null");
    return false;
  }
  if (io_num_.n_output < 6 || (io_num_.n_output % 3) != 0) {
    SetError("Unsupported YOLOv8 output count: " + std::to_string(io_num_.n_output));
    return false;
  }

  std::vector<CandidateBox> candidates;
  for (int branch = 0; branch < static_cast<int>(io_num_.n_output / 3); ++branch) {
    if (!AppendBranchCandidates(branch, outputs, &candidates)) {
      return false;
    }
  }

  ApplyNms(&candidates);
  detection->boxes.clear();
  detection->boxes.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const CandidateBox& candidate = candidates[i];
    DetectionBox box;
    const float x1 = candidate.x - letterbox.pad_x;
    const float y1 = candidate.y - letterbox.pad_y;
    const float x2 = x1 + candidate.width;
    const float y2 = y1 + candidate.height;
    box.x = static_cast<int>(ClampFloat(x1, 0.0f, static_cast<float>(input_width_)) / letterbox.scale);
    box.y = static_cast<int>(ClampFloat(y1, 0.0f, static_cast<float>(input_height_)) / letterbox.scale);
    const int right = static_cast<int>(ClampFloat(x2, 0.0f, static_cast<float>(input_width_)) / letterbox.scale);
    const int bottom = static_cast<int>(ClampFloat(y2, 0.0f, static_cast<float>(input_height_)) / letterbox.scale);
    box.width = std::max(0, right - box.x);
    box.height = std::max(0, bottom - box.y);
    box.class_id = candidate.class_id;
    box.score = candidate.score;
    if (box.class_id >= 0 && box.class_id < kClassCount) {
      box.class_name = kCocoLabels[box.class_id];
    } else {
      box.class_name = "unknown";
    }
    if (box.width > 1 && box.height > 1) {
      detection->boxes.push_back(box);
    }
  }
  return true;
}

bool Yolov8Engine::AppendBranchCandidates(int branch_index, rknn_output* outputs, std::vector<CandidateBox>* candidates) {
  const int output_per_branch = static_cast<int>(io_num_.n_output / 3);
  const int box_index = branch_index * output_per_branch;
  const int score_index = box_index + 1;
  const int score_sum_index = output_per_branch == 3 ? (box_index + 2) : -1;
  const rknn_tensor_attr& box_attr = output_attrs_[box_index];
  const rknn_tensor_attr& score_attr = output_attrs_[score_index];
  const rknn_tensor_attr* score_sum_attr = score_sum_index >= 0 ? &output_attrs_[score_sum_index] : nullptr;
  const void* score_sum_buf = score_sum_index >= 0 ? outputs[score_sum_index].buf : nullptr;

  if (box_attr.fmt == RKNN_TENSOR_NHWC) {
    return AppendNhwcBranchCandidates(box_attr, outputs[box_index].buf, score_attr, outputs[score_index].buf,
                                      score_sum_attr, score_sum_buf, candidates);
  }
  return AppendNchwBranchCandidates(box_attr, outputs[box_index].buf, score_attr, outputs[score_index].buf,
                                    score_sum_attr, score_sum_buf, candidates);
}

bool Yolov8Engine::AppendNchwBranchCandidates(const rknn_tensor_attr& box_attr, const void* box_buf,
                                              const rknn_tensor_attr& score_attr, const void* score_buf,
                                              const rknn_tensor_attr* score_sum_attr, const void* score_sum_buf,
                                              std::vector<CandidateBox>* candidates) const {
  const int grid_h = static_cast<int>(box_attr.dims[2]);
  const int grid_w = static_cast<int>(box_attr.dims[3]);
  const int grid_len = grid_h * grid_w;
  const int dfl_len = static_cast<int>(box_attr.dims[1] / 4);
  const int stride = input_height_ / grid_h;

  for (int i = 0; i < grid_h; ++i) {
    for (int j = 0; j < grid_w; ++j) {
      const int grid_index = i * grid_w + j;
      if (score_sum_attr != nullptr && score_sum_buf != nullptr &&
          ReadScoreSumValue(*score_sum_attr, score_sum_buf, grid_index) < conf_threshold_) {
        continue;
      }

      int best_class = -1;
      float best_score = 0.0f;
      for (int c = 0; c < kClassCount; ++c) {
        const float score = ReadScoreValue(score_attr, score_buf, grid_index, c);
        if (score > conf_threshold_ && score > best_score) {
          best_score = score;
          best_class = c;
        }
      }
      if (best_class < 0) {
        continue;
      }

      float box_dfl[64];
      for (int k = 0; k < dfl_len * 4; ++k) {
        box_dfl[k] = DequantizeValue(box_attr, box_buf, grid_index + k * grid_len);
      }
      float distance[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int side = 0; side < 4; ++side) {
        float max_val = -std::numeric_limits<float>::max();
        for (int k = 0; k < dfl_len; ++k) {
          max_val = std::max(max_val, box_dfl[side * dfl_len + k]);
        }
        float sum = 0.0f;
        float weighted = 0.0f;
        for (int k = 0; k < dfl_len; ++k) {
          const float exp_val = std::exp(box_dfl[side * dfl_len + k] - max_val);
          sum += exp_val;
          weighted += exp_val * k;
        }
        distance[side] = weighted / sum;
      }

      CandidateBox candidate;
      const float x1 = (-distance[0] + j + 0.5f) * stride;
      const float y1 = (-distance[1] + i + 0.5f) * stride;
      const float x2 = (distance[2] + j + 0.5f) * stride;
      const float y2 = (distance[3] + i + 0.5f) * stride;
      candidate.x = x1;
      candidate.y = y1;
      candidate.width = x2 - x1;
      candidate.height = y2 - y1;
      candidate.score = best_score;
      candidate.class_id = best_class;
      candidates->push_back(candidate);
    }
  }
  return true;
}

bool Yolov8Engine::AppendNhwcBranchCandidates(const rknn_tensor_attr& box_attr, const void* box_buf,
                                              const rknn_tensor_attr& score_attr, const void* score_buf,
                                              const rknn_tensor_attr* score_sum_attr, const void* score_sum_buf,
                                              std::vector<CandidateBox>* candidates) const {
  const int grid_h = static_cast<int>(box_attr.dims[1]);
  const int grid_w = static_cast<int>(box_attr.dims[2]);
  const int dfl_len = static_cast<int>(box_attr.dims[3] / 4);
  const int stride = input_height_ / grid_h;

  for (int i = 0; i < grid_h; ++i) {
    for (int j = 0; j < grid_w; ++j) {
      const int grid_index = i * grid_w + j;
      if (score_sum_attr != nullptr && score_sum_buf != nullptr &&
          ReadScoreSumValue(*score_sum_attr, score_sum_buf, grid_index) < conf_threshold_) {
        continue;
      }

      int best_class = -1;
      float best_score = 0.0f;
      for (int c = 0; c < kClassCount; ++c) {
        const float score = ReadScoreValue(score_attr, score_buf, grid_index, c);
        if (score > conf_threshold_ && score > best_score) {
          best_score = score;
          best_class = c;
        }
      }
      if (best_class < 0) {
        continue;
      }

      const int base = grid_index * 4 * dfl_len;
      float distance[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int side = 0; side < 4; ++side) {
        float max_val = -std::numeric_limits<float>::max();
        for (int k = 0; k < dfl_len; ++k) {
          max_val = std::max(max_val, DequantizeValue(box_attr, box_buf, base + side * dfl_len + k));
        }
        float sum = 0.0f;
        float weighted = 0.0f;
        for (int k = 0; k < dfl_len; ++k) {
          const float value = DequantizeValue(box_attr, box_buf, base + side * dfl_len + k);
          const float exp_val = std::exp(value - max_val);
          sum += exp_val;
          weighted += exp_val * k;
        }
        distance[side] = weighted / sum;
      }

      CandidateBox candidate;
      const float x1 = (-distance[0] + j + 0.5f) * stride;
      const float y1 = (-distance[1] + i + 0.5f) * stride;
      const float x2 = (distance[2] + j + 0.5f) * stride;
      const float y2 = (distance[3] + i + 0.5f) * stride;
      candidate.x = x1;
      candidate.y = y1;
      candidate.width = x2 - x1;
      candidate.height = y2 - y1;
      candidate.score = best_score;
      candidate.class_id = best_class;
      candidates->push_back(candidate);
    }
  }
  return true;
}

float Yolov8Engine::DequantizeValue(const rknn_tensor_attr& attr, const void* data, int index) const {
  if (attr.type == RKNN_TENSOR_FLOAT16) {
    return reinterpret_cast<const std::uint16_t*>(data)[index];
  }
  if (attr.type == RKNN_TENSOR_FLOAT32) {
    return reinterpret_cast<const float*>(data)[index];
  }
  if (attr.type == RKNN_TENSOR_INT8) {
    return (reinterpret_cast<const std::int8_t*>(data)[index] - attr.zp) * attr.scale;
  }
  if (attr.type == RKNN_TENSOR_UINT8) {
    return (reinterpret_cast<const std::uint8_t*>(data)[index] - attr.zp) * attr.scale;
  }
  return 0.0f;
}

float Yolov8Engine::ReadScoreValue(const rknn_tensor_attr& attr, const void* data, int grid_index, int class_index) const {
  if (attr.fmt == RKNN_TENSOR_NHWC) {
    return DequantizeValue(attr, data, grid_index * kClassCount + class_index);
  }
  const int grid_len = static_cast<int>(attr.dims[2] * attr.dims[3]);
  return DequantizeValue(attr, data, grid_index + class_index * grid_len);
}

float Yolov8Engine::ReadScoreSumValue(const rknn_tensor_attr& attr, const void* data, int grid_index) const {
  return DequantizeValue(attr, data, grid_index);
}

float Yolov8Engine::IoU(const CandidateBox& lhs, const CandidateBox& rhs) const {
  const float left = std::max(lhs.x, rhs.x);
  const float top = std::max(lhs.y, rhs.y);
  const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const float inter_w = std::max(0.0f, right - left);
  const float inter_h = std::max(0.0f, bottom - top);
  const float intersection = inter_w * inter_h;
  const float union_area = lhs.width * lhs.height + rhs.width * rhs.height - intersection;
  if (union_area <= 0.0f) {
    return 0.0f;
  }
  return intersection / union_area;
}

bool Yolov8Engine::MaybeDumpAiInput(const std::vector<std::uint8_t>& rgb_buffer, std::uint64_t pts_ms) {
  if (config_.dump_ai_input_dir.empty() || config_.dump_ai_input_every <= 0) {
    return true;
  }

  ++ai_input_dump_counter_;
  if ((ai_input_dump_counter_ % static_cast<std::uint64_t>(config_.dump_ai_input_every)) != 0) {
    return true;
  }
  if (rgb_buffer.size() != static_cast<std::size_t>(input_width_ * input_height_ * input_channel_)) {
    SetError("AI input rgb buffer size mismatch for dump");
    return false;
  }
  if (!EnsureDirectoryExists(config_.dump_ai_input_dir)) {
    SetError("Failed to create ai input dump dir: " + config_.dump_ai_input_dir);
    return false;
  }

  std::ostringstream path;
  path << config_.dump_ai_input_dir;
  if (!config_.dump_ai_input_dir.empty() && config_.dump_ai_input_dir[config_.dump_ai_input_dir.size() - 1] != '/') {
    path << "/";
  }
  path << "yolov8_demo_ai_input_" << ai_input_dump_counter_ << "_pts_" << pts_ms
       << "_" << input_width_ << "x" << input_height_ << ".ppm";

  std::ofstream output(path.str().c_str(), std::ios::binary);
  if (!output.good()) {
    SetError("Failed to open ai input dump file: " + path.str());
    return false;
  }
  output << "P6\n" << input_width_ << " " << input_height_ << "\n255\n";
  output.write(reinterpret_cast<const char*>(rgb_buffer.data()), static_cast<std::streamsize>(rgb_buffer.size()));
  if (!output.good()) {
    SetError("Failed to write ai input dump file: " + path.str());
    return false;
  }

  if (config_.detail_info) {
    RKLOG_INFO("APP") << "[AI-DUMP] saved input rgb to " << path.str()
                      << " pts_ms=" << pts_ms
                      << " frame_index=" << ai_input_dump_counter_ << "\n";
  }
  return true;
}

void Yolov8Engine::ApplyNms(std::vector<CandidateBox>* candidates) const {
  std::sort(candidates->begin(), candidates->end(), [](const CandidateBox& lhs, const CandidateBox& rhs) {
    return lhs.score > rhs.score;
  });

  std::vector<CandidateBox> kept;
  for (std::size_t i = 0; i < candidates->size(); ++i) {
    const CandidateBox& candidate = (*candidates)[i];
    bool suppressed = false;
    for (std::size_t j = 0; j < kept.size(); ++j) {
      if (kept[j].class_id == candidate.class_id && IoU(kept[j], candidate) > nms_threshold_) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
    }
  }
  candidates->swap(kept);
}

}  // namespace rk3576_yolo_demo
