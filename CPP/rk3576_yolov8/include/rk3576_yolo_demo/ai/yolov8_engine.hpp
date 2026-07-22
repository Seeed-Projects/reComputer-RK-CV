#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "im2d.hpp"
#include "rknn_api.h"
#include "rk3576_demo/media_types.hpp"
#include "rk3576_yolo_demo/app/app_config_v2.hpp"
#include "rk3576_yolo_demo/common/runtime_types.hpp"

namespace rk3576_yolo_demo {

class Yolov8Engine {
 public:
  Yolov8Engine() = default;
  ~Yolov8Engine();

  void Configure(const AppConfigV2& config);
  bool LoadModel(const std::string& model_path);
  void UnloadModel();
  bool Infer(const rk3576_demo::DecodedFrame& frame, DetectionFrame* detection);

  bool loaded() const { return loaded_; }
  std::string model_path() const { return model_path_; }
  int input_width() const { return input_width_; }
  int input_height() const { return input_height_; }
  std::string LastError() const { return last_error_; }
  std::string TensorSummary() const { return tensor_summary_; }

 private:
  struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int resized_width = 0;
    int resized_height = 0;
  };

  struct CandidateBox {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float score = 0.0f;
    int class_id = -1;
  };

  bool QueryTensorAttributes();
  bool PrepareInputImage(const rk3576_demo::DecodedFrame& frame,
                         std::vector<std::uint8_t>* rgb_buffer,
                         LetterboxInfo* letterbox);
  bool PrepareInputImageWithRga(const rk3576_demo::DecodedFrame& frame,
                                std::vector<std::uint8_t>* rgb_buffer,
                                LetterboxInfo* letterbox);
  bool PrepareInputImageWithCpu(const rk3576_demo::DecodedFrame& frame,
                                std::vector<std::uint8_t>* rgb_buffer,
                                LetterboxInfo* letterbox);
  bool EnsureReusableInputBuffer();
  bool EnsureRgaSourceHandle(const rk3576_demo::DecodedFrame& frame, int src_format);
  bool EnsureRgaDestinationHandle(std::vector<std::uint8_t>* rgb_buffer);
  void ReleaseRgaPreprocessResources();
  bool SampleRgbPixel(const rk3576_demo::DecodedFrame& frame, int x, int y,
                      std::uint8_t* r, std::uint8_t* g, std::uint8_t* b);
  int ToRgaFormat(int mpp_format) const;
  bool PostProcessOutputs(rknn_output* outputs, const LetterboxInfo& letterbox, DetectionFrame* detection);
  bool AppendBranchCandidates(int branch_index, rknn_output* outputs, std::vector<CandidateBox>* candidates);
  bool AppendNchwBranchCandidates(const rknn_tensor_attr& box_attr, const void* box_buf,
                                  const rknn_tensor_attr& score_attr, const void* score_buf,
                                  const rknn_tensor_attr* score_sum_attr, const void* score_sum_buf,
                                  std::vector<CandidateBox>* candidates) const;
  bool AppendNhwcBranchCandidates(const rknn_tensor_attr& box_attr, const void* box_buf,
                                  const rknn_tensor_attr& score_attr, const void* score_buf,
                                  const rknn_tensor_attr* score_sum_attr, const void* score_sum_buf,
                                  std::vector<CandidateBox>* candidates) const;
  float DequantizeValue(const rknn_tensor_attr& attr, const void* data, int index) const;
  float ReadScoreValue(const rknn_tensor_attr& attr, const void* data, int grid_index, int class_index) const;
  float ReadScoreSumValue(const rknn_tensor_attr& attr, const void* data, int grid_index) const;
  float IoU(const CandidateBox& lhs, const CandidateBox& rhs) const;
  void ApplyNms(std::vector<CandidateBox>* candidates) const;
  bool MaybeDumpAiInput(const std::vector<std::uint8_t>& rgb_buffer, std::uint64_t pts_ms);
  void SetError(const std::string& error);

  bool loaded_ = false;
  rknn_context ctx_ = 0;
  rknn_input_output_num io_num_ {};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  bool is_quant_ = false;
  int input_width_ = 640;
  int input_height_ = 640;
  int input_channel_ = 3;
  float conf_threshold_ = 0.25f;
  float nms_threshold_ = 0.45f;
  bool prefer_rga_preprocess_ = true;
  bool rga_preprocess_available_ = true;
  std::vector<std::uint8_t> reusable_input_rgb_;
  AppConfigV2 config_;
  rga_buffer_handle_t cached_rga_src_handle_ = 0;
  rga_buffer_handle_t cached_rga_dst_handle_ = 0;
  int cached_rga_src_fd_ = -1;
  int cached_rga_src_width_ = 0;
  int cached_rga_src_height_ = 0;
  int cached_rga_src_hor_stride_ = 0;
  int cached_rga_src_ver_stride_ = 0;
  int cached_rga_src_format_ = -1;
  void* cached_rga_dst_addr_ = nullptr;
  std::uint64_t ai_input_dump_counter_ = 0;
  std::string model_path_;
  std::string last_error_;
  std::string tensor_summary_;
};

}  // namespace rk3576_yolo_demo
