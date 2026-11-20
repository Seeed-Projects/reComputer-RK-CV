cmake_minimum_required(VERSION 3.18)
project(rk_cpp_api LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libavdevice)
set(ROCKCHIP_SDK /opt/rockchip CACHE PATH "Rockchip SDK prefix")
find_path(RGA_INCLUDE im2d.hpp HINTS ${ROCKCHIP_SDK}/rga/include REQUIRED)
find_path(RKNN_INCLUDE rknn_api.h HINTS ${ROCKCHIP_SDK}/rknn/include REQUIRED)
find_path(MPP_INCLUDE rk_mpi.h HINTS ${ROCKCHIP_SDK}/mpp/include PATH_SUFFIXES rockchip REQUIRED)
find_library(RGA_LIB rga HINTS ${ROCKCHIP_SDK}/rga/lib REQUIRED)
find_library(RKNN_LIB rknnrt HINTS ${ROCKCHIP_SDK}/rknn/lib REQUIRED)
find_library(MPP_LIB rockchip_mpp HINTS ${ROCKCHIP_SDK}/mpp/lib REQUIRED)
pybind11_add_module(_native
    native/bindings.cpp
    native/engine/geometry.cpp
    native/engine/postprocess.cpp
    native/engine/frame.cpp
    native/engine/frame_queue.cpp
    native/engine/dma_buffer.cpp
    native/engine/decoder.cpp
    native/engine/preprocess.cpp
    native/engine/task_model.cpp
    native/engine/task_decode.cpp
    native/engine/task_dense.cpp
    native/engine/task_json.cpp
    native/engine/native_output.cpp
    native/engine/detector.cpp
    native/engine/detector_pool.cpp
    native/engine/jpeg_preview.cpp)
target_include_directories(_native PRIVATE native ${RGA_INCLUDE} ${RKNN_INCLUDE} ${MPP_INCLUDE})
target_link_libraries(_native PRIVATE PkgConfig::AV Threads::Threads nlohmann_json::nlohmann_json ${RGA_LIB} ${RKNN_LIB} ${MPP_LIB})
install(TARGETS _native LIBRARY DESTINATION rkapi)
