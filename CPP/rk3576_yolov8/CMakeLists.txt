cmake_minimum_required(VERSION 3.16)

project(rk3576_yolov8tortsp_demo VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RK3576_YOLO_DEMO_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_DIR ${PROJECT_ROOT}/third_party)
set(INCLUDE_DIR ${PROJECT_ROOT}/include)

set(MPP_INCLUDE_DIR ${THIRD_PARTY_DIR}/mpp/include)
set(MPP_LIB_DIR ${THIRD_PARTY_DIR}/mpp/lib/aarch64)
set(RGA_INCLUDE_DIR ${THIRD_PARTY_DIR}/rga/include)
set(RGA_LIB_DIR ${THIRD_PARTY_DIR}/rga/lib/aarch64)
set(RKNN_INCLUDE_DIR ${THIRD_PARTY_DIR}/rknn/include)
set(RKNN_LIB_DIR ${THIRD_PARTY_DIR}/rknn/lib/aarch64)
set(ZLMEDIAKIT_INCLUDE_DIR ${THIRD_PARTY_DIR}/zlmediakit/include)
set(ZLMEDIAKIT_LIB_DIR ${THIRD_PARTY_DIR}/zlmediakit/lib/aarch64)
set(LIBDRM_INCLUDE_DIR ${THIRD_PARTY_DIR}/libdrm/include)
set(LIBDRM_LIB_DIR ${THIRD_PARTY_DIR}/libdrm/lib/aarch64)

find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME}
  src/main.cpp

  src/app/yolo_rtsp_application.cpp
  src/ai/yolov8_engine.cpp
  src/branch/branch_output.cpp
  src/branch/stream_branch.cpp
  src/pipeline/frame_dispatcher.cpp
  src/source/rtsp_source.cpp
  src/source/source_factory.cpp
  src/source/v4l2_source_adapter.cpp

  src/app/pipeline_app.cpp
  src/camera/v4l2_camera.cpp
  src/codec/mpp_decoder.cpp
  src/codec/mpp_encoder.cpp
  src/rga/rga_processor.cpp
  src/rtsp/rtsp_server.cpp
  src/utils/logger.cpp
  src/utils/system_monitor.cpp
  src/watermark/watermark_renderer.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${INCLUDE_DIR}
  ${MPP_INCLUDE_DIR}
  ${RGA_INCLUDE_DIR}
  ${RKNN_INCLUDE_DIR}
  ${ZLMEDIAKIT_INCLUDE_DIR}
  ${LIBDRM_INCLUDE_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  _GNU_SOURCE
)

target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall
  -Wextra
  -Wno-unused-parameter
)

if(RK3576_YOLO_DEMO_WARNINGS_AS_ERRORS)
  target_compile_options(${PROJECT_NAME} PRIVATE -Werror)
endif()

target_link_directories(${PROJECT_NAME} PRIVATE
  ${MPP_LIB_DIR}
  ${RGA_LIB_DIR}
  ${RKNN_LIB_DIR}
  ${ZLMEDIAKIT_LIB_DIR}
  ${LIBDRM_LIB_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  rockchip_mpp
  rga
  rknnrt
  mk_api
  drm
  Threads::Threads
  dl
  rt
)

target_link_options(${PROJECT_NAME} PRIVATE
  -Wl,--allow-shlib-undefined
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${MPP_LIB_DIR};${RGA_LIB_DIR};${RKNN_LIB_DIR};${ZLMEDIAKIT_LIB_DIR};${LIBDRM_LIB_DIR}"
  INSTALL_RPATH "\$ORIGIN/../lib"
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

install(FILES
  ${MPP_LIB_DIR}/librockchip_mpp.so.0
  ${RGA_LIB_DIR}/librga.so
  ${RKNN_LIB_DIR}/librknnrt.so
  ${ZLMEDIAKIT_LIB_DIR}/libmk_api.so
  ${LIBDRM_LIB_DIR}/libdrm.so
  DESTINATION lib
)

install(FILES README.md DESTINATION .)
install(DIRECTORY docs DESTINATION .)
install(DIRECTORY licenses DESTINATION .)
install(DIRECTORY scripts DESTINATION .)
install(DIRECTORY configs DESTINATION .)
install(DIRECTORY model DESTINATION .)
