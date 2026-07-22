#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
INSTALL_DIR="${ROOT_DIR}/install"
TOOLCHAIN_FILE="${ROOT_DIR}/cmake/toolchain-aarch64-linux-gnu.cmake"

GCC_COMPILER_DEFAULT="/home/parker/Projects/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu"
export GCC_COMPILER="${GCC_COMPILER:-${GCC_COMPILER_DEFAULT}}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}"

echo "Build finished."
echo "Install directory: ${INSTALL_DIR}"
