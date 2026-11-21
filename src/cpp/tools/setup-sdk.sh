#!/usr/bin/env bash
set -euo pipefail
# Pinned SDK sources from rockchip-yolo-cpp tools/setup.sh.
root=/opt/rockchip-build
deps=/opt/rockchip
cache="$root/cache"
mkdir -p "$deps" "$cache"
# Official source snapshots pinned by commit, cached for subsequent builds.
fetch_source()
{
    local name="$1" repo="$2" revision="$3" destination="$cache/$1-$3"
    if [[ ! -f "$destination/.nvr-ready" ]]; then
        echo "Downloading $name ($revision)" >&2
        mkdir -p "$destination" || return 1
        if [[ "$name" == rknn || "$name" == rga ]]; then
            # Fetch only ARM64 runtime/headers, not SDK docs, wheels or other CPUs.
            [[ -d "$destination/.git" ]] || git init "$destination" >&2 || return 1
            git -C "$destination" config remote.origin.url "https://github.com/$repo.git" || return 1
            git -C "$destination" config remote.origin.promisor true || return 1
            git -C "$destination" config remote.origin.partialclonefilter blob:none || return 1
            git -C "$destination" fetch --depth 1 --filter=blob:none \
                origin "$revision" >&2 || return 1
            if [[ "$name" == rknn ]]; then
                git -C "$destination" sparse-checkout set \
                    rknpu2/runtime/Linux/librknn_api/include \
                    rknpu2/runtime/Linux/librknn_api/aarch64 >&2 || return 1
            else
                git -C "$destination" sparse-checkout set include libs/Linux/gcc-aarch64 >&2 || return 1
            fi
            git -C "$destination" checkout --detach "$revision" >&2 || return 1
        else
            curl -fL --retry 3 --connect-timeout 15 -o "$cache/$name-$revision.tar.gz" \
                "https://codeload.github.com/$repo/tar.gz/$revision" || return 1
            tar -xzf "$cache/$name-$revision.tar.gz" --strip-components=1 -C "$destination" || return 1
        fi
        touch "$destination/.nvr-ready" || return 1
    fi
    printf '%s\n' "$destination"
}
copy_sdk()
{
    local name="$1" headers="$2" libraries="$3"
    [[ -d "$headers" && -d "$libraries" ]] || {
        echo "Invalid $name SDK paths: $headers, $libraries" >&2
        exit 1
    }
    mkdir -p "$deps/$name/include" "$deps/$name/lib"
    cp -a "$headers/." "$deps/$name/include/"
    cp -a "$libraries/." "$deps/$name/lib/"
}
sdk="${ROCKCHIP_SDK:-${RK_SDK_PROJECT:+$RK_SDK_PROJECT/third_party}}"
mpp="${MPP_ROOT:-${RK_SDK_PROJECT:+$RK_SDK_PROJECT/build-rk3576/deps/mpp}}"
toolchain=()
[[ -z "${NVR_TOOLCHAIN_FILE:-}" ]] || toolchain=("-DCMAKE_TOOLCHAIN_FILE=$NVR_TOOLCHAIN_FILE")
if [[ ! -f "$deps/rga/include/im2d.hpp" || ! -e "$deps/rga/lib/librga.so" ]]; then
    if [[ -n "$sdk" ]]; then
        copy_sdk rga "$sdk/rga/include" "$sdk/rga/lib"
    else
        source=$(fetch_source rga airockchip/librga 2b32edcb97b601b25683e2941d888c8515da6d55)
        copy_sdk rga "$source/include" "$source/libs/Linux/gcc-aarch64"
    fi
fi
if [[ ! -f "$deps/rknn/include/rknn_api.h" || ! -e "$deps/rknn/lib/librknnrt.so" ]]; then
    if [[ -n "$sdk" ]]; then
        copy_sdk rknn "$sdk/rknn/include" "$sdk/rknn/lib"
    else
        source=$(fetch_source rknn airockchip/rknn-toolkit2 42aa1d426c0a9e0869b6374edba009f7208a1926)
        runtime="$source/rknpu2/runtime/Linux/librknn_api"
        copy_sdk rknn "$runtime/include" "$runtime/aarch64"
    fi
fi
if [[ ! -f "$deps/mpp/include/rockchip/rk_mpi.h" && ! -f "$deps/mpp/include/rk_mpi.h" ]] \
    || [[ ! -e "$deps/mpp/lib/librockchip_mpp.so" ]]; then
    if [[ -n "$mpp" ]]; then
        copy_sdk mpp "$mpp/include" "$mpp/lib"
    else
        source=$(fetch_source mpp rockchip-linux/mpp 7916c3cc000fefc0bdf96636b7d54fe174724f44)
        cmake -S "$source" -B "$cache/mpp-build" "${toolchain[@]}" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$deps/mpp" \
            -DCMAKE_INSTALL_LIBDIR=lib -DRKPLATFORM=ON -DHAVE_DRM=ON -DBUILD_TEST=OFF
        cmake --build "$cache/mpp-build" --parallel "${JOBS:-4}"
        cmake --install "$cache/mpp-build"
    fi
fi

rm -rf "$root"
