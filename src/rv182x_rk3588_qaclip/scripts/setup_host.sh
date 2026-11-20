#!/usr/bin/env bash
# Run on the RK3588 host, never inside the application container.
set -euo pipefail
base=$(cd "$(dirname "$0")/.." && pwd)
if [[ $EUID -ne 0 ]]; then exec sudo bash "$0" "$@"; fi
if [[ $(uname -m) != aarch64 ]]; then echo 'Run this on the RK3588 host.' >&2; exit 1; fi
bdf=${RK182X_DEVICE:-0000:01:00.0}
if [[ ! -d /sys/bus/pci/devices/$bdf ]]; then echo "PCIe device $bdf not found" >&2; exit 1; fi
if [[ $(cat "/sys/bus/pci/devices/$bdf/vendor") != 0x1d87 || $(cat "/sys/bus/pci/devices/$bdf/device") != 0x182a ]]; then
    echo "$bdf is not a supported Rockchip RK182x PCIe device" >&2; exit 1
fi
if [[ ! -c /dev/pcie-rkep-$bdf ]]; then
    test -d "/lib/modules/$(uname -r)/build" || { echo 'Install matching kernel headers first.' >&2; exit 1; }
    (cd "$base/host/pcie-rkep" && make -j2 "KERNELDIR=/lib/modules/$(uname -r)/build")
    insmod "$base/host/pcie-rkep/pcie-rkep.ko"
fi
export LD_LIBRARY_PATH="$base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
case ${1:-} in
    --flash-m2)
        # Explicit opt-in: writes firmware and resets the accelerator.
        (cd "$base/host/firmware" && sha256sum -c SHA256SUMS)
        "$base/host/bin/pcie_upgrade_tool" -s "$bdf" uf "$base/host/firmware/rknn3_rk1820_ext-m2.img" -reset
        ;;
    '') ;;
    *) echo "Usage: $0 [--flash-m2]" >&2; exit 2 ;;
esac
if ! pgrep -f '(^|/)rknn3_transfer_proxy($| )' >/dev/null; then
    start-stop-daemon --start --background --oknodo --exec "$base/host/bin/rknn3_transfer_proxy"
fi
for attempt in 1 2 3 4 5; do
    status=$("$base/host/bin/rknn-smi" info 2>&1 || true)
    if grep -q Online <<< "$status"; then printf '%s\n' "$status"; exit 0; fi
    sleep 1
done
printf '%s\n' "$status" >&2
echo 'Accelerator Offline. Check board type/firmware; --flash-m2 is only for matching M.2 modules.' >&2
exit 1
