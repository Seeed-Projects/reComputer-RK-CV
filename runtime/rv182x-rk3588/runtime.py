"""RK1820 single-card bootstrap, shared by all RKNN3 model images."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import stat
import subprocess
import sys
import time

ROOT = Path('/opt/rk182x-runtime')
PCI_ROOT = Path('/sys/bus/pci/devices')
MISC_ROOT = Path('/sys/class/misc')
SDK_VERSION = '1.1.0'
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


class Failure(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def log(event, **fields):
    print(json.dumps(dict(component='rk182x-init', event=event, **fields)), flush=True)


def run(args, timeout=30):
    try:
        p = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Failure('COMMAND_TIMEOUT', 'Timed out: ' + str(args[0]))
    return p.returncode, ANSI.sub('', p.stdout + p.stderr)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_assets(root=ROOT):
    for line in (root / 'ASSET_SHA256SUMS').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (root / relative.lstrip('*')).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha256(path) != expected:
            raise Failure('ASSET_CHECKSUM_FAILED', 'Invalid bundled asset: ' + relative)


def find_card():
    cards = []
    for path in PCI_ROOT.glob('*'):
        try:
            if (path / 'vendor').read_text().strip() == '0x1d87' and (path / 'device').read_text().strip() == '0x182a':
                cards.append(path.name)
        except OSError:
            continue
    if not cards:
        raise Failure('PCI_DEVICE_NOT_FOUND', 'No RK1820 PCIe device (1d87:182a). Check power, installation and PCIe configuration.')
    if len(cards) != 1:
        raise Failure('MULTIPLE_CARDS_UNSUPPORTED', 'This release supports one RK1820 card per host.')
    bdf = cards[0]
    if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', bdf):
        raise Failure('INVALID_PCI_ADDRESS', 'Invalid PCI address')
    requested = os.environ.get('RK182X_DEVICE')
    if requested and requested != bdf:
        raise Failure('PCI_ADDRESS_MISMATCH', 'RK182X_DEVICE does not match the detected card: ' + bdf)
    return bdf


def proxy_port_open():
    try:
        with socket.create_connection(('127.0.0.1', 18898), timeout=1):
            return True
    except OSError:
        return False


class Runtime:
    def __init__(self, root=ROOT):
        self.root = root
        self.proxy = None
        self.child = None
        self.lock = None
        self.shared_lock = os.path.ismount('/run/lock')
        self.bdf = None
        self.kernel = platform.release()
        self.recovering = False
        self.stop_requested = False

    def acquire(self):
        Path('/run/lock').mkdir(parents=True, exist_ok=True)
        self.lock = open('/run/lock/rk182x-' + self.bdf.replace(':', '_') + '.lock', 'a')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Failure('DEVICE_BUSY', 'Another managed model container owns this card. Stop it before switching models.')
        if not self.shared_lock:
            log('LOCAL_LOCK_ONLY', guidance='Mount /run/lock:/run/lock for cross-container exclusion. Firmware recovery requires this mount.')

    def expose_node(self):
        """A node created after Docker starts is not automatically added to /dev."""
        name = 'pcie-rkep-' + self.bdf
        dev = MISC_ROOT / name / 'dev'
        node = Path('/dev') / name
        if not dev.is_file():
            raise Failure('DRIVER_NODE_MISSING', 'Driver did not register ' + name)
        major, minor = map(int, dev.read_text().strip().split(':'))
        if not node.exists():
            try:
                os.mknod(str(node), stat.S_IFCHR | 0o600, os.makedev(major, minor))
            except OSError as e:
                raise Failure('DEVICE_PERMISSION_DENIED', 'Cannot create device node; use --privileged. ' + str(e))
        mode = node.stat()
        if not stat.S_ISCHR(mode.st_mode) or mode.st_rdev != os.makedev(major, minor):
            raise Failure('DEVICE_NODE_MISMATCH', 'Existing device node does not match the loaded driver')
        try:
            fd = os.open(str(node), os.O_RDWR)
            os.close(fd)
        except OSError as e:
            raise Failure('DEVICE_PERMISSION_DENIED', 'Cannot open card; use --privileged. ' + str(e))

    def load_driver(self):
        if (MISC_ROOT / ('pcie-rkep-' + self.bdf) / 'dev').is_file():
            self.expose_node()
            log('DRIVER_REUSED', kernel=self.kernel)
            return
        manifest = json.loads((self.root / 'drivers/manifest.json').read_text())
        profile = manifest.get(self.kernel)
        if profile:
            module = self.root / 'drivers' / profile['file']
            if sha256(module) != profile['sha256']:
                raise Failure('DRIVER_CHECKSUM_FAILED', 'Bundled module checksum mismatch')
            code, out = run(['modinfo', '-F', 'vermagic', module])
            if code or out.strip() != profile['vermagic']:
                raise Failure('DRIVER_ABI_MISMATCH', 'Module vermagic does not match its verified profile')
            code, out = run(['insmod', module])
            if not code:
                self.expose_node()
                log('DRIVER_LOADED', source='verified-profile', kernel=self.kernel)
                return
            log('PROFILE_LOAD_FAILED', detail=out[-1500:])
        headers = Path('/lib/modules') / self.kernel / 'build'
        if not (headers / 'Makefile').is_file():
            raise Failure('KERNEL_HEADERS_MISSING', 'No matching kernel headers. Mount /lib/modules:/lib/modules:ro and /usr/src:/usr/src:ro; install headers for the running vendor kernel.')
        import shutil
        build = Path('/tmp/rk182x-driver-build')
        if build.exists():
            shutil.rmtree(str(build))
        shutil.copytree(str(self.root / 'host/pcie-rkep'), str(build), ignore=shutil.ignore_patterns('*.o', '*.ko', '.*.cmd', 'Module.symvers', 'modules.order'))
        code, out = run(['make', '-C', build, '-j2', 'KERNELDIR=' + str(headers)], timeout=120)
        if code:
            raise Failure('DRIVER_BUILD_FAILED', 'Cannot build against this kernel; use vendor-matched headers/toolchain. ' + out[-2500:])
        module = build / 'pcie-rkep.ko'
        code, out = run(['insmod', module])
        if code:
            raise Failure('DRIVER_LOAD_FAILED', 'Module rejected; no forced loading was attempted. ' + out[-2000:])
        self.expose_node()
        log('DRIVER_LOADED', source='built-from-source', kernel=self.kernel)

    def status(self):
        code, text = run([self.root / 'host/bin/rknn-smi', 'info'], timeout=10)
        # SDK 1.1.0's smi prints device index and status in the SAME cell.
        row = re.search(r'^\|\s*\d+(?:\s*\|\s*|\s+)(Online|Offline)\s*\|\s*([^\s|]+)', text, re.M)
        if row is None:
            raise Failure('CARD_STATUS_QUERY_FAILED' if code else 'CARD_STATUS_UNKNOWN',
                          'Unrecognized card status; no firmware write is allowed. ' + text[-1800:])
        # This bundled SDK utility can return nonzero even for a complete
        # Online/OK table. Require the explicit table, then verify real inference.
        if code:
            log('SMI_NONSTANDARD_EXIT', exit_code=code, status=row[1], health=row[2])
        if row[1] == 'Online' and row[2] != 'OK':
            raise Failure('CARD_HEALTH_ERROR', 'Card is Online but health is not OK; no firmware write is allowed. ' + text[-1800:])
        return row[1] == 'Online', text

    def start_proxy(self):
        if proxy_port_open():
            log('PROXY_REUSED', address='127.0.0.1:18898')
            return
        self.proxy = subprocess.Popen([str(self.root / 'host/bin/rknn3_transfer_proxy')], start_new_session=True)
        log('PROXY_STARTED', pid=self.proxy.pid)

    def wait_online(self, seconds):
        deadline = time.monotonic() + seconds
        last = ''
        while time.monotonic() < deadline:
            ok, last = self.status()
            if ok and proxy_port_open():
                log('CARD_ONLINE', bdf=self.bdf)
                return True
            if self.proxy is not None and self.proxy.poll() is not None:
                # A compatible external proxy may have won a startup race.
                if proxy_port_open():
                    self.proxy = None
                else:
                    raise Failure('PROXY_START_FAILED', 'Transfer proxy exited before the card became ready')
            time.sleep(1)
        log('CARD_OFFLINE', detail=last[-1800:])
        return False

    def stop_proxy(self):
        if self.proxy is not None:
            self.stop_process(self.proxy)
            self.proxy = None

    @staticmethod
    def stop_process(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def recover(self):
        if os.environ.get('RK182X_ALLOW_FIRMWARE_RECOVERY', '0') != '1':
            raise Failure('CARD_OFFLINE', 'Firmware recovery was not authorized. Stop all card users, verify the M.2 board and enable RK182X_ALLOW_FIRMWARE_RECOVERY=1 only for maintenance.')
        if os.environ.get('RK182X_BOARD') != 'ext-m2':
            raise Failure('BOARD_CONFIRMATION_REQUIRED', 'Recovery requires explicit RK182X_BOARD=ext-m2; PCI IDs alone do not establish the module board type.')
        if not self.shared_lock:
            raise Failure('RECOVERY_LOCK_REQUIRED', 'Recovery requires -v /run/lock:/run/lock to prevent concurrent managed users.')
        # Never kill an existing host service or another model container.
        if self.proxy is None:
            raise Failure('EXTERNAL_PROXY_ACTIVE', 'Stop the external host proxy and all legacy model containers before firmware recovery.')
        self.stop_proxy()
        if proxy_port_open():
            raise Failure('EXTERNAL_PROXY_ACTIVE', 'Another proxy is still active; recovery aborted')
        verify_assets(self.root)
        firmware = self.root / 'host/firmware/rknn3_rk1820_ext-m2.img'
        log('FIRMWARE_RECOVERY_BEGIN', board='ext-m2', sdk=SDK_VERSION, sha256=sha256(firmware), attempts=1)
        # One attempt only. Do not promote an SDK/model/self-test error to flashing.
        self.recovering = True
        try:
            code, out = run([self.root / 'host/bin/pcie_upgrade_tool', '-s', self.bdf, 'uf', firmware, '-reset'], timeout=180)
        finally:
            self.recovering = False
        if code:
            raise Failure('FIRMWARE_RECOVERY_FAILED', out[-3000:])
        log('FIRMWARE_RECOVERY_COMPLETE', sdk=SDK_VERSION)
        if self.stop_requested:
            raise KeyboardInterrupt()
        self.start_proxy()
        if not self.wait_online(30):
            raise Failure('CARD_OFFLINE_AFTER_RECOVERY', 'One recovery attempt completed but the card remains Offline; inspect PCIe/power/firmware logs.')

    def selftest(self):
        for attempt in range(3):
            code, out = run([sys.executable, self.root / 'selftest.py'], timeout=30)
            if not code:
                log('SELF_TEST_PASSED', detail=out[-1000:])
                return
            if attempt < 2:
                time.sleep(1)
        raise Failure('SELF_TEST_FAILED', 'Card was Online but W8A8 inference failed; no firmware recovery was attempted. ' + out[-2500:])

    def bootstrap(self):
        if platform.machine() not in ('aarch64', 'arm64'):
            raise Failure('UNSUPPORTED_ARCHITECTURE', 'Run on Linux ARM64 RK3588')
        if os.geteuid() != 0:
            raise Failure('ROOT_REQUIRED', 'Run with the documented --privileged container configuration')
        verify_assets(self.root)
        self.bdf = find_card()
        self.acquire()
        log('BOOTSTRAP_BEGIN', sdk=SDK_VERSION, kernel=self.kernel, bdf=self.bdf)
        self.load_driver()
        self.start_proxy()
        if not self.wait_online(15):
            self.recover()
        self.selftest()
        log('MODEL_START_READY', sdk=SDK_VERSION)

    def close(self):
        if self.child is not None:
            self.stop_process(self.child)
        self.stop_proxy()
        if self.lock is not None:
            self.lock.close()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # Help is safe without a board; it does not initialize or repair hardware.
    if args and args[0] != 'doctor' and ('--help' in args or '-h' in args):
        return subprocess.call(args)
    runtime = Runtime()
    def stop(signum, frame):
        if runtime.recovering:
            runtime.stop_requested = True
            log('FINISHING_FIRMWARE_WRITE_BEFORE_STOP')
            return
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if not args or args == ['doctor']:
            verify_assets()
            runtime.bdf = find_card()
            online, detail = runtime.status()
            log('DOCTOR_READ_ONLY', sdk=SDK_VERSION, kernel=runtime.kernel, bdf=runtime.bdf,
                driver_present=(MISC_ROOT / ('pcie-rkep-' + runtime.bdf) / 'dev').is_file(),
                proxy_present=proxy_port_open(), online=online, detail=detail[-1800:])
            return 0 if online and proxy_port_open() else 1
        runtime.bootstrap()
        runtime.child = subprocess.Popen(args, start_new_session=True)
        return runtime.child.wait()
    except Failure as e:
        log('FAILED', code=e.code, message=str(e))
        return 70
    except KeyboardInterrupt:
        log('STOPPING')
        return 130
    except Exception as e:
        log('FAILED', code='INITIALIZATION_ERROR', message=str(e))
        return 70
    finally:
        runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
