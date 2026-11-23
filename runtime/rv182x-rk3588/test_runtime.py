import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location('runtime', Path(__file__).with_name('runtime.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RuntimePolicies(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rt = r.Runtime(self.root)
        self.rt.bdf = '0000:01:00.0'

    def tearDown(self):
        self.temp.cleanup()

    def failure(self, code, call):
        with self.assertRaises(r.Failure) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_pci_discovery_requires_one_supported_card(self):
        with patch.object(r, 'PCI_ROOT', self.root):
            self.failure('PCI_DEVICE_NOT_FOUND', r.find_card)
            dev = self.root / self.rt.bdf
            dev.mkdir()
            (dev / 'vendor').write_text('0x1d87')
            (dev / 'device').write_text('0x182a')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(r.find_card(), self.rt.bdf)
            with patch.dict(os.environ, {'RK182X_DEVICE': '0000:02:00.0'}, clear=True):
                self.failure('PCI_ADDRESS_MISMATCH', r.find_card)
            other = self.root / '0001:01:00.0'
            other.mkdir()
            (other / 'vendor').write_text('0x1d87')
            (other / 'device').write_text('0x182a')
            self.failure('MULTIPLE_CARDS_UNSUPPORTED', r.find_card)

    def test_corrupt_asset_is_rejected(self):
        (self.root / 'firmware.img').write_bytes(b'expected')
        digest = r.sha256(self.root / 'firmware.img')
        (self.root / 'ASSET_SHA256SUMS').write_text(digest + '  firmware.img\n')
        r.verify_assets(self.root)
        (self.root / 'firmware.img').write_bytes(b'corrupted')
        self.failure('ASSET_CHECKSUM_FAILED', lambda: r.verify_assets(self.root))

    def driver_profile(self):
        self.rt.kernel = 'test-vendor-kernel'
        driver = self.root / 'drivers/pcie-rkep.ko'
        driver.parent.mkdir()
        driver.write_bytes(b'verified-module')
        (driver.parent / 'manifest.json').write_text(json.dumps({self.rt.kernel: {
            'file': driver.name, 'sha256': r.sha256(driver),
            'vermagic': 'test-vendor-kernel SMP aarch64'}}))
        return driver

    def test_modified_driver_never_reaches_insmod(self):
        driver = self.driver_profile()
        driver.write_bytes(b'changed-module')
        with patch.object(r, 'MISC_ROOT', self.root), patch.object(r, 'run') as command:
            self.failure('DRIVER_CHECKSUM_FAILED', self.rt.load_driver)
            command.assert_not_called()

    def test_wrong_driver_abi_is_not_force_loaded(self):
        self.driver_profile()
        with patch.object(r, 'MISC_ROOT', self.root), patch.object(r, 'run', return_value=(0, 'other-kernel SMP aarch64')) as command:
            self.failure('DRIVER_ABI_MISMATCH', self.rt.load_driver)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(command.call_args[0][0][:3], ['modinfo', '-F', 'vermagic'])

    def test_offline_does_not_flash_without_explicit_authorization(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(r, 'run') as command:
            self.failure('CARD_OFFLINE', self.rt.recover)
            command.assert_not_called()

    def test_firmware_requires_explicit_m2_board(self):
        with patch.dict(os.environ, {'RK182X_ALLOW_FIRMWARE_RECOVERY': '1'}, clear=True), patch.object(r, 'run') as command:
            self.failure('BOARD_CONFIRMATION_REQUIRED', self.rt.recover)
            command.assert_not_called()

    def test_firmware_requires_shared_lock_mount(self):
        self.rt.shared_lock = False
        with patch.dict(os.environ, {'RK182X_ALLOW_FIRMWARE_RECOVERY': '1', 'RK182X_BOARD': 'ext-m2'}, clear=True), patch.object(r, 'run') as command:
            self.failure('RECOVERY_LOCK_REQUIRED', self.rt.recover)
            command.assert_not_called()

    def test_existing_external_proxy_is_never_killed_or_flashed(self):
        self.rt.shared_lock = True
        with patch.dict(os.environ, {'RK182X_ALLOW_FIRMWARE_RECOVERY': '1', 'RK182X_BOARD': 'ext-m2'}, clear=True), patch.object(r, 'run') as command, patch.object(r.os, 'killpg') as kill:
            self.failure('EXTERNAL_PROXY_ACTIVE', self.rt.recover)
            self.rt.close()
            command.assert_not_called()
            kill.assert_not_called()

    def prepare_recovery(self):
        self.rt.shared_lock = True
        self.rt.proxy = MagicMock()
        fw = self.root / 'host/firmware/rknn3_rk1820_ext-m2.img'
        fw.parent.mkdir(parents=True)
        fw.write_bytes(b'test-payload')

    def test_authorized_recovery_uses_exact_board_firmware_once(self):
        self.prepare_recovery()
        with patch.dict(os.environ, {'RK182X_ALLOW_FIRMWARE_RECOVERY': '1', 'RK182X_BOARD': 'ext-m2'}, clear=True), patch.object(r, 'verify_assets'), patch.object(r, 'proxy_port_open', return_value=False), patch.object(self.rt, 'stop_proxy'), patch.object(self.rt, 'start_proxy'), patch.object(self.rt, 'wait_online', return_value=True), patch.object(r, 'run', return_value=(0, 'OK')) as command:
            self.rt.recover()
            self.assertEqual(command.call_count, 1)
            args = [str(x) for x in command.call_args[0][0]]
            self.assertEqual(args[-3:], ['uf', str(self.root / 'host/firmware/rknn3_rk1820_ext-m2.img'), '-reset'])

    def test_failed_flash_is_not_retried(self):
        self.prepare_recovery()
        with patch.dict(os.environ, {'RK182X_ALLOW_FIRMWARE_RECOVERY': '1', 'RK182X_BOARD': 'ext-m2'}, clear=True), patch.object(r, 'verify_assets'), patch.object(r, 'proxy_port_open', return_value=False), patch.object(self.rt, 'stop_proxy'), patch.object(r, 'run', return_value=(1, 'write failed')) as command:
            self.failure('FIRMWARE_RECOVERY_FAILED', self.rt.recover)
            self.assertEqual(command.call_count, 1)

    def test_online_selftest_failure_never_triggers_firmware_recovery(self):
        with patch.object(r.platform, 'machine', return_value='aarch64'), patch.object(r.os, 'geteuid', return_value=0), patch.object(r, 'verify_assets'), patch.object(r, 'find_card', return_value=self.rt.bdf), patch.object(self.rt, 'acquire'), patch.object(self.rt, 'load_driver'), patch.object(self.rt, 'start_proxy'), patch.object(self.rt, 'wait_online', return_value=True), patch.object(self.rt, 'selftest', side_effect=r.Failure('SELF_TEST_FAILED', 'bad SDK')), patch.object(self.rt, 'recover') as recover:
            self.failure('SELF_TEST_FAILED', self.rt.bootstrap)
            recover.assert_not_called()

    def test_online_requires_health_ok_not_merely_online_word(self):
        with patch.object(r, 'run', return_value=(0, '| 0 | Offline | NA |')):
            self.assertFalse(self.rt.status()[0])
        with patch.object(r, 'run', return_value=(0, '| 0 | Online | BAD |')):
            self.failure('CARD_HEALTH_ERROR', self.rt.status)
        with patch.object(r, 'run', return_value=(0, '| 0 | Online | OK |')):
            self.assertTrue(self.rt.status()[0])

    def test_real_sdk_table_places_device_and_status_in_same_cell(self):
        table = '''+========================+===============+===============+======================+
| Device        Status   | Health        | Power(mW)     | Npu(%)               |
| Chip          Name     | Bus-Id        | Temp(C)       | Memory-Usage(MB)     |
+========================+===============+===============+======================+
| 0             Online   | OK            | NA            | 0                    |
| 0             RK1820   | 0000:01:00.0  | 41            | 32   / 2560          |
+========================+===============+===============+======================+
'''
        with patch.object(r, 'run', return_value=(0, table)):
            self.assertTrue(self.rt.status()[0])
        with patch.object(r, 'run', return_value=(1, table)):
            self.assertTrue(self.rt.status()[0])
        with patch.object(r, 'run', return_value=(0, table.replace('Online', 'Offline').replace('OK', 'NA'))):
            self.assertFalse(self.rt.status()[0])

    def test_unknown_or_failed_status_query_never_becomes_offline(self):
        with patch.object(r, 'run', return_value=(0, 'unknown table format')):
            self.failure('CARD_STATUS_UNKNOWN', self.rt.status)
        with patch.object(r, 'run', return_value=(1, 'driver ioctl error')):
            self.failure('CARD_STATUS_QUERY_FAILED', self.rt.status)


if __name__ == '__main__':
    unittest.main()
