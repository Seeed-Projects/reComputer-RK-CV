"""Board-side V4L2 camera discovery and preview capture."""

import fcntl
import os
from pathlib import Path
import re
import struct
from threading import Lock

import cv2
import numpy as np

from . import _hdmi
from .backends.base import SourceUnavailable

_QUERYCAP = 0x80685600
_CAPTURE = 0x00000001 | 0x00001000
_DEVICE_CAPS = 0x80000000
_lock = Lock()
_active = set()


def cameras():
    devices = []
    for path in sorted(Path("/dev").glob("video[0-9]*")):
        if not re.fullmatch(r"video[0-9]+", path.name) or not path.is_char_device():
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                capability = bytearray(104)
                fcntl.ioctl(fd, _QUERYCAP, capability)
            finally:
                os.close(fd)
        except OSError:
            continue
        caps, device_caps = struct.unpack_from("=II", capability, 84)
        if (device_caps if caps & _DEVICE_CAPS else caps) & _CAPTURE:
            label = capability[16:48].split(b"\0", 1)[0].decode(errors="replace")
            driver = capability[:16].split(b"\0", 1)[0].decode(errors="replace")
            devices.append({"id": path.name, "name": label or path.name, "driver": driver})
    return devices


class CameraPreview:
    def __init__(self, camera, runtime):
        self.device = camera["id"]
        self.is_hdmi = camera["driver"] == "rk_hdmirx"
        self.runtime = runtime
        self.capture = None

    def acquire_live(self):
        with _lock:
            if self.device in _active:
                raise SourceUnavailable("This camera already has a preview open")
            _active.add(self.device)
        try:
            path = f"/dev/{self.device}"
            if self.is_hdmi:
                try:
                    self.capture = _hdmi.HdmiCapture(path)
                except RuntimeError as error:
                    raise SourceUnavailable(str(error)) from error
                return
            self.capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                raise SourceUnavailable("Could not open the board camera")
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.release_live()
            raise

    def release_live(self):
        try:
            if self.capture is not None:
                if not self.is_hdmi:
                    self.capture.release()
                self.capture = None
        finally:
            with _lock:
                _active.discard(self.device)

    def read(self):
        if self.is_hdmi:
            try:
                width, height, pixels = self.capture.read()
            except RuntimeError as error:
                raise SourceUnavailable(str(error)) from error
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        else:
            ok, image = self.capture.read()
            if not ok:
                raise SourceUnavailable("The board camera stopped delivering frames")
        height, width = image.shape[:2]
        scale = min(1280 / width, 720 / height, 1)
        if scale < 1:
            image = cv2.resize(image, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
        return image

    def frame(self, image):
        conf, iou = self.runtime.thresholds(None, None)
        result, candidates = self.runtime.preview_image(image, conf, iou, "camera")
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise SourceUnavailable("Could not encode the board camera frame")
        return result, jpeg.tobytes(), candidates
