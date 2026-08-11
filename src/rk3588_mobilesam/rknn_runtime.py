import contextlib
import os
import sys
import threading
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


@contextlib.contextmanager
def _quiet_rknn_initialization():
    """Hide known static-model startup warnings without muting inference errors."""
    if os.environ.get("RKNN_LOG_LEVEL") != "0":
        yield
        return

    saved_fds = []
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
            try:
                stream_fd = stream.fileno()
            except (AttributeError, OSError):
                continue
            saved_fds.append((stream, stream_fd, os.dup(stream_fd)))
            os.dup2(null_fd, stream_fd)
        yield
    finally:
        for stream, stream_fd, saved_fd in saved_fds:
            stream.flush()
            os.dup2(saved_fd, stream_fd)
            os.close(saved_fd)
        os.close(null_fd)


class RKNNModel:
    """Small thread-safe RKNNLite wrapper shared by the packaged demos."""

    def __init__(self, model_path, platform, core_index=None):
        self.model_path = str(Path(model_path))
        self.platform = platform
        self.lock = threading.Lock()
        if core_index == 0:
            core_mask = RKNNLite.NPU_CORE_0
        elif core_index == 1:
            core_mask = RKNNLite.NPU_CORE_1
        elif core_index == 2:
            core_mask = (
                RKNNLite.NPU_CORE_0
                if platform == "rk3576"
                else RKNNLite.NPU_CORE_2
            )
        else:
            core_mask = (
                RKNNLite.NPU_CORE_0_1
                if platform == "rk3576"
                else RKNNLite.NPU_CORE_0_1_2
            )
        with _quiet_rknn_initialization():
            self.model = RKNNLite()
            ret = self.model.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"load_rknn failed ({ret}): {self.model_path}")
            ret = self.model.init_runtime(core_mask=core_mask)
            if ret != 0:
                self.model.release()
                raise RuntimeError(f"init_runtime failed ({ret}): {self.model_path}")

    def run(self, inputs, data_format=None):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        normalized = []
        for value in inputs:
            value = np.asarray(value)
            normalized.append(value)
        kwargs = {"inputs": normalized}
        if data_format is not None:
            kwargs["data_format"] = data_format
        with self.lock:
            outputs = self.model.inference(**kwargs)
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        return outputs

    def release(self):
        if self.model is not None:
            self.model.release()
            self.model = None
