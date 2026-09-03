import hashlib
import threading
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def materialize_model(model_path):
    """Reassemble a chunked RKNN model on first use."""
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path

    parts = sorted(model_path.parent.glob(f"{model_path.name}.part-*"))
    if not parts:
        raise FileNotFoundError(f"RKNN model not found: {model_path}")

    temporary_path = model_path.with_name(f".{model_path.name}.assembling")
    digest = hashlib.sha256()
    try:
        with temporary_path.open("wb") as output:
            for part_path in parts:
                with part_path.open("rb") as part:
                    while chunk := part.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)

        checksum_path = model_path.with_name(f"{model_path.name}.sha256")
        if checksum_path.is_file():
            expected = checksum_path.read_text(encoding="utf-8").split()[0].lower()
            if digest.hexdigest() != expected:
                raise RuntimeError(f"checksum mismatch while assembling {model_path.name}")
        temporary_path.replace(model_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return model_path


class RKNNModel:
    """Small thread-safe RKNNLite wrapper shared by the packaged demos."""

    def __init__(self, model_path, platform, core_index=None):
        self.model_path = str(materialize_model(model_path))
        self.platform = platform
        self.lock = threading.Lock()
        self.model = RKNNLite()
        ret = self.model.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed ({ret}): {self.model_path}")

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
