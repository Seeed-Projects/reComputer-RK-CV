"""Shared adapter contract; HTTP handling does not depend on detection outputs."""
import threading
from abc import ABC, abstractmethod


class SourceUnavailable(RuntimeError):
    pass


class UnsupportedOperation(RuntimeError):
    pass


class BaseBackend(ABC):
    def __init__(self):
        self.config = {"obj_thresh": 0.25, "nms_thresh": 0.45}
        self.config_lock = threading.Lock()
        self.description = {}
        self.prediction_metadata = {}
        self.demo_video = None

    def thresholds(self, conf, iou):
        with self.config_lock:
            return (self.config["obj_thresh"] if conf is None else conf,
                    self.config["nms_thresh"] if iou is None else iou)

    def configure(self, values):
        with self.config_lock:
            self.config.update(values)

    @abstractmethod
    def image(self, image, conf, iou, source):
        """Return a JSON-compatible task result with a timing dictionary.

        image is decoded BGR. conf and iou are optional detection controls;
        adapters for other tasks may ignore them. Task-specific outputs belong
        in the returned dictionary, not in the shared HTTP layer.
        """

    def source_status(self):
        return None

    def predict_realtime(self, conf, iou):
        raise UnsupportedOperation("this backend does not support live inference")

    def preview(self):
        raise UnsupportedOperation("this backend does not support video preview")

    def demo_result(self, after):
        raise UnsupportedOperation("this backend does not support timed detection results")

    def demo_candidates(self, after):
        raise UnsupportedOperation("this backend does not support timed detection candidates")

    def annotate(self, jpeg, raw):
        return jpeg

    def close(self):
        pass
