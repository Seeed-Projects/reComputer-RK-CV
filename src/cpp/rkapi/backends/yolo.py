"""Rockchip RGA/DMA/RKNN inference and task results."""
import json
import os
from pathlib import Path
from queue import Queue
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

from .. import _native
from .base import BaseBackend, SourceUnavailable

COCO = ("person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
        "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
        "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
        "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
        "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
        "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
        "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
        "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,"
        "teddy bear,hair drier,toothbrush").split(",")


def prediction_output(task, package):
    field = {"classify": "classification", "semantic": "semantic", "depth": "depth"}.get(
        task, "predictions")
    output = {"field": field}
    if task != "classify":
        output["coordinate_space"] = "source_image_pixels"

    if task == "pose":
        output.update(item_field="keypoints", point_fields=["x", "y", "confidence"],
                      keypoint_count=package.get("kpt_shape", [17])[0])
    elif task == "obb":
        output.update(item_field="polygon", point_fields=["x", "y"], point_count=4)
    elif task == "segment":
        output["item_field"] = "mask"

    if task in ("segment", "semantic", "depth"):
        output["raster"] = {
            "encoding": "rle", "order": "row_major", "run_order": ["value", "count"],
            "box_order": ["x1", "y1", "x2", "y2"],
            "values": {"segment": "binary", "semantic": "class_id", "depth": "quantized_depth"}[task],
        }
        if task == "depth":
            output["raster"]["invalid_value"] = 65536
    return output


class Backend(BaseBackend):
    def __init__(self, package):
        super().__init__()
        model = package["file"]
        nms_mode = os.getenv("RK_NMS", "on" if package.get("nms", True) else "off").lower()
        if nms_mode not in ("on", "off"):
            raise ValueError("RK_NMS must be on or off")
        options = {key: package[key] for key in ("classes", "end2end", "coordinates",
                                                 "preprocess", "dense_resize", "kpt_shape",
                                                 "head", "anchors")
                   if key in package}
        options.update(task="detect" if package["task"] == "detection" else package["task"],
                       nms=nms_mode == "on")
        if package.get("text_embeddings"):
            embeddings = np.load(package["text_embeddings"], allow_pickle=False)
            if (embeddings.shape != (1, package["classes"], 512) or
                    embeddings.dtype != np.float32 or not np.isfinite(embeddings).all()):
                raise ValueError("Invalid YOLO-World text embeddings")
            options["text_embeddings"] = embeddings.ravel().tolist()
        if "runtime_preprocess" in package:
            options["preprocess"] = package["runtime_preprocess"]
        encoded_options = json.dumps(options)
        self.predictor = _native.Predictor(model, encoded_options)
        self.description = json.loads(self.predictor.description())
        self.description["chip"] = package["chip"]
        self.description["candidate_preview"] = self.predictor.supports_candidates()
        self.description["input_transform"] = options.get(
            "preprocess", "center_crop" if package["task"] == "classify" else "letterbox")
        self.labels = COCO if self.description["classes"] == 80 else []
        if path := os.getenv("RK_CLASSES", package.get("labels")):
            with open(path, encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]
            if len(self.labels) != self.description["classes"]:
                raise ValueError("class file must have one label per model class")
        task = package["task"]
        classification_names = Path("/opt/demo/photos/labels.json")
        self.classification_names = {}
        if task == "classify" and classification_names.is_file():
            self.classification_names = json.loads(classification_names.read_text(encoding="utf-8"))
        self.prediction_metadata = {
            "model_input": {"width": self.description["input_width"],
                            "height": self.description["input_height"]},
            "class_count": self.description["classes"] if task != "depth" else 0,
            "labels": self.labels if task != "depth" else [],
            "output": prediction_output(task, package),
        }
        self.live = None
        self.live_lock = Lock()
        self.viewers = 0
        self.model = model
        self.model_options = encoded_options
        self.workers = int(os.getenv("RK_WORKERS", "0"))
        if self.workers not in (0, 1):
            raise ValueError("RK_WORKERS must be 0 (auto) or 1")
        self.preview_workers = self.workers or {"rk3576": 2, "rk3588": 3}.get(package["chip"], 1)
        self.preview_masks = ([1 << index for index in range(self.preview_workers)]
                              if self.preview_workers > 1 else ["all"])
        self.description.update(preview_workers=self.preview_workers, preview_core_masks=self.preview_masks)
        self.preview_pool = None
        self.preview_lock = Lock()
        bundled = Path("/opt/demo/input.mp4")
        override = os.getenv("RK_SOURCE")
        self.source = str(bundled) if override is None and bundled.is_file() else override
        loop = os.getenv("RK_LOOP_VIDEO", "1" if override is None and self.source else "0")
        if loop == "1":
            path = Path(self.source or "")
            if path.suffix.lower() != ".mp4" or not path.is_file():
                raise ValueError("RK_LOOP_VIDEO requires a local MP4 as RK_SOURCE")
            self.demo_video = path.resolve()
        if self.source:
            self.description["live"] = {
                "candidate_stream": self.demo_video is not None and self.predictor.supports_candidates()
            }

    def acquire_live(self):
        with self.live_lock:
            if not self.source:
                raise SourceUnavailable("configure RK_SOURCE for realtime predictions")
            if self.live is None:
                try:
                    live = _native.Live(self.source, self.model, self.workers,
                                        self.model_options, loop_file=self.demo_video is not None)
                except RuntimeError as error:
                    raise SourceUnavailable(str(error)) from error
                try:
                    with self.config_lock:
                        live.configure(self.config["obj_thresh"], self.config["nms_thresh"])
                    self.description["live"] = json.loads(live.description())
                except Exception:
                    live.close()
                    raise
                self.live = live
            self.viewers += 1

    def release_live(self):
        with self.live_lock:
            self.viewers -= 1
            if self.viewers == 0:
                self.live.close()
                self.live = None

    def response(self, raw, source):
        predictions = []
        for detection in raw["detections"]:
            idx = detection["class_id"]
            label = self.labels[idx] if 0 <= idx < len(self.labels) else str(idx)
            prediction = {"class": label, "class_id": idx, "confidence": detection["confidence"],
                          "box": dict(zip(("x1", "y1", "x2", "y2"), detection["box"]))}
            for key in ("keypoints", "polygon", "mask"):
                if key in detection:
                    prediction[key] = detection[key]
            predictions.append(prediction)
        result = {"success": True, "source": source, "predictions": predictions,
                "image": {"width": raw["source_width"], "height": raw["source_height"]},
                "timing": {key: raw[key] for key in (
                    "context_wait_ms", "dma_upload_ms", "preprocess_ms", "inference_ms",
                    "rknn_run_ms", "output_get_ms", "postprocess_ms", "nms_ms") if key in raw}}
        if "classification" in raw:
            result["classification"] = []
            for item in raw["classification"]:
                label = self.labels[item["class_id"]] if self.labels else str(item["class_id"])
                result["classification"].append({**item, "class": label,
                                                 "class_name": self.classification_names.get(label, label)})
        if "dense" in raw:
            result[raw["task"]] = raw["dense"]
            if raw["task"] == "semantic":
                result["semantic"]["labels"] = self.labels
        return result

    def image(self, image, conf, iou, source):
        result, _ = self._image(self.predictor, image, conf, iou, source, False)
        return result

    def preview_image(self, image, conf, iou, source):
        candidate_preview = self.description["candidate_preview"]
        if self.preview_workers == 1:
            result, candidates = self._image(self.predictor, image, conf, iou, source, candidate_preview)
            result["npu_core_mask"] = "all"
            return result, candidates

        queued = perf_counter()
        with self.preview_lock:
            if self.preview_pool is None:
                pool = Queue(maxsize=self.preview_workers)
                for mask in self.preview_masks:
                    pool.put_nowait((mask, _native.Predictor(self.model, self.model_options, mask)))
                self.preview_pool = pool
            pool = self.preview_pool
        worker = pool.get()
        wait_ms = (perf_counter() - queued) * 1000
        mask, predictor = worker
        try:
            result, candidates = self._image(predictor, image, conf, iou, source, candidate_preview)
        finally:
            pool.put_nowait(worker)
        result["npu_core_mask"] = mask
        result["timing"]["preview_queue_ms"] = wait_ms
        return result, candidates

    def _image(self, predictor, image, conf, iou, source, candidate_preview):
        started = perf_counter()
        if image is None or image.size == 0:
            raise ValueError("cannot decode image or video frame")
        height, width = image.shape[:2]
        if width > 8192 or height > 8192 or width * height > 8192 * 8192:
            raise ValueError("image dimensions exceed 8192 x 8192")
        # NV12 needs even dimensions; preserve original API coordinates below.
        w, h = (width + 1) & ~1, (height + 1) & ~1
        if (w, h) != (width, height):
            image = cv2.copyMakeBorder(image, 0, h - height, 0, w - width, cv2.BORDER_REPLICATE)
        planar = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        y_size = w * h
        uv_size = y_size // 4
        nv12 = np.empty(y_size + uv_size * 2, dtype=np.uint8)
        nv12[:y_size] = planar[:y_size]
        nv12[y_size::2] = planar[y_size:y_size + uv_size]
        nv12[y_size + 1::2] = planar[y_size + uv_size:]
        pixels = nv12.tobytes()
        prepare_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        if candidate_preview:
            encoded, candidates = predictor.preview(pixels, w, h, conf, iou)
        else:
            encoded = predictor.predict(pixels, w, h, conf, iou)
            candidates = b""
        native_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        raw = json.loads(encoded)
        raw["source_width"], raw["source_height"] = width, height
        for item in raw["detections"]:
            item["box"] = [max(0, min(value, width if i % 2 == 0 else height))
                           for i, value in enumerate(item["box"])]
        result = self.response(raw, source)
        result["timing"].update({"color_convert_ms": prepare_ms, "native_call_ms": native_ms,
                                 "response_build_ms": (perf_counter() - started) * 1000})
        return result, candidates

    def annotate(self, jpeg, raw):
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        sx, sy = image.shape[1] / raw["source_width"], image.shape[0] / raw["source_height"]
        for detection in self.response(raw, "realtime")["predictions"]:
            box = detection["box"]
            x1, y1, x2, y2 = (int(box["x1"] * sx), int(box["y1"] * sy),
                              int(box["x2"] * sx), int(box["y2"] * sy))
            cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 90), 2)
            cv2.putText(image, f'{detection["class"]} {detection["confidence"]:.2f}',
                        (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 220, 90), 1)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return encoded.tobytes() if ok else jpeg

    def preview(self):
        if not self.live:
            raise SourceUnavailable("configure RK_SOURCE for realtime predictions")
        state, jpeg = self.live.snapshot()
        state = json.loads(state)
        if state["error"]:
            raise SourceUnavailable(state["error"])
        return state, jpeg

    def source_status(self):
        with self.live_lock:
            if not self.source:
                return None
            if self.live is None:
                return {"active": False, "viewers": 0, "error": "", "ended": False, "dropped": 0}
            return {"active": True, "viewers": self.viewers} | json.loads(self.live.state())

    def live_result(self, after=0):
        if not self.live:
            raise SourceUnavailable("configure RK_SOURCE for realtime predictions")
        state = json.loads(self.live.state(True, after))
        if state["error"]:
            raise SourceUnavailable(state["error"])
        return state["result"]

    def demo_result(self, after):
        raw = self.live_result(after)
        if raw is None or "media_time" not in raw:
            return None
        result = self.response(raw, "realtime")
        return {key: raw[key] for key in ("frame", "media_time", "cycle")} | {
            key: result[key] for key in ("predictions", "image", "classification", "semantic", "depth")
            if key in result}

    def demo_candidates(self, after):
        try:
            return self.live.candidates(after)
        except RuntimeError as error:
            raise SourceUnavailable(str(error)) from error

    def predict_realtime(self, conf, iou):
        with self.live_lock:
            if self.live is None:
                raise SourceUnavailable("open a preview stream to start live inference")
            if self.live_result() is None:
                raise SourceUnavailable("live source has no frame yet")
            raw = self.live.predict(self.predictor, conf, iou)
        return self.response(json.loads(raw), "realtime")

    def configure(self, values):
        with self.live_lock, self.config_lock:
            self.config.update(values)
            if self.live:
                self.live.configure(self.config["obj_thresh"], self.config["nms_thresh"])

    def close(self):
        with self.live_lock:
            if self.live:
                self.live.close()
                self.live = None
