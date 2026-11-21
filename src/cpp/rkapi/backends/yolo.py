"""Rockchip RGA/DMA/RKNN inference and task results."""
import json
import os
from pathlib import Path
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
                                                 "preprocess", "dense_resize", "kpt_shape")
                   if key in package}
        options.update(task="detect" if package["task"] == "detection" else package["task"],
                       nms=nms_mode == "on")
        encoded_options = json.dumps(options)
        self.predictor = _native.Predictor(model, encoded_options)
        self.description = json.loads(self.predictor.description())
        self.description["chip"] = package["chip"]
        self.labels = COCO if self.description["classes"] == 80 else []
        if path := os.getenv("RK_CLASSES", package.get("labels")):
            with open(path, encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]
            if len(self.labels) != self.description["classes"]:
                raise ValueError("class file must have one label per model class")
        task = package["task"]
        self.prediction_metadata = {
            "model_input": {"width": self.description["input_width"],
                            "height": self.description["input_height"]},
            "class_count": self.description["classes"] if task != "depth" else 0,
            "labels": self.labels if task != "depth" else [],
            "output": prediction_output(task, package),
        }
        self.live = None
        source = os.getenv("RK_SOURCE")
        if os.getenv("RK_LOOP_VIDEO") == "1":
            path = Path(source or "")
            if task != "detection" or path.suffix.lower() != ".mp4" or not path.is_file():
                raise ValueError("RK_LOOP_VIDEO requires a detection model and a local MP4 as RK_SOURCE")
            self.demo_video = path.resolve()
        if source:
            self.live = _native.Live(source, model, int(os.getenv("RK_WORKERS", "0")),
                                     encoded_options, loop_file=self.demo_video is not None)
            self.description["live"] = json.loads(self.live.description())

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
            result["classification"] = [
                {**item, "class": self.labels[item["class_id"]] if self.labels else str(item["class_id"])}
                for item in raw["classification"]]
        if "dense" in raw:
            result[raw["task"]] = raw["dense"]
            if raw["task"] == "semantic":
                result["semantic"]["labels"] = self.labels
        return result

    def image(self, image, conf, iou, source):
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
        encoded = self.predictor.predict(pixels, w, h, conf, iou)
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
        return result

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
        if not self.live:
            return None
        return json.loads(self.live.state())

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
            "predictions": result["predictions"], "image": result["image"]}

    def demo_candidates(self, after):
        try:
            return self.live.candidates(after)
        except RuntimeError as error:
            raise SourceUnavailable(str(error)) from error

    def predict_realtime(self, conf, iou):
        if self.live_result() is None:
            raise SourceUnavailable("live source has no frame yet")
        raw = self.live.predict(self.predictor, conf, iou)
        return self.response(json.loads(raw), "realtime")

    def configure(self, values):
        with self.config_lock:
            self.config.update(values)
            if self.live:
                self.live.configure(self.config["obj_thresh"], self.config["nms_thresh"])

    def close(self):
        if self.live:
            self.live.close()
