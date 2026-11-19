"""RK1820 YOLO26 six-head decode, adapted from airockchip/rknn3-model-zoo.

YOLO26 box heads contain four distances, not YOLOv8's DFL distribution.
Score heads contain logits. RKNN3 Lite returns dequantized float outputs.
"""
from pathlib import Path
import threading
import time

import cv2
import numpy as np


def letterbox(frame, size=640):
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    rw, rh = round(w * ratio), round(h * ratio)
    x, y = (size - rw) // 2, (size - rh) // 2
    resized = cv2.resize(frame, (rw, rh))
    padded = cv2.copyMakeBorder(resized, y, size - rh - y, x, size - rw - x,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), (ratio, x, y)


def decode(outputs, conf=0.25, iou=0.45, size=640):
    if outputs is None or len(outputs) != 6:
        raise RuntimeError('Expected six YOLO26 detection heads')
    boxes, scores, classes = [], [], []
    for branch in range(3):
        box, score = [np.asarray(v, dtype=np.float32) for v in outputs[branch * 2:branch * 2 + 2]]
        if box.shape[1] != 4 or score.shape[1] != 80:
            raise RuntimeError(f'Unexpected YOLO26 head shapes: {box.shape}, {score.shape}')
        h, w = box.shape[2:]
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        center = np.stack((grid_x + 0.5, grid_y + 0.5), axis=0)[None]
        xyxy = np.concatenate((center - box[:, :2], center + box[:, 2:]), axis=1)
        xyxy *= np.array([size / w, size / h, size / w, size / h])[None, :, None, None]
        logits = score.transpose(0, 2, 3, 1).reshape(-1, 80)
        cls = np.argmax(logits, axis=1)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits.max(axis=1), -80, 80)))
        mask = prob >= conf
        boxes.append(xyxy.transpose(0, 2, 3, 1).reshape(-1, 4)[mask])
        scores.append(prob[mask])
        classes.append(cls[mask])
    boxes, scores, classes = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    keep = []
    for cls in np.unique(classes):
        order = np.flatnonzero(classes == cls)
        order = order[np.argsort(scores[order])[::-1]]
        while len(order):
            current = order[0]
            keep.append(current)
            rest = order[1:]
            left = np.maximum(boxes[current, :2], boxes[rest, :2])
            right = np.minimum(boxes[current, 2:], boxes[rest, 2:])
            inter = np.prod(np.maximum(right - left, 0), axis=1)
            area = np.prod(np.maximum(boxes[current, 2:] - boxes[current, :2], 0))
            other = np.prod(np.maximum(boxes[rest, 2:] - boxes[rest, :2], 0), axis=1)
            overlap = inter / np.maximum(area + other - inter, 1e-9)
            order = rest[overlap <= iou]
    keep = sorted(keep, key=lambda n: float(scores[n]), reverse=True)[:300]
    return boxes[keep], classes[keep], scores[keep]


class YOLO26:
    """One serialized RKNN3 context shared by preview and HTTP worker threads."""
    def __init__(self, model_path, weight_path, labels_path, core_mask=1):
        from rknn3lite.api import RKNN3Lite
        for path in (model_path, weight_path, labels_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        self.labels = Path(labels_path).read_text().splitlines()
        if len(self.labels) != 80:
            raise ValueError('This model requires 80 COCO labels')
        self.lock = threading.Lock()
        self.calls = 0
        self.last_inference_ms = 0.0
        self.last_total_ms = 0.0
        self.runtime = RKNN3Lite()
        try:
            if self.runtime.load_rknn(str(model_path), str(weight_path)) != 0:
                raise RuntimeError('RKNN3 load_rknn failed')
            if self.runtime.init_runtime(target='rk1820', core_mask=core_mask) != 0:
                raise RuntimeError('RKNN3 init failed: check host PCIe driver, firmware 1.1.0 and proxy service')
        except Exception:
            self.runtime.release()
            raise

    def detect(self, frame, conf=0.25, iou=0.45):
        with self.lock:
            started = time.perf_counter()
            image, (ratio, x, y) = letterbox(frame)
            infer_start = time.perf_counter()
            outputs = self.runtime.inference(inputs=[image[None]], data_format=['nhwc'])
            infer_ms = (time.perf_counter() - infer_start) * 1000
            boxes, classes, scores = decode(outputs, conf, iou)
            boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - x) / ratio, 0, frame.shape[1])
            boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - y) / ratio, 0, frame.shape[0])
            predictions = [dict(
                **{'class': self.labels[int(cls)].strip()}, class_id=int(cls), confidence=float(score),
                box=dict(zip(('x1', 'y1', 'x2', 'y2'), (int(v) for v in box))))
                for box, cls, score in zip(boxes, classes, scores)]
            total_ms = (time.perf_counter() - started) * 1000
            self.calls += 1
            self.last_inference_ms, self.last_total_ms = infer_ms, total_ms
            return predictions, {'runtime_ms': round(infer_ms, 2), 'total_ms': round(total_ms, 2)}

    def release(self):
        with self.lock:
            self.runtime.release()


def draw(frame, predictions):
    scale = max(1.0, frame.shape[1] / 1280)
    thickness = max(2, round(2 * scale))
    for p in predictions:
        b = p['box']
        cv2.rectangle(frame, (b['x1'], b['y1']), (b['x2'], b['y2']), (0, 230, 118), thickness)
        cv2.putText(frame, f"{p['class']} {p['confidence']:.2f}", (b['x1'], max(round(15 * scale), b['y1'] - round(6 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 230, 118), thickness)
    return frame
