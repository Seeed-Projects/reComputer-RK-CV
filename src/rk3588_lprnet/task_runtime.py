from pathlib import Path

import cv2
import numpy as np

from py_utils.lprnet_official import CHARS, decode
from rknn_runtime import RKNNModel


class Runtime:
    name = "lprnet"
    input_kind = "image"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.model = RKNNModel(self.model_dir / "lprnet.rknn", platform)
        self.model_names = ["lprnet.rknn"]
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_russian_plate_number.xml"
        self.plate_cascade = cv2.CascadeClassifier(str(cascade_path))

    @staticmethod
    def _looks_like_plate(image):
        height, width = image.shape[:2]
        ratio = width / max(height, 1)
        return 2.0 <= ratio <= 6.8 and (height <= 160 or width <= 640)

    @staticmethod
    def _expanded_box(box, image_width, image_height):
        x1, y1, x2, y2 = box
        pad_x = max(2, int((x2 - x1) * 0.08))
        pad_y = max(2, int((y2 - y1) * 0.18))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image_width, x2 + pad_x),
            min(image_height, y2 + pad_y),
        )

    @staticmethod
    def _iou(first, second):
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
        second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / float(first_area + second_area - intersection)

    def _detect_candidates(self, image, max_plates):
        image_height, image_width = image.shape[:2]
        scale = min(1.0, 960.0 / max(image_width, 1))
        work = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image
        )
        gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        work_height, work_width = gray.shape
        candidates = []

        if not self.plate_cascade.empty():
            minimum_width = max(32, int(work_width * 0.025))
            minimum_height = max(10, int(minimum_width / 6.5))
            for x, y, width, height in self.plate_cascade.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.08,
                minNeighbors=2,
                minSize=(minimum_width, minimum_height),
            ):
                candidates.append((0.95, (x, y, x + width, y + height)))

        gradient = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        minimum = float(gradient.min())
        maximum = float(gradient.max())
        if maximum > minimum:
            gradient = ((gradient - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)
            gradient = cv2.morphologyEx(
                gradient,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
            )
            binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)),
                iterations=2,
            )
            binary = cv2.dilate(cv2.erode(binary, None, iterations=1), None, iterations=1)
            contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                ratio = width / max(height, 1)
                area_ratio = (width * height) / float(work_width * work_height)
                if not (2.0 <= ratio <= 7.5 and 0.00012 <= area_ratio <= 0.08):
                    continue
                if width < max(28, int(work_width * 0.018)) or height < 8:
                    continue
                edge_density = float(np.count_nonzero(gradient[y : y + height, x : x + width] > 80)) / max(
                    1, width * height
                )
                if not 0.06 <= edge_density <= 0.78:
                    continue
                geometry = 1.0 - min(abs(ratio - 4.0) / 5.0, 1.0)
                score = 0.45 + edge_density * 0.35 + geometry * 0.20
                candidates.append((score, (x, y, x + width, y + height)))

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, (15, 55, 45), (140, 255, 255))
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
            iterations=2,
        )
        for contour in cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, width, height = cv2.boundingRect(contour)
            ratio = width / max(height, 1)
            area_ratio = (width * height) / float(work_width * work_height)
            fill_ratio = cv2.contourArea(contour) / max(1.0, float(width * height))
            if 2.0 <= ratio <= 7.5 and 0.00012 <= area_ratio <= 0.08 and fill_ratio >= 0.30:
                candidates.append((0.65 + min(fill_ratio, 1.0) * 0.2, (x, y, x + width, y + height)))

        selected = []
        for score, box in sorted(candidates, key=lambda item: item[0], reverse=True):
            if any(self._iou(box, existing[1]) > 0.45 for existing in selected):
                continue
            x1, y1, x2, y2 = box
            scaled = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            selected.append((score, self._expanded_box(scaled, image_width, image_height)))
            if len(selected) >= max_plates:
                break
        return selected

    def _recognize(self, crop):
        resized = cv2.resize(crop, (94, 24), interpolation=cv2.INTER_LINEAR)
        output = np.asarray(self.model.run(resized[None])[0])
        labels, _ = decode(output, CHARS)
        logits = output[0].astype(np.float32)
        logits -= logits.max(axis=0, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-8)
        return labels[0], float(np.mean(probabilities.max(axis=0)))

    @staticmethod
    def _draw_result(preview, plate, index):
        x1, y1, x2, y2 = plate["box"]
        color = (0, 220, 80)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        ascii_text = "".join(character for character in plate["text"] if ord(character) < 128)
        label = f"#{index} {ascii_text}".strip()
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(label_height + baseline + 4, y1)
        cv2.rectangle(
            preview,
            (x1, label_y - label_height - baseline - 6),
            (min(preview.shape[1] - 1, x1 + label_width + 8), label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            preview,
            label,
            (x1 + 4, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )

    def predict(self, image, params):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        preview = image.copy()
        image_height, image_width = image.shape[:2]
        max_plates = max(1, min(32, int(params.get("max_plates", 8))))
        min_score = max(0.0, min(1.0, float(params.get("min_score", 0.0))))
        min_text_length = max(1, min(8, int(params.get("min_text_length", 6))))
        detect = bool(params.get("detect", True))
        force_whole_image = bool(params.get("whole_image", False))

        if force_whole_image or not detect or self._looks_like_plate(image):
            candidates = [(1.0, (0, 0, image_width, image_height))]
            source_mode = "plate_crop"
        else:
            candidates = self._detect_candidates(image, max_plates)
            source_mode = "scene"

        plates = []
        for candidate_score, box in candidates:
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text, recognition_score = self._recognize(crop)
            if len(text) < min_text_length or recognition_score < min_score:
                continue
            plate = {
                "text": text,
                "recognition_score": round(recognition_score, 4),
                "candidate_score": round(float(candidate_score), 4),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            }
            plates.append(plate)
            self._draw_result(preview, plate, len(plates))

        summary = f"Plates: {len(plates)}"
        cv2.rectangle(preview, (0, 0), (max(150, 16 + len(summary) * 11), 36), (15, 15, 15), -1)
        cv2.putText(preview, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2)
        return {
            "plates": plates,
            "count": len(plates),
            "source_mode": source_mode,
            "image": {"width": image_width, "height": image_height},
        }, preview

    def warmup_preview(self, setter):
        image = cv2.imread(str(self.model_dir / "test.jpg"))
        if image is not None:
            _, preview = self.predict(image, {"whole_image": True})
            setter(preview)


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
