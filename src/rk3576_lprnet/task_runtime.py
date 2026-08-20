from pathlib import Path
import re

import cv2
import numpy as np

from py_utils.lprnet_official import CHARS, decode
from rknn_runtime import RKNNModel


CHINESE_PREFIXES = set(CHARS[:31])
INTERNATIONAL_STYLES = {"light", "light_red"}


class Runtime:
    name = "lprnet"
    input_kind = "image"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.model = RKNNModel(self.model_dir / "lprnet.rknn", platform)
        self.model_names = ["lprnet.rknn"]
        self.international_model = None
        self.ppocr_characters = []
        ppocr_model_path = self.model_dir / "ppocr_rec.rknn"
        ppocr_dictionary_path = self.model_dir / "ppocr_keys_v1.txt"
        if ppocr_model_path.exists() and ppocr_dictionary_path.exists():
            self.international_model = RKNNModel(ppocr_model_path, platform)
            with ppocr_dictionary_path.open("r", encoding="utf-8") as dictionary:
                self.ppocr_characters = ["blank"] + [line.rstrip("\r\n") for line in dictionary] + [" "]
            self.model_names.extend(["ppocr_rec.rknn", "ppocr_keys_v1.txt"])
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_russian_plate_number.xml"
        self.plate_cascade = cv2.CascadeClassifier(str(cascade_path))

    @staticmethod
    def _looks_like_plate(image):
        height, width = image.shape[:2]
        ratio = width / max(height, 1)
        return 1.8 <= ratio <= 8.5 and (height <= 240 or width <= 960)

    @staticmethod
    def _expanded_box(box, image_width, image_height, pad_x=0.08, pad_y=0.18):
        x1, y1, x2, y2 = box
        horizontal = max(2, int((x2 - x1) * pad_x))
        vertical = max(2, int((y2 - y1) * pad_y))
        return (
            max(0, x1 - horizontal),
            max(0, y1 - vertical),
            min(image_width, x2 + horizontal),
            min(image_height, y2 + vertical),
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

    @staticmethod
    def _smaller_overlap(first, second):
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
        second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / float(min(first_area, second_area))

    @staticmethod
    def _mask_contours(mask):
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    def _detect_candidates(self, image, max_plates):
        image_height, image_width = image.shape[:2]
        scale = min(1.0, 1280.0 / max(image_width, 1))
        work = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image
        )
        work_height, work_width = work.shape[:2]
        frame_area = float(work_width * work_height)
        gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        candidates = []

        def add_candidate(score, box, style, pad_x=0.08, pad_y=0.18):
            x1, y1, x2, y2 = [int(value) for value in box]
            if x2 <= x1 or y2 <= y1:
                return
            candidates.append(
                {
                    "score": float(score),
                    "box": (x1, y1, x2, y2),
                    "style": style,
                    "pad_x": pad_x,
                    "pad_y": pad_y,
                }
            )

        if not self.plate_cascade.empty():
            minimum_width = max(24, int(work_width * 0.012))
            minimum_height = max(8, int(minimum_width / 7.0))
            for x, y, width, height in self.plate_cascade.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.06,
                minNeighbors=2,
                minSize=(minimum_width, minimum_height),
            ):
                add_candidate(0.95, (x, y, x + width, y + height), "auto")

        raw_gradient = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        minimum = float(raw_gradient.min())
        maximum = float(raw_gradient.max())
        gradient = np.zeros_like(gray)
        if maximum > minimum:
            gradient = ((raw_gradient - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)
            connected = cv2.morphologyEx(
                gradient,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
            )
            connected = cv2.threshold(connected, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            connected = cv2.morphologyEx(
                connected,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)),
                iterations=2,
            )
            connected = cv2.dilate(cv2.erode(connected, None, iterations=1), None, iterations=1)
            for contour in self._mask_contours(connected):
                x, y, width, height = cv2.boundingRect(contour)
                ratio = width / max(height, 1)
                area_ratio = (width * height) / frame_area
                if not (1.8 <= ratio <= 8.5 and 0.00004 <= area_ratio <= 0.18):
                    continue
                if width < max(20, int(work_width * 0.01)) or height < 6:
                    continue
                edge_density = float(np.count_nonzero(gradient[y : y + height, x : x + width] > 70)) / max(
                    1, width * height
                )
                if not 0.045 <= edge_density <= 0.82:
                    continue
                geometry = 1.0 - min(abs(ratio - 4.0) / 5.0, 1.0)
                add_candidate(0.42 + edge_density * 0.36 + geometry * 0.20, (x, y, x + width, y + height), "auto")

        color_specs = (
            ("blue", (90, 60, 35), (145, 255, 255)),
            ("yellow", (12, 55, 50), (43, 255, 255)),
            ("green", (35, 45, 35), (92, 255, 255)),
        )
        for style, lower, upper in color_specs:
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
                iterations=2,
            )
            for contour in self._mask_contours(mask):
                rectangle = cv2.minAreaRect(contour)
                rect_width, rect_height = rectangle[1]
                long_side = max(rect_width, rect_height)
                short_side = max(1.0, min(rect_width, rect_height))
                ratio = long_side / short_side
                area_ratio = (long_side * short_side) / frame_area
                fill_ratio = cv2.contourArea(contour) / max(1.0, long_side * short_side)
                if not (1.8 <= ratio <= 8.5 and 0.00004 <= area_ratio <= 0.65 and fill_ratio >= 0.22):
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                if width < 18 or height < 6:
                    continue
                size_score = 1.0 - min(abs(ratio - 4.0) / 6.0, 1.0)
                add_candidate(0.72 + min(fill_ratio, 1.0) * 0.16 + size_score * 0.08, (x, y, x + width, y + height), style)

        red_low = cv2.inRange(hsv, (0, 70, 40), (15, 255, 255))
        red_high = cv2.inRange(hsv, (165, 70, 40), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_low, red_high)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (29, 3)),
            iterations=2,
        )
        red_mask = cv2.dilate(red_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
        for contour in self._mask_contours(red_mask):
            x, y, width, height = cv2.boundingRect(contour)
            ratio = width / max(height, 1)
            area_ratio = (width * height) / frame_area
            if not (1.2 <= ratio <= 12.0 and 0.00001 <= area_ratio <= 0.16 and width >= 14 and height >= 4):
                continue
            expanded = self._expanded_box((x, y, x + width, y + height), work_width, work_height, 1.2, 0.85)
            x1, y1, x2, y2 = expanded
            region = hsv[y1:y2, x1:x2]
            if region.size == 0:
                continue
            light_fraction = float(np.mean((region[..., 1] < 115) & (region[..., 2] > 105)))
            if light_fraction < 0.18:
                continue
            add_candidate(0.84 + min(light_fraction, 0.6) * 0.15, expanded, "light_red", 0.03, 0.05)

        white_mask = cv2.inRange(hsv, (0, 0, 135), (180, 90, 255))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
            iterations=2,
        )
        for contour in self._mask_contours(white_mask):
            x, y, width, height = cv2.boundingRect(contour)
            ratio = width / max(height, 1)
            area_ratio = (width * height) / frame_area
            fill_ratio = cv2.contourArea(contour) / max(1.0, float(width * height))
            if not (1.8 <= ratio <= 8.5 and 0.00003 <= area_ratio <= 0.35 and fill_ratio >= 0.35):
                continue
            edge_density = float(np.count_nonzero(gradient[y : y + height, x : x + width] > 55)) / max(
                1, width * height
            )
            if edge_density < 0.035:
                continue
            add_candidate(0.62 + min(fill_ratio, 1.0) * 0.12 + min(edge_density, 0.5) * 0.2, (x, y, x + width, y + height), "light")

        selected_work_boxes = []
        selected = []
        def ranking_score(candidate):
            # A bright rectangular plate is a stronger scene-level signal than
            # isolated red regions (tail lights and traffic lights are common).
            style_adjustment = {
                "light": 0.24,
                "blue": 0.08,
                "yellow": 0.08,
                "green": 0.08,
                "light_red": -0.12,
            }.get(candidate["style"], 0.0)
            return candidate["score"] + style_adjustment

        for candidate in sorted(candidates, key=ranking_score, reverse=True):
            work_box = candidate["box"]
            if any(
                self._iou(work_box, existing) > 0.42 or self._smaller_overlap(work_box, existing) > 0.82
                for existing in selected_work_boxes
            ):
                continue
            selected_work_boxes.append(work_box)
            x1, y1, x2, y2 = work_box
            original_box = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
            original_box = self._expanded_box(
                original_box,
                image_width,
                image_height,
                candidate["pad_x"],
                candidate["pad_y"],
            )
            selected.append(
                {
                    "score": candidate["score"],
                    "box": original_box,
                    "style": candidate["style"],
                }
            )
            if len(selected) >= max_plates:
                break
        return selected

    @staticmethod
    def _classify_style(crop, hint="auto"):
        if hint != "auto":
            return hint
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        blue = float(np.mean((hue >= 90) & (hue <= 145) & (saturation >= 60) & (value >= 35)))
        yellow = float(np.mean((hue >= 12) & (hue <= 43) & (saturation >= 55) & (value >= 50)))
        green = float(np.mean((hue >= 35) & (hue <= 92) & (saturation >= 45) & (value >= 35)))
        red = float(np.mean(((hue <= 15) | (hue >= 165)) & (saturation >= 70) & (value >= 40)))
        light = float(np.mean((saturation < 105) & (value > 110)))
        if blue >= 0.16:
            return "blue"
        if yellow >= 0.16:
            return "yellow"
        if green >= 0.16:
            return "green"
        if red >= 0.012 and light >= 0.18:
            return "light_red"
        if light >= 0.35:
            return "light"
        return "unknown"

    @staticmethod
    def _normalize_to_blue(crop, style):
        resized = cv2.resize(crop, (94, 24), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if style == "light_red":
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            red = cv2.bitwise_or(
                cv2.inRange(hsv, (0, 55, 30), (18, 255, 255)),
                cv2.inRange(hsv, (162, 55, 30), (180, 255, 255)),
            )
            ink = cv2.bitwise_or(ink, red)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        normalized = np.zeros((24, 94, 3), dtype=np.uint8)
        normalized[:] = (255, 0, 0)
        normalized[ink > 0] = (255, 255, 255)
        return normalized

    def _recognize_lprnet(self, crop, style, plate_layout):
        if style in {"yellow", "green", "light"}:
            model_input = self._normalize_to_blue(crop, style)
            normalization = "blue_white"
        else:
            model_input = cv2.resize(crop, (94, 24), interpolation=cv2.INTER_CUBIC)
            normalization = "original"

        output = np.asarray(self.model.run(model_input[None])[0])
        labels, _ = decode(output, CHARS)
        logits = output[0].astype(np.float32)
        logits -= logits.max(axis=0, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-8)
        text = labels[0]
        if plate_layout == "international" or style in INTERNATIONAL_STYLES:
            while text and text[0] in CHINESE_PREFIXES:
                text = text[1:]
        return text, float(np.mean(probabilities.max(axis=0))), normalization, "lprnet"

    def _recognize_ppocr(self, crop):
        resized = cv2.resize(crop, (320, 48), interpolation=cv2.INTER_CUBIC)
        model_input = resized.astype(np.float32) / 255.0
        output = np.asarray(self.international_model.run(model_input[None])[0], dtype=np.float32)
        indices = output.argmax(axis=2)[0]
        scores = output.max(axis=2)[0]
        characters = []
        confidences = []
        previous = -1
        for index, score in zip(indices, scores):
            index = int(index)
            if index != previous and index != 0 and index < len(self.ppocr_characters):
                characters.append(self.ppocr_characters[index])
                confidences.append(float(score))
            previous = index
        raw_text = "".join(characters).upper()
        text = re.sub(r"[^0-9A-Z\u0080-\uffff]", "", raw_text)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return text, confidence, "0_1", "ppocr_rec"

    def _recognize(self, crop, style_hint="auto", plate_layout="auto"):
        style = self._classify_style(crop, style_hint)
        if plate_layout == "international":
            style = "light_red" if style == "light_red" else "light"
        use_international = plate_layout == "international" or (
            plate_layout == "auto" and style in INTERNATIONAL_STYLES
        )
        if use_international and self.international_model is not None:
            text, confidence, normalization, recognizer = self._recognize_ppocr(crop)
        else:
            text, confidence, normalization, recognizer = self._recognize_lprnet(crop, style, plate_layout)
        return text, confidence, style, normalization, recognizer

    @staticmethod
    def _perspective_crop(image, points):
        image_height, image_width = image.shape[:2]
        raw = np.asarray(points, dtype=np.float32)
        if raw.shape != (4, 2) or not np.isfinite(raw).all():
            raise ValueError("manual_quad must contain four finite [x, y] points")
        raw[:, 0] = np.clip(raw[:, 0], 0, image_width - 1)
        raw[:, 1] = np.clip(raw[:, 1], 0, image_height - 1)
        hull = cv2.convexHull(raw, clockwise=False, returnPoints=True).reshape(-1, 2)
        if len(hull) != 4 or abs(cv2.contourArea(hull)) < 16:
            raise ValueError("manual_quad must form a non-crossing quadrilateral with a visible area")

        center = hull.mean(axis=0)
        angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
        ordered = hull[np.argsort(angles)]
        ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
        if ordered[1, 0] < ordered[-1, 0]:
            ordered = ordered[[0, 3, 2, 1]]
        top_left, top_right, bottom_right, bottom_left = ordered

        target_width = int(
            round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
        )
        target_height = int(
            round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
        )
        if target_width < 8 or target_height < 4:
            raise ValueError("manual_quad is too small for recognition")
        destination = np.float32(
            [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]]
        )
        transform = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
        crop = cv2.warpPerspective(
            image,
            transform,
            (target_width, target_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        box = (
            int(np.floor(ordered[:, 0].min())),
            int(np.floor(ordered[:, 1].min())),
            int(np.ceil(ordered[:, 0].max())) + 1,
            int(np.ceil(ordered[:, 1].max())) + 1,
        )
        return crop, ordered, box

    @staticmethod
    def _draw_result(preview, candidate, index):
        x1, y1, x2, y2 = candidate["box"]
        accepted = candidate["accepted"]
        color = (0, 220, 80) if accepted else (0, 170, 255)
        if candidate.get("quad"):
            polygon = np.asarray(candidate["quad"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(preview, [polygon], True, color, 3, cv2.LINE_AA)
            for point in polygon.reshape(-1, 2):
                cv2.circle(preview, tuple(point), 5, color, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        ascii_text = "".join(character for character in candidate["text"] if ord(character) < 128)
        label = f"#{index} {ascii_text}".strip() if ascii_text else f"Candidate #{index}"
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
        plate_layout = str(params.get("plate_layout", "auto"))
        if plate_layout not in {"auto", "chinese", "international"}:
            raise ValueError("plate_layout must be auto, chinese, or international")
        detect = bool(params.get("detect", True))
        force_whole_image = bool(params.get("whole_image", False))

        manual_quad = params.get("manual_quad")
        manual_box = params.get("manual_box")
        if manual_quad is not None:
            crop, ordered_quad, box = self._perspective_crop(image, manual_quad)
            candidates = [
                {
                    "score": 1.0,
                    "box": box,
                    "quad": ordered_quad.tolist(),
                    "crop": crop,
                    "rectified_size": [int(crop.shape[1]), int(crop.shape[0])],
                    "style": "auto",
                }
            ]
            source_mode = "manual_quad"
        elif manual_box is not None:
            if not isinstance(manual_box, (list, tuple)) or len(manual_box) != 4:
                raise ValueError("manual_box must be [x1, y1, x2, y2]")
            x1, y1, x2, y2 = [int(value) for value in manual_box]
            box = (
                max(0, min(image_width - 1, x1)),
                max(0, min(image_height - 1, y1)),
                max(1, min(image_width, x2)),
                max(1, min(image_height, y2)),
            )
            candidates = [{"score": 1.0, "box": box, "style": "auto"}]
            source_mode = "manual_box"
        elif force_whole_image or not detect or self._looks_like_plate(image):
            candidates = [{"score": 1.0, "box": (0, 0, image_width, image_height), "style": "auto"}]
            source_mode = "plate_crop"
        else:
            candidates = self._detect_candidates(image, max_plates)
            source_mode = "scene"

        accepted_plates = []
        recognized_candidates = []
        warnings = []
        for index, candidate in enumerate(candidates, start=1):
            x1, y1, x2, y2 = candidate["box"]
            crop = candidate.get("crop")
            if crop is None:
                crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text, recognition_score, style, normalization, recognizer = self._recognize(
                crop,
                candidate.get("style", "auto"),
                plate_layout,
            )
            required_length = min_text_length
            if plate_layout == "international" or style in INTERNATIONAL_STYLES:
                required_length = min(required_length, 4)
                if self.international_model is None and "International plate recognition fallback is unavailable; using the Chinese LPRNet model." not in warnings:
                    warnings.append(
                        "International plate recognition fallback is unavailable; using the Chinese LPRNet model."
                    )
            format_valid = True
            format_reason = ""
            if recognizer == "lprnet" and text:
                format_valid = text[0] in CHINESE_PREFIXES
                if not format_valid:
                    format_reason = "Chinese LPRNet output does not start with a province character"
            elif recognizer == "ppocr_rec" and text:
                format_valid = bool(re.fullmatch(r"[0-9A-Z\u0080-\uffff]{4,12}", text))
                if not format_valid:
                    format_reason = "international plate text does not match the expected alphanumeric format"
            accepted = len(text) >= required_length and recognition_score >= min_score and format_valid
            record = {
                "text": text,
                "recognition_score": round(recognition_score, 4),
                "candidate_score": round(float(candidate["score"]), 4),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "style": style,
                "recognizer": recognizer,
                "normalization": normalization,
                "accepted": accepted,
            }
            if candidate.get("quad"):
                record["quad"] = [[round(float(x), 2), round(float(y), 2)] for x, y in candidate["quad"]]
                record["rectified_size"] = candidate["rectified_size"]
            if not accepted:
                if len(text) < required_length:
                    record["reject_reason"] = f"text length {len(text)} is below {required_length}"
                elif recognition_score < min_score:
                    record["reject_reason"] = (
                        f"recognition score {recognition_score:.4f} is below {min_score:.4f}"
                    )
                else:
                    record["reject_reason"] = format_reason
            recognized_candidates.append(record)
            self._draw_result(preview, record, index)
            if accepted:
                accepted_plates.append(dict(record))

        summary = f"Plates: {len(accepted_plates)}  Candidates: {len(recognized_candidates)}"
        cv2.rectangle(preview, (0, 0), (min(preview.shape[1], max(320, 16 + len(summary) * 11)), 36), (15, 15, 15), -1)
        cv2.putText(preview, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2)
        return {
            "plates": accepted_plates,
            "candidates": recognized_candidates,
            "count": len(accepted_plates),
            "candidate_count": len(recognized_candidates),
            "source_mode": source_mode,
            "plate_layout": plate_layout,
            "warnings": warnings,
            "image": {"width": image_width, "height": image_height},
        }, preview

    def warmup_preview(self, setter):
        image = cv2.imread(str(self.model_dir / "test.jpg"))
        if image is not None:
            _, preview = self.predict(image, {"whole_image": True, "plate_layout": "chinese"})
            setter(preview)


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
