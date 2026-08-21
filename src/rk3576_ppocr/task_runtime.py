import base64
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rknn_runtime import RKNNModel
from utils.db_postprocess import DBPostProcess, DetPostProcess
from utils.operators import DetResizeForTest, NormalizeImage
from utils.rec_postprocess import CTCLabelDecode


def sorted_boxes(boxes):
    """Sort quadrilaterals from top to bottom and left to right."""
    ordered = list(sorted(boxes, key=lambda box: (box[0][1], box[0][0])))
    for index in range(len(ordered) - 1):
        for current in range(index, -1, -1):
            same_line = abs(ordered[current + 1][0][1] - ordered[current][0][1]) < 10
            if same_line and ordered[current + 1][0][0] < ordered[current][0][0]:
                ordered[current], ordered[current + 1] = ordered[current + 1], ordered[current]
            else:
                break
    return ordered


def perspective_crop(image, points):
    """Rectify a four-point text polygon using the official PPOCR-System method."""
    points = np.asarray(points, dtype=np.float32)
    width = max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]),
    )
    height = max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]),
    )
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    target = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
        crop = np.rot90(crop).copy()
    return crop


def encode_crop(crop):
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class Runtime:
    name = "ppocr"
    input_kind = "image"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.det_model = RKNNModel(
            self.model_dir / "ppocr_det.rknn", platform, core_index=0
        )
        self.rec_model = RKNNModel(
            self.model_dir / "ppocr_rec.rknn", platform, core_index=1
        )
        self.model_names = ["ppocr_det.rknn", "ppocr_rec.rknn"]
        self.det_resize = DetResizeForTest(image_shape=[480, 480])
        self.det_normalize = NormalizeImage(
            std=[1, 1, 1], mean=[0, 0, 0], scale="1.", order="hwc"
        )
        self.rec_normalize = NormalizeImage(
            std=[1, 1, 1], mean=[0, 0, 0], scale="1./255.", order="hwc"
        )
        self.det_filter = DetPostProcess()
        self.decoder = CTCLabelDecode(
            character_dict_path=str(self.model_dir / "ppocr_keys_v1.txt"),
            use_space_char=True,
        )
        self.font_path = self.model_dir / "simfang.ttf"

    def detect(self, image, params):
        prepared = self.det_resize({"image": image.copy()})
        prepared = self.det_normalize(prepared)
        output = self.det_model.run(prepared["image"][None])[0].astype(np.float32)
        postprocess = DBPostProcess(
            thresh=float(params["det_threshold"]),
            box_thresh=float(params["box_threshold"]),
            max_candidates=1000,
            unclip_ratio=float(params["unclip_ratio"]),
            use_dilation=False,
            score_mode="fast",
        )
        result = postprocess({"maps": output}, prepared["shape"])
        return self.det_filter.filter_tag_det_res(result[0]["points"], image.shape)

    def recognize(self, crop):
        resized = cv2.resize(crop, (320, 48), interpolation=cv2.INTER_LINEAR)
        prepared = self.rec_normalize({"image": resized})
        output = self.rec_model.run(prepared["image"][None])[0].astype(np.float32)
        decoded = self.decoder(output)
        return str(decoded[0][0]), float(decoded[0][1])

    def annotate(self, image, lines):
        preview = image.copy()
        for line in lines:
            points = np.asarray(line["box"], dtype=np.int32)
            color = (0, 210, 118) if line["accepted"] else (120, 120, 120)
            cv2.polylines(preview, [points], True, color, 2, cv2.LINE_AA)

        canvas = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        font_size = max(16, min(30, image.shape[1] // 45))
        try:
            font = ImageFont.truetype(str(self.font_path), font_size)
        except OSError:
            font = ImageFont.load_default()
        for line in lines:
            x = max(0, int(min(point[0] for point in line["box"])))
            y = max(0, int(min(point[1] for point in line["box"])) - font_size - 6)
            label = f'{line["index"]}. {line["text"]} ({line["confidence"]:.2f})'
            bounds = draw.textbbox((x, y), label, font=font)
            draw.rectangle(bounds, fill=(7, 18, 25))
            color = (0, 230, 118) if line["accepted"] else (180, 180, 180)
            draw.text((x, y), label, font=font, fill=color)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def predict(self, image, params):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        started = time.perf_counter()
        det_started = time.perf_counter()
        boxes = sorted_boxes(self.detect(image, params))
        det_ms = (time.perf_counter() - det_started) * 1000

        detected_count = len(boxes)
        boxes = boxes[: int(params["max_results"])]
        include_crops = bool(params.get("include_crops", False))
        lines = []
        rec_ms = 0.0
        for index, box in enumerate(boxes, start=1):
            crop = perspective_crop(image, box)
            rec_started = time.perf_counter()
            text, confidence = self.recognize(crop)
            rec_ms += (time.perf_counter() - rec_started) * 1000
            line = {
                "index": index,
                "text": text,
                "confidence": round(confidence, 6),
                "accepted": confidence >= float(params["drop_score"]),
                "box": np.rint(box).astype(np.int32).tolist(),
                "crop_size": {"width": int(crop.shape[1]), "height": int(crop.shape[0])},
            }
            if include_crops:
                line["crop_image"] = encode_crop(crop)
            lines.append(line)

        accepted = [line for line in lines if line["accepted"]]
        preview = self.annotate(image, lines)
        total_ms = (time.perf_counter() - started) * 1000
        result = {
            "text": "\n".join(line["text"] for line in accepted if line["text"]),
            "lines": lines,
            "count": len(accepted),
            "detected_count": detected_count,
            "processed_count": len(lines),
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "timing_ms": {
                "detection": round(det_ms, 2),
                "recognition": round(rec_ms, 2),
                "total": round(total_ms, 2),
            },
        }
        return result, preview

    def release(self):
        self.det_model.release()
        self.rec_model.release()


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
