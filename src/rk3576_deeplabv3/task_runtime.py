from pathlib import Path

import cv2
import numpy as np

from rknn_runtime import RKNNModel


LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tv",
]


def pascal_color(index):
    red = green = blue = 0
    value = index
    for shift in range(8):
        red |= ((value >> 0) & 1) << (7 - shift)
        green |= ((value >> 1) & 1) << (7 - shift)
        blue |= ((value >> 2) & 1) << (7 - shift)
        value >>= 3
    return blue, green, red


class Runtime:
    name = "deeplabv3"
    input_kind = "image"

    def __init__(self, platform, model_path, sample_path):
        self.platform = platform
        self.model_path = Path(model_path)
        self.sample_path = Path(sample_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"DeepLabV3 model not found: {self.model_path}")
        self.model = RKNNModel(self.model_path, platform)
        self.model_names = [self.model_path.name]

    def predict(self, image, params):
        if image is None:
            raise ValueError("Input image is empty")
        overlay_alpha = float(params.get("overlay_alpha", 0.5))
        if not 0 <= overlay_alpha <= 1:
            raise ValueError("overlay_alpha must be between 0 and 1")

        original = image.copy()
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(
            cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        output = np.asarray(self.model.run(rgb[None])[0])
        if output.ndim != 4:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        if output.shape[-1] == len(LABELS):
            logits = output[0]
        elif output.shape[1] == len(LABELS):
            logits = output[0].transpose(1, 2, 0)
        else:
            raise ValueError(
                f"Expected 21 output classes, got output shape {output.shape}"
            )

        logits = cv2.resize(
            logits,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = np.argmax(logits, axis=-1).astype(np.uint8)
        color = np.zeros_like(original)
        present = []
        for index in np.unique(mask):
            class_id = int(index)
            color[mask == index] = pascal_color(class_id)
            present.append(
                {
                    "id": class_id,
                    "class": LABELS[class_id],
                    "pixels": int(np.sum(mask == index)),
                }
            )

        preview = cv2.addWeighted(
            original,
            1.0 - overlay_alpha,
            color,
            overlay_alpha,
            0,
        )
        return {
            "classes": present,
            "width": width,
            "height": height,
            "overlay_alpha": overlay_alpha,
        }, preview

    def warmup_preview(self, setter, params):
        image = cv2.imread(str(self.sample_path))
        if image is None:
            raise FileNotFoundError(
                f"DeepLabV3 sample image not found: {self.sample_path}"
            )
        _, preview = self.predict(image, params)
        setter(preview)


def create_runtime(platform, model_path, sample_path):
    return Runtime(platform, model_path, sample_path)
