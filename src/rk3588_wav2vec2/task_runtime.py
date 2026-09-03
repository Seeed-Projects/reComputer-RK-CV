from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from rknn_runtime import RKNNModel


TOKENS = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>", 4: "|", 5: "E", 6: "T", 7: "A", 8: "O", 9: "N", 10: "I", 11: "H", 12: "S", 13: "R", 14: "D", 15: "L", 16: "U", 17: "M", 18: "W", 19: "C", 20: "F", 21: "G", 22: "Y", 23: "P", 24: "B", 25: "V", 26: "K", 27: "'", 28: "X", 29: "J", 30: "Q", 31: "Z"}


def read_audio(contents):
    try:
        audio, rate = sf.read(BytesIO(contents), dtype="float32")
    except (RuntimeError, ValueError) as exc:
        raise ValueError("The uploaded file is not a supported audio file") from exc
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("The audio is empty or contains invalid samples")
    original_duration = len(audio) / float(rate)
    if rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, rate)
    truncated = len(audio) > 320000
    audio = np.asarray(audio[:320000], dtype=np.float32)
    processed_duration = len(audio) / 16000.0
    model_input = np.pad(audio, (0, max(0, 320000 - len(audio)))).astype(np.float32)[None]
    return model_input, original_duration, processed_duration, truncated


class Runtime:
    name = "wav2vec2"
    input_kind = "audio"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.model = RKNNModel(self.model_dir / "wav2vec2.rknn", platform)
        self.model_names = ["wav2vec2.rknn"]

    def describe(self):
        return {
            "task": "English automatic speech recognition",
            "sample_rate": 16000,
            "max_duration_seconds": 20,
        }

    def get_config(self):
        return {}

    def update_config(self, values):
        if values:
            raise ValueError("Wav2Vec2 has no runtime settings")
        return {}

    def predict(self, contents, params):
        model_input, original_duration, processed_duration, truncated = read_audio(contents)
        logits = np.asarray(self.model.run(model_input)[0])
        ids = np.argmax(logits, axis=-1)[0].tolist()
        collapsed = [token for index, token in enumerate(ids) if index == 0 or token != ids[index - 1]]
        text = "".join(" " if token == 4 else TOKENS.get(token, "") if token > 4 else "" for token in collapsed)
        return {
            "text": text.strip(),
            "frames": int(logits.shape[-2]),
            "sample_rate": 16000,
            "original_duration_seconds": round(original_duration, 3),
            "processed_duration_seconds": round(processed_duration, 3),
            "truncated": truncated,
            "warnings": ["Only the first 20 seconds were transcribed"] if truncated else [],
        }


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
