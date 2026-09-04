from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

from rknn_runtime import RKNNModel


MAX_LENGTH = 200
MIN_SPEAKING_RATE = 0.6
MAX_SPEAKING_RATE = 1.4
VOCAB = {' ': 19, "'": 1, '-': 14, '0': 23, '1': 15, '2': 28, '3': 11, '4': 27, '5': 35, '6': 36, '_': 30, 'a': 26, 'b': 24, 'c': 12, 'd': 5, 'e': 7, 'f': 20, 'g': 37, 'h': 6, 'i': 18, 'j': 16, 'k': 0, 'l': 21, 'm': 17, 'n': 29, 'o': 22, 'p': 13, 'q': 34, 'r': 25, 's': 8, 't': 33, 'u': 4, 'v': 32, 'w': 9, 'x': 31, 'y': 3, 'z': 2, '–': 10}


def tokenize(text):
    text = text.strip().lower()
    if not text:
        raise ValueError("Enter English text to synthesize")
    unsupported = sorted(set(char for char in text if char not in VOCAB))
    if unsupported:
        display = " ".join(repr(char) for char in unsupported[:12])
        raise ValueError(f"Unsupported characters: {display}")
    ids = []
    for char in text:
        ids.extend((0, VOCAB[char]))
    ids.append(0)
    if len(ids) > MAX_LENGTH:
        raise ValueError(f"Text is too long: {len(text)} characters; the model accepts at most 99")
    mask = [1] * len(ids)
    ids += [0] * (MAX_LENGTH - len(ids))
    mask += [0] * (MAX_LENGTH - len(mask))
    return np.asarray(ids, dtype=np.int64)[None], np.asarray(mask, dtype=np.int64)[None]


def build_attention(log_duration, input_mask, speaking_rate=1.0):
    duration = np.ceil(np.exp(log_duration) * input_mask / speaking_rate).astype(np.int64)
    requested = max(1, int(duration.sum()))
    output_length = MAX_LENGTH * 2
    predicted = min(requested, output_length)
    output_mask = (np.arange(output_length)[None, None, :] < predicted).astype(input_mask.dtype)
    attention = np.zeros((1, 1, output_length, MAX_LENGTH), dtype=np.float32)
    cursor = 0
    for index, amount in enumerate(duration.reshape(-1)):
        end = min(output_length, cursor + int(amount))
        attention[0, 0, cursor:end, index] = 1
        cursor = end
    attention *= input_mask[:, :, None, :] * output_mask[:, :, :, None]
    return attention, output_mask, predicted, requested > output_length


class Runtime:
    name = "mms_tts"
    input_kind = "text"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.encoder = RKNNModel(self.model_dir / "encoder.rknn", platform, core_index=0)
        self.decoder = RKNNModel(self.model_dir / "decoder.rknn", platform, core_index=1)
        self.model_names = ["encoder.rknn", "decoder.rknn"]
        self.config = {"speaking_rate": 1.0}

    def describe(self):
        return {
            "task": "English text-to-speech synthesis",
            "sample_rate": 16000,
            "max_characters": 99,
            "speaking_rate_range": [MIN_SPEAKING_RATE, MAX_SPEAKING_RATE],
            "supported_characters": "".join(sorted(VOCAB)),
        }

    def get_config(self):
        return dict(self.config)

    def update_config(self, values):
        speaking_rate = float(values.get("speaking_rate", self.config["speaking_rate"]))
        if not MIN_SPEAKING_RATE <= speaking_rate <= MAX_SPEAKING_RATE:
            raise ValueError(
                f"speaking_rate must be between {MIN_SPEAKING_RATE} and {MAX_SPEAKING_RATE}"
            )
        self.config["speaking_rate"] = speaking_rate
        return self.get_config()

    def predict(self, text, params):
        speaking_rate = float(params.get("speaking_rate", self.config["speaking_rate"]))
        if not MIN_SPEAKING_RATE <= speaking_rate <= MAX_SPEAKING_RATE:
            raise ValueError(
                f"speaking_rate must be between {MIN_SPEAKING_RATE} and {MAX_SPEAKING_RATE}"
            )
        input_ids, attention_mask = tokenize(text)
        log_duration, input_padding_mask, means, log_variances = self.encoder.run([input_ids, attention_mask])
        attention, output_mask, predicted, duration_clipped = build_attention(
            np.asarray(log_duration), np.asarray(input_padding_mask), speaking_rate
        )
        waveform = np.asarray(self.decoder.run([attention, output_mask, means, log_variances])[0]).reshape(-1)
        waveform = waveform[:predicted * 256]
        if not len(waveform) or not np.isfinite(waveform).all():
            raise RuntimeError("The model generated invalid audio")
        output = BytesIO()
        sf.write(output, waveform, 16000, format="WAV")
        return {
            "text": text,
            "sample_rate": 16000,
            "samples": len(waveform),
            "duration_seconds": round(len(waveform) / 16000.0, 3),
            "speaking_rate": speaking_rate,
            "duration_clipped": duration_clipped,
            "warnings": ["The generated duration reached the decoder limit; use shorter text or a faster speaking rate"] if duration_clipped else [],
            "_audio_bytes": output.getvalue(),
        }


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
