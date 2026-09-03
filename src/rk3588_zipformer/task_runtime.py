from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from rknn_runtime import RKNNModel


CONFIG = {
    "x": (1, 103, 80),
    **{f"cached_len_{i}": (2, 1) for i in range(5)},
    **{f"cached_avg_{i}": (2, 1, 256) for i in range(5)},
    **{f"cached_key_{i}": (2, size, 1, 192) for i, size in enumerate((192, 96, 48, 24, 96))},
    **{f"cached_val_{i}": (2, size, 1, 96) for i, size in enumerate((192, 96, 48, 24, 96))},
    **{f"cached_val2_{i}": (2, size, 1, 96) for i, size in enumerate((192, 96, 48, 24, 96))},
    **{f"cached_conv1_{i}": (2, 1, 256, 30) for i in range(5)},
    **{f"cached_conv2_{i}": (2, 1, 256, 30) for i in range(5)},
}


def read_audio(contents):
    try:
        audio, rate = sf.read(BytesIO(contents), dtype="float32")
    except (RuntimeError, ValueError) as exc:
        raise ValueError("The uploaded file is not a supported audio file") from exc
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("The audio is empty or contains invalid samples")
    if rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, rate)
    return np.asarray(audio, dtype=np.float32)


def hz_to_mel(frequency):
    return 1127.0 * np.log1p(frequency / 700.0)


def kaldi_fbank(audio):
    frame_length, frame_shift, fft_size = 400, 160, 512
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    half = frame_length // 2
    number = max(1, (len(audio) + frame_shift // 2) // frame_shift)
    padded = np.pad(audio, (half - frame_shift // 2, half + frame_shift // 2), mode="reflect")
    starts = np.arange(number) * frame_shift
    frames = np.stack([padded[start:start + frame_length] for start in starts])
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] *= 0.03
    spectrum = np.abs(np.fft.rfft(frames * (np.hanning(frame_length) ** 0.85), n=fft_size)) ** 2

    mel_points = np.linspace(hz_to_mel(20.0), hz_to_mel(7600.0), 82)
    frequencies = np.arange(fft_size // 2 + 1) * 16000.0 / fft_size
    frequency_mels = hz_to_mel(frequencies)
    filters = np.zeros((80, len(frequencies)), dtype=np.float32)
    for index in range(80):
        left, center, right = mel_points[index:index + 3]
        filters[index] = np.maximum(0, np.minimum(
            (frequency_mels - left) / (center - left),
            (right - frequency_mels) / (right - center),
        ))
    energies = np.maximum(spectrum @ filters.T, np.finfo(np.float32).eps)
    return np.log(energies).astype(np.float32)


class StreamingSession:
    def __init__(self, runtime):
        self.runtime = runtime
        self.inputs = []
        for name, shape in CONFIG.items():
            dtype = np.int64 if "cached_len" in name else np.float32
            self.inputs.append(np.zeros(shape, dtype=dtype))
        self.audio = np.empty(0, dtype=np.float32)
        self.processed_features = 0
        self.hypothesis = [0, 0]
        self.decoder_output = runtime.decoder.run(np.asarray([self.hypothesis], dtype=np.int64))[0]
        self.timestamps = []
        self.frame_offset = 0

    def _run_block(self, frames):
        if len(frames) < 103:
            frames = np.pad(frames, ((0, 103 - len(frames)), (0, 0)))
        self.inputs[0] = frames[None].astype(np.float32)
        outputs = self.runtime.encoder.run(self.inputs)
        for index in range(1, len(self.inputs)):
            value = np.asarray(outputs[index])
            if index > 10 and value.ndim == 4 and value.shape != self.inputs[index].shape:
                value = np.transpose(value, (0, 2, 3, 1))
            self.inputs[index] = value
        encoder_output = np.asarray(outputs[0]).squeeze(0)
        for frame_index in range(encoder_output.shape[0]):
            current = encoder_output[frame_index:frame_index + 1]
            logits = np.asarray(self.runtime.joiner.run([current, self.decoder_output])[0]).squeeze(0)
            token = int(np.argmax(logits))
            if token not in (0, 2):
                self.hypothesis.append(token)
                self.timestamps.append(round((self.frame_offset + frame_index) * 0.04, 2))
                self.decoder_output = self.runtime.decoder.run(
                    np.asarray([self.hypothesis[-2:]], dtype=np.int64)
                )[0]
        self.frame_offset += encoder_output.shape[0]

    def feed_audio(self, samples, final=False):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples):
            self.audio = np.concatenate((self.audio, samples))
        features = kaldi_fbank(self.audio) if len(self.audio) else np.empty((0, 80), dtype=np.float32)
        processed = False
        while len(features) - self.processed_features >= 103:
            self._run_block(features[self.processed_features:self.processed_features + 103])
            self.processed_features += 96
            processed = True
        if final and self.processed_features < len(features):
            self._run_block(features[self.processed_features:self.processed_features + 103])
            self.processed_features = len(features)
            processed = True
        return self.result(processed)

    def feed_pcm16(self, contents, final=False):
        if len(contents) % 2:
            raise ValueError("PCM chunks must contain 16-bit little-endian samples")
        samples = np.frombuffer(contents, dtype="<i2").astype(np.float32) / 32768.0
        return self.feed_audio(samples, final=final)

    def result(self, processed=False):
        ids = self.hypothesis[2:]
        text = "".join(self.runtime.vocab.get(token, "") for token in ids).replace("▁", " ").strip()
        return {
            "text": text,
            "tokens": ids,
            "timestamps": self.timestamps,
            "audio_duration_seconds": round(len(self.audio) / 16000.0, 3),
            "processed": processed,
        }


class Runtime:
    name = "zipformer"
    input_kind = "audio"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.encoder = RKNNModel(self.model_dir / "encoder.rknn", platform, core_index=0)
        self.decoder = RKNNModel(self.model_dir / "decoder.rknn", platform, core_index=1)
        self.joiner = RKNNModel(self.model_dir / "joiner.rknn", platform, core_index=2)
        self.model_names = ["encoder.rknn", "decoder.rknn", "joiner.rknn"]
        self.vocab = {}
        for line in (self.model_dir / "vocab.txt").read_text().splitlines():
            token, identifier = line.rsplit(" ", 1)
            self.vocab[int(identifier)] = token

    def describe(self):
        return {
            "task": "Chinese-English streaming automatic speech recognition",
            "sample_rate": 16000,
            "stream_format": "pcm_s16le",
            "supports_websocket": True,
        }

    def get_config(self):
        return {}

    def update_config(self, values):
        if values:
            raise ValueError("Zipformer has no runtime settings")
        return {}

    def create_stream_session(self):
        return StreamingSession(self)

    def predict(self, contents, params):
        session = self.create_stream_session()
        return session.feed_audio(read_audio(contents), final=True)


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
