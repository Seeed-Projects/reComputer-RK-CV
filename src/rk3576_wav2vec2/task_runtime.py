from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from rknn_runtime import RKNNModel


TOKENS = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>", 4: "|", 5: "E", 6: "T", 7: "A", 8: "O", 9: "N", 10: "I", 11: "H", 12: "S", 13: "R", 14: "D", 15: "L", 16: "U", 17: "M", 18: "W", 19: "C", 20: "F", 21: "G", 22: "Y", 23: "P", 24: "B", 25: "V", 26: "K", 27: "'", 28: "X", 29: "J", 30: "Q", 31: "Z"}
SAMPLE_RATE = 16000
MAX_SAMPLES = SAMPLE_RATE * 20
OVERLAP_SAMPLES = SAMPLE_RATE // 2


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
    if rate != SAMPLE_RATE:
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE, rate)
    audio = np.asarray(audio, dtype=np.float32)
    audio -= float(np.mean(audio))

    peak_before = float(np.max(np.abs(audio)))
    clipped = int(np.count_nonzero(np.abs(audio) >= 0.999))
    frame_length = SAMPLE_RATE // 50
    hop = frame_length // 2
    if len(audio) >= frame_length:
        starts = np.arange(0, len(audio) - frame_length + 1, hop)
        rms_frames = np.sqrt(np.asarray([
            np.mean(audio[start:start + frame_length] ** 2) for start in starts
        ]))
        threshold = max(10 ** (-50 / 20), float(rms_frames.max()) * 0.08)
        active = np.flatnonzero(rms_frames >= threshold)
        if active.size:
            pad = int(0.2 * SAMPLE_RATE)
            first = max(0, int(starts[active[0]]) - pad)
            last = min(len(audio), int(starts[active[-1]]) + frame_length + pad)
            audio = audio[first:last]
    if not len(audio):
        raise ValueError("No speech was found in the audio")

    rms_before = float(np.sqrt(np.mean(audio ** 2)))
    if rms_before > 1e-7:
        target_rms = 0.1
        gain = min(target_rms / rms_before, 10 ** (10 / 20))
        if peak_before > 0:
            gain = min(gain, 0.98 / peak_before)
        audio *= gain
    return audio, original_duration, {
        "input_sample_rate": int(rate),
        "peak": round(peak_before, 5),
        "rms_dbfs": round(20 * np.log10(max(rms_before, 1e-8)), 2),
        "clipped_samples": clipped,
    }


def make_chunks(audio):
    if len(audio) <= MAX_SAMPLES:
        return [audio]
    chunks = []
    start = 0
    while start < len(audio):
        end = min(len(audio), start + MAX_SAMPLES)
        chunks.append(audio[start:end])
        if end == len(audio):
            break
        start = end - OVERLAP_SAMPLES
    return chunks


def model_input(audio):
    return np.pad(audio, (0, MAX_SAMPLES - len(audio))).astype(np.float32)[None]


def ctc_decode(logits, beam_width):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits[0]
    if beam_width == 1:
        ids = np.argmax(logits, axis=-1).tolist()
        ids = [token for index, token in enumerate(ids) if index == 0 or token != ids[index - 1]]
    else:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs[:, 1:4] = -np.inf
        beams = {(): (0.0, -np.inf)}
        token_limit = min(log_probs.shape[1], max(8, beam_width + 2))
        for frame in log_probs:
            candidates = np.argpartition(frame, -token_limit)[-token_limit:]
            if 0 not in candidates:
                candidates = np.append(candidates, 0)
            next_beams = {}
            for prefix, (blank_score, text_score) in beams.items():
                total = np.logaddexp(blank_score, text_score)
                old_blank, old_text = next_beams.get(prefix, (-np.inf, -np.inf))
                next_beams[prefix] = (np.logaddexp(old_blank, total + frame[0]), old_text)
                for token in candidates:
                    token = int(token)
                    if token <= 3:
                        continue
                    if prefix and token == prefix[-1]:
                        old_blank, old_text = next_beams.get(prefix, (-np.inf, -np.inf))
                        next_beams[prefix] = (old_blank, np.logaddexp(old_text, text_score + frame[token]))
                        extended = prefix + (token,)
                        old_blank, old_text = next_beams.get(extended, (-np.inf, -np.inf))
                        next_beams[extended] = (old_blank, np.logaddexp(old_text, blank_score + frame[token]))
                    else:
                        extended = prefix + (token,)
                        old_blank, old_text = next_beams.get(extended, (-np.inf, -np.inf))
                        next_beams[extended] = (old_blank, np.logaddexp(old_text, total + frame[token]))
            beams = dict(sorted(
                next_beams.items(), key=lambda item: np.logaddexp(*item[1]), reverse=True
            )[:beam_width])
        ids = list(max(beams, key=lambda prefix: np.logaddexp(*beams[prefix])))
    return "".join(
        " " if token == 4 else TOKENS.get(token, "") if token > 4 else "" for token in ids
    ).strip()


def merge_transcripts(parts):
    merged = ""
    for part in parts:
        part = " ".join(part.split())
        if not part:
            continue
        overlap = 0
        for size in range(min(120, len(merged), len(part)), 3, -1):
            if merged[-size:] == part[:size]:
                overlap = size
                break
        if not merged:
            merged = part
        elif overlap:
            merged += part[overlap:]
        else:
            merged += " " + part
    return merged.strip()


class Runtime:
    name = "wav2vec2"
    input_kind = "audio"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.model = RKNNModel(self.model_dir / "wav2vec2.rknn", platform)
        self.model_names = ["wav2vec2.rknn"]
        self.config = {"beam_width": 8}

    def describe(self):
        return {
            "task": "English automatic speech recognition",
            "sample_rate": 16000,
            "segment_duration_seconds": 20,
            "language": "English",
        }

    def get_config(self):
        return dict(self.config)

    def update_config(self, values):
        beam_width = int(values.get("beam_width", self.config["beam_width"]))
        if not 1 <= beam_width <= 32:
            raise ValueError("beam_width must be between 1 and 32")
        self.config["beam_width"] = beam_width
        return self.get_config()

    def predict(self, contents, params):
        audio, original_duration, quality = read_audio(contents)
        beam_width = int(params.get("beam_width", self.config["beam_width"]))
        if not 1 <= beam_width <= 32:
            raise ValueError("beam_width must be between 1 and 32")
        chunks = make_chunks(audio)
        texts = []
        frames = 0
        for chunk in chunks:
            logits = np.asarray(self.model.run(model_input(chunk))[0])
            frames += int(logits.shape[-2])
            texts.append(ctc_decode(logits, beam_width))
        warnings = []
        if quality["clipped_samples"]:
            warnings.append("The source contains clipped samples; reduce the recording level")
        if len(chunks) > 1:
            warnings.append(f"Long audio was transcribed in {len(chunks)} overlapping segments")
        return {
            "text": merge_transcripts(texts),
            "frames": frames,
            "sample_rate": SAMPLE_RATE,
            "original_duration_seconds": round(original_duration, 3),
            "processed_duration_seconds": round(len(audio) / SAMPLE_RATE, 3),
            "segments": len(chunks),
            "beam_width": beam_width,
            "audio_quality": quality,
            "truncated": False,
            "warnings": warnings,
        }


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
