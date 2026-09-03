from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from rknn_runtime import RKNNModel


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
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) > 16000 * 300:
        raise ValueError("Audio duration must not exceed 300 seconds")
    return audio


class Runtime:
    name = "yamnet"
    input_kind = "audio"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.model = RKNNModel(self.model_dir / "yamnet.rknn", platform)
        self.model_names = ["yamnet.rknn"]
        self.labels = {}
        for line in (self.model_dir / "yamnet_class_map.txt").read_text().splitlines():
            key, _, value = line.partition(" ")
            self.labels[int(key)] = value
        self.config = {"topk": 5, "threshold": 0.1, "hop_seconds": 1.0}

    def describe(self):
        return {
            "task": "Audio event classification",
            "class_count": len(self.labels),
            "sample_rate": 16000,
            "window_seconds": 3.0,
            "max_duration_seconds": 300,
        }

    def get_config(self):
        return dict(self.config)

    def update_config(self, values):
        updated = dict(self.config)
        updated.update(values)
        updated["topk"] = int(updated["topk"])
        updated["threshold"] = float(updated["threshold"])
        updated["hop_seconds"] = float(updated["hop_seconds"])
        if not 1 <= updated["topk"] <= 20:
            raise ValueError("topk must be between 1 and 20")
        if not 0 <= updated["threshold"] <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.25 <= updated["hop_seconds"] <= 3:
            raise ValueError("hop_seconds must be between 0.25 and 3")
        self.config = updated
        return dict(self.config)

    def _rank(self, scores, topk):
        indices = np.argsort(scores)[-topk:][::-1]
        return [
            {"id": int(index), "label": self.labels.get(int(index), str(index)), "score": float(scores[index])}
            for index in indices
        ]

    def predict(self, contents, params):
        config = self.update_config(params)
        audio = read_audio(contents)
        window_samples = 48000
        hop_samples = max(1, int(config["hop_seconds"] * 16000))
        starts = list(range(0, max(1, len(audio)), hop_samples))
        all_scores = []
        segments = []
        for start in starts:
            window = audio[start:start + window_samples]
            if not len(window):
                break
            model_input = np.pad(window, (0, window_samples - len(window))).astype(np.float32)[None]
            scores = np.asarray(self.model.run(model_input)[2]).mean(axis=0)
            all_scores.append(scores)
            ranked = self._rank(scores, config["topk"])
            visible = [item for item in ranked if item["score"] >= config["threshold"]]
            if not visible:
                visible = ranked[:1]
            segments.append({
                "start": round(start / 16000.0, 3),
                "end": round(min(len(audio), start + window_samples) / 16000.0, 3),
                "top_class": ranked[0],
                "predictions": visible,
            })
            if start + window_samples >= len(audio):
                break

        mean_scores = np.mean(all_scores, axis=0)
        predictions = self._rank(mean_scores, config["topk"])
        events = []
        for segment in segments:
            label = segment["top_class"]["label"]
            if events and events[-1]["label"] == label and segment["start"] <= events[-1]["end"]:
                events[-1]["end"] = segment["end"]
                events[-1]["score"] = max(events[-1]["score"], segment["top_class"]["score"])
            else:
                events.append({
                    "label": label,
                    "score": segment["top_class"]["score"],
                    "start": segment["start"],
                    "end": segment["end"],
                })
        return {
            "duration_seconds": round(len(audio) / 16000.0, 3),
            "sample_rate": 16000,
            "window_seconds": 3.0,
            "hop_seconds": config["hop_seconds"],
            "top_class": predictions[0],
            "predictions": predictions,
            "segments": segments,
            "events": events,
        }


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
