import re
from io import BytesIO
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from rknn_runtime import RKNNModel


def bytes_to_unicode():
    values = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, chars)))


class ClipTokenizer:
    def __init__(self, header_path):
        header = Path(header_path).read_text()
        merges_text = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", header)).decode("utf-8")
        merges = [tuple(line.split()) for line in merges_text.splitlines()[1:] if len(line.split()) == 2]
        byte_values = bytes_to_unicode()
        vocab = list(byte_values.values()) + [value + "</w>" for value in byte_values.values()] + [a + b for a, b in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {token: index for index, token in enumerate(vocab)}
        self.ranks = {pair: index for index, pair in enumerate(merges)}
        self.byte_encoder = byte_values

    @lru_cache(maxsize=4096)
    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(word[index], word[index + 1]) for index in range(len(word) - 1)}
            pair = min(pairs, key=lambda item: self.ranks.get(item, float("inf")))
            if pair not in self.ranks:
                break
            merged = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and word[index] == pair[0] and word[index + 1] == pair[1]:
                    merged.append(pair[0] + pair[1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
        return word

    def encode(self, text, length=20):
        text = " ".join(text.strip().lower().split())
        pattern = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]+", re.I)
        tokens = [49406]
        for match in pattern.findall(text):
            encoded = "".join(self.byte_encoder[value] for value in match.encode("utf-8"))
            tokens.extend(self.encoder[piece] for piece in self.bpe(encoded))
        tokens = tokens[:length - 1] + [49407]
        return np.asarray(tokens + [49407] * (length - len(tokens)), dtype=np.int64)[None]


class Runtime:
    name = "clip"
    input_kind = "image"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.image_model = RKNNModel(self.model_dir / "clip_images.rknn", platform, core_index=0)
        self.text_model = RKNNModel(self.model_dir / "clip_text.rknn", platform, core_index=1)
        self.tokenizer = ClipTokenizer(self.model_dir / "clip_vocab.h")
        self.model_names = ["clip_images.rknn", "clip_text.rknn"]
        self.config = {"topk": 5}

    def describe(self):
        return {
            "task": "Zero-shot image classification and text-to-image retrieval",
            "image_size": [224, 224],
            "max_text_tokens": 20,
            "max_retrieval_images": 32,
        }

    def get_config(self):
        return dict(self.config)

    def update_config(self, values):
        topk = int(values.get("topk", self.config["topk"]))
        if not 1 <= topk <= 32:
            raise ValueError("topk must be between 1 and 32")
        self.config = {"topk": topk}
        return dict(self.config)

    @staticmethod
    def _read_image(contents):
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a supported image") from exc
        original_size = list(image.size)
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)[None], original_size

    @staticmethod
    def _normalize(feature):
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        return feature / max(float(np.linalg.norm(feature)), 1e-12)

    def _image_feature(self, contents):
        image, size = self._read_image(contents)
        return self._normalize(self.image_model.run(image)[0]), size

    def _text_feature(self, text):
        return self._normalize(self.text_model.run(self.tokenizer.encode(text))[0])

    def predict(self, image, params):
        prompts = params.get("prompts") or ["a photo of a dog", "a photo of a cat"]
        prompts = [str(item).strip() for item in prompts if str(item).strip()]
        if not prompts:
            raise ValueError("Provide at least one candidate label")
        if len(prompts) > 32:
            raise ValueError("Provide no more than 32 candidate labels")
        image_feature, image_size = self._image_feature(image)
        text_features = [self._text_feature(prompt) for prompt in prompts]
        scores = np.asarray(text_features) @ image_feature
        scores = scores * np.exp(4.605170249938965)
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        ranking = np.argsort(probabilities)[::-1]
        topk = min(self.update_config({"topk": params.get("topk", self.config["topk"])})["topk"], len(ranking))
        return {
            "mode": "classification",
            "image": {"width": image_size[0], "height": image_size[1]},
            "candidate_count": len(prompts),
            "predictions": [
                {"text": prompts[index], "score": float(probabilities[index])}
                for index in ranking[:topk]
            ],
        }

    def retrieve(self, images, text, topk=None):
        if not images:
            raise ValueError("Upload at least one image")
        query_feature = self._text_feature(text)
        ranked = []
        for filename, contents in images:
            image_feature, size = self._image_feature(contents)
            ranked.append({
                "filename": filename,
                "similarity": float(np.dot(query_feature, image_feature)),
                "image": {"width": size[0], "height": size[1]},
            })
        ranked.sort(key=lambda item: item["similarity"], reverse=True)
        selected_topk = self.update_config({"topk": topk if topk is not None else self.config["topk"]})["topk"]
        limit = min(selected_topk, len(ranked))
        return {"mode": "retrieval", "query": text, "image_count": len(images), "matches": ranked[:limit]}


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
