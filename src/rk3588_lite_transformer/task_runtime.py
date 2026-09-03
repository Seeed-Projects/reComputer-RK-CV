from functools import lru_cache
from pathlib import Path

import numpy as np

from rknn_runtime import RKNNModel


class BPETokenizer:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.ranks = {}
        for line in (model_dir / "bpe_order.txt").read_text().splitlines():
            left, right, score = line.rsplit(" ", 2)
            self.ranks[(left, right)] = int(score)
        self.tokens = {}
        self.reverse = {}
        for line in (model_dir / "dict_order.txt").read_text().splitlines():
            word, score = line.rsplit(" ", 1)
            token = int(score) + 4
            self.tokens[word] = token
            self.reverse[token] = word
        self.common = {}
        for line in (model_dir / "cw_token_map_order.txt").read_text().splitlines():
            values = line.split()
            if len(values) < 3 or not values[1].isdigit():
                continue
            count = int(values[1])
            self.common[values[0]] = [int(value) for value in values[2:2 + count]]

    @lru_cache(maxsize=4096)
    def encode_word(self, word):
        if word in self.common:
            return tuple(self.common[word])
        pieces = list(word)
        if not pieces:
            return ()
        pieces[-1] += "</w>"
        while len(pieces) > 1:
            candidates = [((pieces[index], pieces[index + 1]), index) for index in range(len(pieces) - 1)]
            pair, index = min(candidates, key=lambda item: self.ranks.get(item[0], 1 << 30))
            if pair not in self.ranks:
                break
            pieces[index:index + 2] = [pair[0] + pair[1]]
        if pieces[-1] == "</w>":
            pieces.pop()
        elif pieces[-1].endswith("</w>"):
            pieces[-1] = pieces[-1][:-4]
        pieces = [piece + "@@" if index < len(pieces) - 1 else piece for index, piece in enumerate(pieces)]
        return tuple(self.tokens.get(piece, 3) for piece in pieces)

    def encode(self, text):
        result = []
        for word in text.strip().split():
            result.extend(self.encode_word(word))
        return result

    def decode(self, ids):
        return "".join(self.reverse.get(token, "") for token in ids).replace("@@", "")


class Runtime:
    name = "lite_transformer"
    input_kind = "text"

    def __init__(self, platform, model_dir):
        self.platform = platform
        self.model_dir = Path(model_dir)
        self.encoder = RKNNModel(self.model_dir / "encoder.rknn", platform, core_index=0)
        self.decoder = RKNNModel(self.model_dir / "decoder.rknn", platform, core_index=1)
        self.model_names = ["encoder.rknn", "decoder.rknn"]
        self.tokenizer = BPETokenizer(self.model_dir)
        self.token_embedding = np.memmap(self.model_dir / "token_embed.bin", dtype=np.float32, mode="r").reshape(-1, 256)
        self.position_embedding = np.memmap(self.model_dir / "position_embed.bin", dtype=np.float32, mode="r").reshape(-1, 256)

    def describe(self):
        return {
            "task": "English-to-Chinese translation",
            "source_language": "English",
            "target_language": "Chinese",
            "max_source_tokens": 15,
        }

    def get_config(self):
        return {}

    def update_config(self, values):
        if values:
            raise ValueError("Lite Transformer has no runtime settings")
        return {}

    def embed(self, tokens):
        positions = []
        position = 1
        for token in tokens:
            position = position + 1 if token != 1 else 1
            positions.append(position)
        return (self.token_embedding[np.asarray(tokens)] * 16 + self.position_embedding[positions]).astype(np.float32)

    def predict(self, text, params):
        source = self.tokenizer.encode(text)
        if not source:
            raise ValueError("Enter an English sentence")
        if len(source) > 15:
            raise ValueError(f"The input contains {len(source)} BPE tokens; the model accepts at most 15")
        aligned = [1] * 16
        aligned[-1] = 2
        aligned[15 - len(source):15] = source
        encoder_mask = np.asarray([1 if token == 1 else 0 for token in aligned], dtype=np.float32)
        expanded_mask = np.tile(encoder_mask, (16, 1))[None]
        encoder_embedding = self.embed(aligned)[None]
        encoder_output = self.encoder.run([encoder_embedding, expanded_mask])[0]

        output_tokens = [2] + [1] * 15
        caches = [np.zeros((1, 15, 64, 4), dtype=np.float32) for _ in range(6)]
        for iteration in range(15):
            decoder_embedding = self.embed(output_tokens[:iteration + 1])[-1:][None]
            decoder_mask = np.asarray([1 if index < 15 - iteration else 0 for index in range(16)], dtype=np.float32)[None]
            outputs = self.decoder.run([decoder_embedding, encoder_output, encoder_mask[None], decoder_mask] + caches)
            next_token = int(np.argmax(np.asarray(outputs[0]).reshape(-1)))
            output_tokens[iteration + 1] = next_token
            caches = [np.transpose(np.asarray(value), (0, 2, 3, 1))[:, 1:, :, :] for value in outputs[1:7]]
            if next_token == 2:
                break
        generated = [token for token in output_tokens[1:] if token not in (1, 2)]
        return {
            "input_text": text,
            "translation": self.tokenizer.decode(generated),
            "input_tokens": source,
            "output_tokens": generated,
            "input_token_count": len(source),
            "output_token_count": len(generated),
        }


def create_runtime(platform, model_dir):
    return Runtime(platform, model_dir)
