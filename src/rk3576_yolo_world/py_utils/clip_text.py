import re
from functools import lru_cache
from pathlib import Path

import numpy as np


MAX_PROMPTS = 80
SEQUENCE_LENGTH = 20
END_OF_TEXT = 49407


def bytes_to_unicode():
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(161, 173)) + list(range(174, 256))
    characters = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


def parse_prompts(value):
    if isinstance(value, str):
        prompts = [item.strip() for item in value.split("|")]
    elif isinstance(value, (list, tuple)):
        prompts = [str(item).strip() for item in value]
    else:
        raise ValueError("prompts must be a pipe-separated string or a list")

    prompts = [item for item in prompts if item]
    if not prompts:
        raise ValueError("At least one prompt is required")
    if len(prompts) > MAX_PROMPTS:
        raise ValueError(f"At most {MAX_PROMPTS} prompts are supported")
    if any(len(item) > 200 for item in prompts):
        raise ValueError("Each prompt must contain at most 200 characters")
    return tuple(prompts)


class ClipTokenizer:
    def __init__(self, header_path):
        vocab_path = Path(header_path)
        if vocab_path.suffix == ".h":
            header = vocab_path.read_text(encoding="utf-8")
            values = re.findall(r"0x([0-9a-fA-F]{2})", header)
            if not values:
                raise ValueError(
                    f"No CLIP vocabulary bytes found in {header_path}"
                )
            merges_text = bytes(
                int(value, 16) for value in values
            ).decode("utf-8")
        else:
            merges_text = vocab_path.read_text(encoding="utf-8")
        merges = [
            tuple(line.split())
            for line in merges_text.splitlines()[1:]
            if len(line.split()) == 2
        ]
        byte_values = bytes_to_unicode()
        vocab = list(byte_values.values())
        vocab += [value + "</w>" for value in byte_values.values()]
        vocab += [first + second for first, second in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {token: index for index, token in enumerate(vocab)}
        self.ranks = {pair: index for index, pair in enumerate(merges)}
        self.byte_encoder = byte_values
        self.pattern = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
            r"[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]+",
            re.I,
        )

    @lru_cache(maxsize=4096)
    def bpe(self, token):
        if not token:
            return ()
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {
                (word[index], word[index + 1])
                for index in range(len(word) - 1)
            }
            pair = min(
                pairs,
                key=lambda item: self.ranks.get(item, float("inf")),
            )
            if pair not in self.ranks:
                break
            merged = []
            index = 0
            while index < len(word):
                if (
                    index + 1 < len(word)
                    and word[index] == pair[0]
                    and word[index + 1] == pair[1]
                ):
                    merged.append(pair[0] + pair[1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
        return word

    def encode(self, text, length=SEQUENCE_LENGTH):
        text = " ".join(text.strip().lower().split())
        tokens = [49406]
        for match in self.pattern.findall(text):
            encoded = "".join(
                self.byte_encoder[value] for value in match.encode("utf-8")
            )
            tokens.extend(self.encoder[piece] for piece in self.bpe(encoded))
        tokens = tokens[: length - 1] + [END_OF_TEXT]
        tokens += [END_OF_TEXT] * (length - len(tokens))
        return np.asarray(tokens, dtype=np.int64)[None]
