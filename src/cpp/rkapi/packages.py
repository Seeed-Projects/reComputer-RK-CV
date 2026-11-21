"""Model catalog and packaged-model validation; usable without inference libraries."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

IDENTIFIER = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
HF_REVISION = re.compile(r"/[^/]+/[^/]+/resolve/[0-9a-f]{40}/.+")


def checked_path(root, value):
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path must stay inside {root}: {value}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_spec(spec):
    for key in ("id", "family", "chip", "task", "backend", "tag"):
        if not isinstance(spec.get(key), str) or not IDENTIFIER.fullmatch(spec[key]):
            raise ValueError(f"Invalid model {key}: {spec.get(key)!r}")
    if spec["chip"] not in ("rk3576", "rk3588"):
        raise ValueError(f"Unsupported chip: {spec['chip']}")
    if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")):
        raise ValueError(f"{spec['id']}: a SHA-256 checksum is required")
    if not isinstance(spec.get("nms", True), bool):
        raise ValueError(f"{spec['id']}: nms must be a boolean")
    for key in ("aliases", "id_aliases"):
        aliases = spec.get(key, [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) or
                                              not IDENTIFIER.fullmatch(alias) for alias in aliases):
            raise ValueError(f"{spec['id']}: {key} must be a list of identifiers")


def load_catalog(directory, root):
    directory = Path(directory)
    models = []
    for manifest in sorted(directory.glob("*/*/*/*/*/model.json")):
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        family, task, size, chip, precision = manifest.relative_to(directory).parts[:-1]
        expected = {"family": family, "task": "detection" if task == "detect" else task,
                    "size": size, "chip": chip, "precision": precision}
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Model manifest fields do not match its path: {manifest}")
        models.append(spec)
    if not models:
        raise ValueError("The model catalog is empty")
    ids, tags = set(), set()
    for spec in models:
        validate_spec(spec)
        model_ids = [spec["id"], *spec.get("id_aliases", [])]
        model_tags = [spec["tag"], *spec.get("aliases", [])]
        if (len(set(model_ids)) != len(model_ids) or ids.intersection(model_ids) or
                len(set(model_tags)) != len(model_tags) or tags.intersection(model_tags)):
            raise ValueError("Model IDs and image tags must be unique")
        ids.update(model_ids)
        tags.update(model_tags)
        for key in ("url", "metadata_url"):
            value = spec.get(key)
            url = urlsplit(value if isinstance(value, str) else "")
            if (url.scheme != "https" or url.netloc != "huggingface.co" or
                    not HF_REVISION.fullmatch(url.path) or url.query or url.fragment):
                raise ValueError(f"{spec['id']}: {key} must pin a Hugging Face commit")
        if not re.fullmatch(r"[0-9a-f]{64}", spec.get("metadata_sha256", "")):
            raise ValueError(f"{spec['id']}: metadata SHA-256 is required")
        backend = checked_path(root, f"rkapi/backends/{spec['backend']}.py")
        if not backend.is_file():
            raise ValueError(f"{spec['id']}: backend module missing: {spec['backend']}")
    return models


def load_package(path):
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported model package schema")
    validate_spec(spec)
    for key in ("file", "labels", "metadata"):
        if spec.get(key):
            resolved = checked_path(path.parent, spec[key])
            if not resolved.is_file():
                raise ValueError(f"Packaged {key} is missing: {resolved}")
            spec[key] = str(resolved)
    if not spec.get("file") or sha256(spec["file"]) != spec["sha256"]:
        raise ValueError("Packaged model checksum mismatch")
    return spec
