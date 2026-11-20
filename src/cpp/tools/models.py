#!/usr/bin/env python3
"""Validate the catalog, generate CI matrices, and prepare model image assets."""

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rkapi.packages import IDENTIFIER, load_catalog, sha256


def image_references(models):
    references = {}
    used = set()
    for spec in models:
        name = spec.get("name")
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name.lower()):
            raise ValueError(f"{spec['id']}: invalid image package name: {name!r}")
        package = name.lower()
        tag = f"{spec['chip']}-{spec['precision']}"
        if not IDENTIFIER.fullmatch(tag):
            raise ValueError(f"{spec['id']}: invalid image tag: {tag!r}")
        reference = f"{package}:{tag}"
        if reference in used:
            raise ValueError(f"Duplicate image reference: {reference}")
        used.add(reference)
        references[spec["id"]] = (package, tag)
    return references


def download(url, target, checksum):
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            if not response.url.startswith("https://"):
                raise ValueError(f"Download redirected to an insecure URL: {url}")
            with target.open("wb") as stream:
                shutil.copyfileobj(response, stream)
    except OSError as error:
        raise ValueError(f"Could not download {url}: {error}") from error
    if sha256(target) != checksum:
        raise ValueError(f"Checksum mismatch: {url}")


def prepare_remote_metadata(spec, staged, package):
    import yaml

    metadata = staged / "metadata.yaml"
    download(spec["metadata_url"], metadata, spec["metadata_sha256"])
    info = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    expected = {"task": "detect" if spec["task"] == "detection" else spec["task"],
                "target_platform": spec["chip"]}
    if "precision" in info:
        expected["precision"] = spec["precision"]
    expected.update({key: spec[key] for key in ("end2end", "coordinates", "preprocess",
                                               "dense_resize", "kpt_shape") if key in spec})
    defaults = {"dense_resize": "none", "coordinates": "pixels", "preprocess": "letterbox",
                "end2end": False}
    for key, value in expected.items():
        if info.get(key, defaults.get(key)) != value:
            raise ValueError(f"Model metadata does not match catalog field {key}")
    names = info["names"]
    if isinstance(names, dict):
        names = {int(index): label for index, label in names.items()}
        names = [names[index] for index in range(len(names))]
    if len(names) != spec["classes"] or any(not isinstance(label, str) or not label.strip()
                                           or "\n" in label or "\r" in label for label in names):
        raise ValueError("Metadata labels do not match the class contract")
    (staged / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    package.update(metadata=metadata.name, labels="classes.txt")


def prepare(spec, output):
    output = output.resolve()
    marker = ".rockchip-model-package"
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("Package output must not replace the repository or its parent")
    if output.exists() and any(output.iterdir()) and not (output / marker).is_file():
        raise ValueError("Refusing to replace a directory not created by this tool")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "package"
        staged.mkdir()
        model = staged / "model.rknn"
        download(spec["url"], model, spec["sha256"])
        package = {key: value for key, value in spec.items()
                   if key not in ("url", "metadata_url")}
        package.update({"schema_version": 1, "file": "model.rknn"})
        prepare_remote_metadata(spec, staged, package)
        (staged / "package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
        (staged / marker).touch()
        if output.exists():
            output.rmdir() if not any(output.iterdir()) else shutil.rmtree(output)
        shutil.move(str(staged), str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "list", "matrix", "prepare"))
    parser.add_argument("--model", help="Catalog model ID; omit for the complete CI matrix")
    parser.add_argument("--output", type=Path, default=ROOT / ".model-package")
    args = parser.parse_args()
    models = load_catalog(ROOT / "models", ROOT)
    images = image_references(models)
    selected = [spec for spec in models
                if not args.model or args.model in (spec["id"], *spec.get("id_aliases", []))]
    if not selected:
        raise ValueError(f"Unknown model ID: {args.model}")
    if args.command == "matrix":
        matrix = []
        for spec in selected:
            package, tag = images[spec["id"]]
            matrix.append({"id": spec["id"], "package": package, "tag": tag,
                           "legacy_tag": spec["tag"], "aliases": spec.get("aliases", [])})
        print(json.dumps({"include": matrix}))
    elif args.command == "list":
        for spec in selected:
            package, tag = images[spec["id"]]
            print(f"{spec['id']:24} {f'{package}:{tag}':28} {spec['task']:16} {spec['backend']}")
    elif args.command == "validate":
        print(f"Validated {len(models)} model packages")
    else:
        if not args.model:
            raise ValueError("prepare requires --model")
        prepare(selected[0], args.output)
        print(f"Prepared {selected[0]['id']} in {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
