#!/usr/bin/env python3
"""Build YOLO26 packages from one loaded ARM64 runtime image."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Lock

from models import ROOT, image_references, load_catalog, prepare


def docker(*args, capture=False):
    return subprocess.run(["docker", *args], check=True, text=True, capture_output=capture)


def layers(image):
    return json.loads(docker("image", "inspect", "--format", "{{json .RootFS.Layers}}",
                             image, capture=True).stdout)


DEMO_CLIPS = {"pose": "exercise", "obb": "harbor"}


def stage_video(spec, context):
    clip = DEMO_CLIPS.get(spec["task"], "street")
    video = context / "video"
    video.mkdir()
    for source, target in ((f"{clip}.mp4", "input.mp4"), (f"{clip}.txt", "ATTRIBUTION.txt")):
        path = video / target
        shutil.copyfile(ROOT / "demo" / source, path)
        path.chmod(0o644)
        os.utime(path, (0, 0))
    os.utime(video, (0, 0))
    return clip


def build(spec, reference, runtime, runtime_layers, revision, publish, video_layers, layer_lock):
    image = f"{reference[0]}:{reference[1]}"
    versioned = f"{image}-{revision}" if revision else None
    with tempfile.TemporaryDirectory(prefix="rk-model-") as directory:
        context = Path(directory)
        prepare(spec, context / "package")
        clip = stage_video(spec, context)
        docker("build", "--builder", "default", "--platform", "linux/arm64",
               "--quiet", "--pull=false", "--build-arg", f"RUNTIME_IMAGE={runtime}",
               "--file", str(ROOT / "Dockerfile.model"), "--tag", image, str(context))

    image_layers = layers(image)
    if image_layers[:-2] != runtime_layers:
        raise RuntimeError(f"{image} must append exactly one video layer and one model layer to {runtime}")
    with layer_lock:
        if video_layers.setdefault(clip, image_layers[-2]) != image_layers[-2]:
            raise RuntimeError(f"{image} did not reuse the {clip} video layer")

    docker("run", "--rm", "--entrypoint", "python", image, "-c",
           "import sys; from rkapi import _native; from rkapi.app import app; "
           "from rkapi.backends.yolo import Backend; "
           "from rkapi.packages import load_package; "
           "assert load_package('/opt/model/package.json')['id'] == sys.argv[1]; "
           "from pathlib import Path; assert Path('/opt/demo/input.mp4').is_file(); "
           "app.openapi()", spec["id"])

    if versioned:
        docker("image", "tag", image, versioned)
    if publish:
        if versioned:
            docker("push", versioned)
        docker("push", image)
    print(f"Verified shared runtime layers for {image}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, help="Local runtime image tag")
    parser.add_argument("--repository", required=True, help="Image namespace, such as ghcr.io/owner/repo")
    parser.add_argument("--model", help="Catalog model ID (default: all models)")
    parser.add_argument("--revision", help="Add a versioned tag for this source revision")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    specs = load_catalog(ROOT / "models", ROOT)
    references = image_references(specs)
    selected = [spec for spec in specs if not args.model or spec["id"] == args.model]
    if not selected:
        parser.error(f"Unknown model ID: {args.model}")
    if not args.repository or args.repository != args.repository.lower():
        parser.error("The image namespace must be lowercase")
    if args.revision and (len(args.revision) != 40 or any(ch not in "0123456789abcdef" for ch in args.revision)):
        parser.error("The revision must be a full lowercase Git SHA")
    runtime_layers = layers(args.runtime)
    if not runtime_layers:
        parser.error("The runtime image has no filesystem layers")

    failures = []
    video_layers = {}
    layer_lock = Lock()
    with ThreadPoolExecutor(max_workers=4) as workers:
        builds = {}
        for spec in selected:
            package, tag = references[spec["id"]]
            future = workers.submit(build, spec, (f"{args.repository}/{package}", tag),
                                    args.runtime, runtime_layers, args.revision, args.publish,
                                    video_layers, layer_lock)
            builds[future] = spec["id"]
        for future in as_completed(builds):
            try:
                future.result()
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                failures.append(builds[future])
                print(f"{builds[future]}: {error}", flush=True)
    for clip, digest in sorted(video_layers.items()):
        print(f"Video layer {clip}: {digest}")
    print(f"Verified {len(selected) - len(failures)} of {len(selected)} model images")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
