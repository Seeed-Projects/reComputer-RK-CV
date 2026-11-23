#!/usr/bin/env python3
"""Build YOLO26 packages from one loaded ARM64 runtime image."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from models import ROOT, image_references, load_catalog, prepare


def docker(*args, capture=False):
    return subprocess.run(["docker", *args], check=True, text=True, capture_output=capture)


def layers(image):
    return json.loads(docker("image", "inspect", "--format", "{{json .RootFS.Layers}}",
                             image, capture=True).stdout)


DEMO_CLIPS = {"pose": "exercise", "obb": "harbor"}


def stage_video(clip, context):
    with tarfile.open(context / "video.tar", "w", format=tarfile.USTAR_FORMAT) as archive:
        for name in ("opt", "opt/demo"):
            directory = tarfile.TarInfo(name)
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            archive.addfile(directory)
        for source, target in ((f"{clip}.mp4", "input.mp4"), (f"{clip}.txt", "ATTRIBUTION.txt")):
            source = ROOT / "demo" / source
            item = tarfile.TarInfo(f"opt/demo/{target}")
            item.mode = 0o644
            item.size = source.stat().st_size
            with source.open("rb") as content:
                archive.addfile(item, content)


def build_video(clip, runtime, runtime_layers, image):
    with tempfile.TemporaryDirectory(prefix="rk-video-") as directory:
        context = Path(directory)
        stage_video(clip, context)
        docker("build", "--builder", "default", "--platform", "linux/arm64",
               "--quiet", "--pull=false", "--build-arg", f"RUNTIME_IMAGE={runtime}",
               "--build-arg", "SOURCE_DATE_EPOCH=0",
               "--file", str(ROOT / "Dockerfile.video"), "--tag", image, str(context))
    image_layers = layers(image)
    if image_layers[:-1] != runtime_layers:
        raise RuntimeError(f"{image} must append exactly one video layer to {runtime}")
    print(f"Video layer {clip}: {image_layers[-1]}", flush=True)
    return image_layers


def build(spec, reference, base, base_layers, revision, publish):
    image = f"{reference[0]}:{reference[1]}"
    versioned = f"{image}-{revision}" if revision else None
    with tempfile.TemporaryDirectory(prefix="rk-model-") as directory:
        context = Path(directory)
        prepare(spec, context / "package")
        docker("build", "--builder", "default", "--platform", "linux/arm64",
               "--quiet", "--pull=false", "--build-arg", f"RUNTIME_IMAGE={base}",
               "--file", str(ROOT / "Dockerfile.model"), "--tag", image, str(context))

    if layers(image)[:-1] != base_layers:
        raise RuntimeError(f"{image} must append exactly one model layer to {base}")

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
    clips = {DEMO_CLIPS.get(spec["task"], "street") for spec in selected}
    bases = {}
    with tempfile.TemporaryDirectory(prefix="rk-video-tag-") as tag_scope:
        try:
            for clip in sorted(clips):
                image = f"rockchip-cv-demo-base:{clip}-{Path(tag_scope).name}"
                bases[clip] = (image, build_video(clip, args.runtime, runtime_layers, image))
            with ThreadPoolExecutor(max_workers=4) as workers:
                builds = {}
                for spec in selected:
                    package, tag = references[spec["id"]]
                    base, base_layers = bases[DEMO_CLIPS.get(spec["task"], "street")]
                    future = workers.submit(build, spec, (f"{args.repository}/{package}", tag),
                                            base, base_layers, args.revision, args.publish)
                    builds[future] = spec["id"]
                for future in as_completed(builds):
                    try:
                        future.result()
                    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                        failures.append(builds[future])
                        print(f"{builds[future]}: {error}", flush=True)
        finally:
            if bases:
                subprocess.run(["docker", "image", "rm", *(base for base, _ in bases.values())],
                               check=False, stdout=subprocess.DEVNULL)
    print(f"Verified {len(selected) - len(failures)} of {len(selected)} model images")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
