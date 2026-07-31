#!/usr/bin/env python3
"""Validate deployable Python model projects without importing their dependencies."""

import argparse
import ast
from pathlib import Path
from typing import Iterable, Set


REQUIRED_ROUTES = {
    "/",
    "/api/health",
    "/api/config",
    "/api/video_feed",
    "/api/video/upload",
    "/api/video/analyze",
    "/api/video/status",
}

REQUIRED_REQUIREMENTS = {
    "numpy",
    "opencv-python-headless",
    "fastapi",
    "uvicorn",
    "python-multipart",
}


def fail(errors: list, message: str) -> None:
    errors.append(message)


def route_paths(source: Path) -> Set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    routes = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                routes.add(decorator.args[0].value)
    return routes


def has_real_asset(paths: Iterable[Path], minimum_size: int = 256) -> bool:
    for path in paths:
        if path.is_file() and path.stat().st_size >= minimum_size:
            with path.open("rb") as asset:
                header = asset.read(64)
            if not header.startswith(b"version https://git-lfs"):
                return True
    return False


def validate(root: Path, project_name: str) -> list:
    errors = []
    project = root / "src" / project_name
    if "_" not in project_name:
        return [f"{project_name}: expected rk<platform>_<model> naming"]

    platform, model_name = project_name.split("_", 1)
    dockerfile = root / "docker" / platform / f"{model_name}.dockerfile"

    required_files = [
        project / "web_classification.py",
        project / "requirements.txt",
        project / "README.md",
        project / "README_zh.md",
        project / ".dockerignore",
        project / "lib" / "librknnrt.so",
        dockerfile,
    ]
    for path in required_files:
        if not path.is_file():
            fail(errors, f"{project_name}: missing {path.relative_to(root)}")

    if errors:
        return errors

    try:
        routes = route_paths(project / "web_classification.py")
    except (SyntaxError, OSError) as exc:
        fail(errors, f"{project_name}: cannot parse application: {exc}")
        routes = set()

    missing_routes = REQUIRED_ROUTES - routes
    if missing_routes:
        fail(errors, f"{project_name}: missing routes {sorted(missing_routes)}")
    if not any(
        route.startswith("/api/models/") and route.endswith("/predict")
        for route in routes
    ):
        fail(errors, f"{project_name}: missing model prediction route")

    requirements = (project / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in REQUIRED_REQUIREMENTS:
        if package not in requirements:
            fail(errors, f"{project_name}: missing requirement {package}")

    asset_checks = [
        ("RKNN model", (project / "model").glob("*.rknn")),
        ("sample video", (project / "video").glob("*.mp4")),
        ("RKNN Lite wheel", (project / "rknn-toolkit-lite2-packages").glob("*.whl")),
    ]
    for label, paths in asset_checks:
        if not has_real_asset(paths):
            fail(errors, f"{project_name}: missing checked-out {label} asset")

    docker_text = dockerfile.read_text(encoding="utf-8")
    for marker in ("EXPOSE 8000", "HEALTHCHECK", "web_classification.py"):
        if marker not in docker_text:
            fail(errors, f"{project_name}: Dockerfile missing {marker!r}")

    workflow_path = root / ".github" / "workflows" / "docker-build.yml"
    if not workflow_path.is_file():
        fail(errors, f"{project_name}: missing Docker build workflow")
    else:
        workflow = workflow_path.read_text(encoding="utf-8")
        if project_name not in workflow:
            fail(errors, f"{project_name}: not present in Docker build workflow")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", nargs="+", help="Names below src/")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    args = parser.parse_args()

    all_errors = []
    for project_name in args.projects:
        errors = validate(args.root.resolve(), project_name)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"PASS {project_name}")

    for error in all_errors:
        print(f"FAIL {error}")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
