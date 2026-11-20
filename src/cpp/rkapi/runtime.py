"""Load one validated model package and its task adapter per service process."""
import importlib
import os
from pathlib import Path

from .backends.base import BaseBackend
from .packages import load_package


def load_runtime():
    package = load_package(os.getenv("RK_PACKAGE", "/opt/model/package.json"))
    compatible = Path("/proc/device-tree/compatible")
    if compatible.exists() and f"rockchip,{package['chip']}" not in compatible.read_text(errors="replace"):
        raise RuntimeError(f"This image requires {package['chip']}; use the image matching your board")
    module = importlib.import_module(f"rkapi.backends.{package['backend']}")
    if not issubclass(module.Backend, BaseBackend):
        raise TypeError("Model backend must implement BaseBackend")
    backend = module.Backend(package)
    backend.model_info = {key: package[key] for key in
                          ("id", "name", "family", "task", "chip", "backend", "tag", "size", "precision")
                          if key in package}
    backend.aliases = {package["id"], package["family"], *package.get("id_aliases", [])}
    # Retain the original Seeed-compatible URL for existing YOLO26 clients.
    if package["family"] == "yolo26" and package["task"] == "detection":
        backend.aliases.add("yolo11")
    return backend
