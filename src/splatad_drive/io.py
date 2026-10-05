"""Small, strict artifact exporters with no implicit coordinate transforms."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


def to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def to_jsonable(value: Any) -> Any:
    """Convert tensor/dataclass trees; nonfinite numbers become JSON null.

    JSON has no Infinity/NaN. Retain metric status fields when exporting e.g.
    perfect reconstruction (infinite PSNR) or unavailable measurements.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return to_jsonable(to_numpy(value).tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def save_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    payload = json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload + "\n", encoding="utf-8")
    return destination


def _colors(values: Any, name: str) -> np.ndarray:
    array = to_numpy(values)
    if array.dtype == np.uint8:
        return array
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must be uint8 [0, 255] or finite floating point [0, 1]")
    return np.rint(array * 255).astype(np.uint8)


def save_png(path: str | Path, image: Any) -> Path:
    array = _colors(image, "image")
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (3, 4)) or min(array.shape[:2]) == 0:
        raise ValueError("image must be nonempty HW grayscale, HWC RGB, or HWC RGBA")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(destination, format="PNG")
    return destination


def save_ply(path: str | Path, points: Any, colors: Any = None) -> Path:
    """Export ASCII XYZ (metres) and optional RGB; permits empty point clouds."""
    xyz = to_numpy(points)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points must be a finite Nx3 array")
    rgb = _colors(colors, "colors") if colors is not None else None
    if rgb is not None and rgb.shape != xyz.shape:
        raise ValueError("colors must have the same Nx3 shape as points")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\ncomment coordinates in metres\nelement vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        if rgb is None:
            np.savetxt(stream, xyz, fmt="%.9g %.9g %.9g")
        else:
            np.savetxt(stream, np.column_stack((xyz, rgb)), fmt="%.9g %.9g %.9g %d %d %d")
    return destination
