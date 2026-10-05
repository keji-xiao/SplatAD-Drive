"""Export official checkpoint Gaussian centres on CPU, without building a model.

The PLY RGB fields are clipped learned features_dc, not decoded sensor RGB.
These are centre point clouds, not standalone renderable SplatAD checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from splatad_drive.trainer_checkpoint import restricted_checkpoint_load


VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_gaussians(state: dict) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, str]:
    """Require one Gaussian set and its matching actor table; preserve row order."""
    pipeline = state.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("checkpoint pipeline must be a tensor mapping")
    keys = [key for key in pipeline if key.endswith("gauss_params.means")]
    if len(keys) != 1:
        raise ValueError("checkpoint must contain one unique Gaussian means tensor")
    prefix = keys[0][:-len("gauss_params.means")]

    def array(name: str, width: int | None = None) -> np.ndarray:
        value = pipeline.get(prefix + name)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if value.ndim != 2 or (width is not None and value.shape[1] != width):
            raise ValueError(f"{name} has an invalid shape")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point values")
        return value.detach().numpy()

    widths = {"means": 3, "features_dc": 3, "features_rest": None,
              "scales": 3, "quats": 4, "opacities": 1, "id": 1}
    gaussians = {name: array("gauss_params." + name, width) for name, width in widths.items()}
    count = len(gaussians["means"])
    if count == 0 or any(len(value) != count for value in gaussians.values()):
        raise ValueError("Gaussian parameters must have the same nonzero row count")
    sizes = array("dynamic_actors.actor_sizes", 3)
    if np.any(sizes <= 0):
        raise ValueError("actor sizes must be positive")
    ids = gaussians["id"].ravel()
    if np.any(ids != np.rint(ids)) or np.any((ids < 0) | (ids > len(sizes))):
        raise ValueError("Gaussian IDs must be integral in [0, number_of_actors]")
    padding_tensor = pipeline.get(prefix + "dynamic_actors.actor_padding")
    if padding_tensor is None:
        padding = np.zeros(3, dtype=np.float32)
    else:
        if (not isinstance(padding_tensor, torch.Tensor) or padding_tensor.device.type != "cpu"
                or padding_tensor.shape != (3,) or not torch.isfinite(padding_tensor).all()
                or (padding_tensor < 0).any()):
            raise ValueError("actor padding must be a finite nonnegative CPU vector of length 3")
        padding = padding_tensor.detach().numpy()
    return gaussians, sizes, padding, prefix


def write_ply(path: Path, points: np.ndarray, features: np.ndarray, frame: str) -> dict:
    """Write and reread packed little-endian XYZ/RGB; never filter coordinates."""
    if points.shape != features.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and features must both be Nx3")
    if not np.isfinite(points).all() or not np.isfinite(features).all():
        raise ValueError("cannot export nonfinite coordinates or preview features")
    vertices = np.empty(len(points), dtype=VERTEX_DTYPE)
    colors = np.rint(np.clip(features, 0, 1) * 255).astype(np.uint8)
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment coordinates metres; frame {frame}\n"
        "comment centre point cloud; no Gaussian covariance or decoder\n"
        "comment colors clamp(features_dc_first_3,0,1)*255; NOT decoded RGB\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(header)
            vertices.tofile(stream)
        reread = np.memmap(temporary, dtype=VERTEX_DTYPE, mode="r", offset=len(header), shape=(len(points),)) if len(points) else np.empty(0, dtype=VERTEX_DTYPE)
        if temporary.stat().st_size != len(header) + vertices.nbytes or not np.array_equal(reread, vertices):
            raise ValueError("PLY payload roundtrip failed")
        del reread
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": path.name, "format": "binary_little_endian_xyz_rgb", "vertices": len(points),
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "payload_roundtrip": "PASS"}


def group_geometry(points: np.ndarray, features: np.ndarray, logits: np.ndarray,
                   size: np.ndarray | None = None, padding: np.ndarray | None = None) -> dict:
    result = {"count": len(points), "finite_coordinates": bool(np.isfinite(points).all()),
              "finite_preview_features": bool(np.isfinite(features).all())}
    if not len(points):
        return result | {"aabb_metres": None, "radius_from_origin_metres": None}
    # Float64 avoids squaring large finite float32 coordinates into infinity.
    xyz = points.astype(np.float64, copy=False)
    radii = np.linalg.norm(xyz, axis=1)
    quantiles = [0.5, 0.95, 0.99, 0.999, 1.0]
    opacity = torch.sigmoid(torch.from_numpy(logits)).numpy().ravel()
    result.update({
        "aabb_metres": {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist()},
        "coordinate_quantiles_metres": {str(q): row.tolist() for q, row in zip(
            (0.01, 0.5, 0.99), np.quantile(xyz, [0.01, 0.5, 0.99], axis=0))},
        "radius_from_origin_metres": {str(q): float(v) for q, v in zip(quantiles, np.quantile(radii, quantiles))},
        "opacity_probability_quantiles": {str(q): float(v) for q, v in zip(
            (0.0, 0.5, 0.95, 1.0), np.quantile(opacity, [0, 0.5, 0.95, 1]))},
        "preview_feature_raw_min": features.min(0).tolist(),
        "preview_feature_raw_max": features.max(0).tolist(),
        "preview_color_clipped_vertex_count": int(((features < 0) | (features > 1)).any(1).sum()),
    })
    if size is not None:
        outside = (np.abs(xyz) > size / 2).any(1)
        outside_padded = (np.abs(xyz) > size / 2 + padding).any(1)
        opacity_total = float(opacity.sum(dtype=np.float64))
        result.update({
            "actor_size_wlh_metres": size.tolist(),
            "actor_padding_each_side_metres": padding.tolist(),
            "outside_nominal_actor_box_count": int(outside.sum()),
            "outside_nominal_actor_box_fraction": float(outside.mean()),
            "outside_padded_actor_box_count": int(outside_padded.sum()),
            "outside_padded_actor_box_fraction": float(outside_padded.mean()),
            "outside_padded_box_with_opacity_ge_0_1_count": int((outside_padded & (opacity >= 0.1)).sum()),
            "outside_padded_box_opacity_mass_fraction": float(opacity[outside_padded].sum(dtype=np.float64) / opacity_total) if opacity_total else None,
            "box_test_definition": "any(abs(local_xyz) > actor_sizes/2 [+ actor_padding]); centres only",
        })
    return result


def export_checkpoint(checkpoint: Path, output: Path) -> dict:
    checkpoint, output = checkpoint.resolve(), output.resolve()
    state = restricted_checkpoint_load(checkpoint)
    gaussians, sizes, padding, prefix = extract_gaussians(state)
    step = state["step"]
    # Release optimizer state before allocating point-cloud payloads.
    del state
    actor_count = len(sizes)
    ids = gaussians["id"].ravel()
    files = [output / "static_gaussians.ply"] + [output / f"actor_{i:03d}_local.ply" for i in range(actor_count)]
    if any(path.exists() for path in files + [output / "manifest.json"]):
        raise FileExistsError("export files already exist; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "status": "EXPORTED_AND_VERIFIED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                       "sha256": sha256_file(checkpoint), "step": step, "model_prefix": prefix},
        "device": "cpu", "loader": "restricted_checkpoint_load(map_location=cpu, weights_only=True)",
        "exporter_sha256": sha256_file(Path(__file__)),
        "total_gaussians": len(ids), "actor_count": actor_count, "static_id": actor_count,
        "finite_gaussian_parameter_tensors_checked": len(gaussians),
        "coordinates": {"static": "checkpoint model/world coordinates in metres; no inverse dataparser transform applied",
                        "actors": "actor-local box coordinates in metres; no trajectory/world transform applied"},
        "color": "round(clamp(features_dc[:, :3], 0, 1)*255); learned feature preview, NOT decoder RGB, NOT SH conversion",
        "scope": "All Gaussian centres exported without pruning; PLY omits covariance/opacity/features/decoder and cannot reproduce SplatAD renders.",
        "groups": [],
    }
    for group_id, destination in [(actor_count, files[0])] + list(enumerate(files[1:])):
        selection = ids == group_id
        points = gaussians["means"][selection]
        features = gaussians["features_dc"][selection]
        static = group_id == actor_count
        frame = "model_world" if static else f"actor_{group_id:03d}_local"
        group = {"id": group_id, "kind": "static" if static else "actor", "frame": frame}
        group.update(group_geometry(points, features, gaussians["opacities"][selection],
                                    None if static else sizes[group_id], None if static else padding))
        group["ply"] = write_ply(destination, points, features, frame)
        manifest["groups"].append(group)
        print(f"Exported {destination.name}: {len(points):,} centres (CPU)", flush=True)
    if sum(group["count"] for group in manifest["groups"]) != manifest["total_gaussians"]:
        raise AssertionError("export groups do not partition all Gaussian rows")
    manifest["partition_count_check"] = "PASS"
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
