"""CPU-only static opacity filter producing a separately marked inference artifact.

All actors are retained. This uses no visibility or GT hit statistics and makes
no claim of output equivalence or acceleration before paired rendering checks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from splatad_drive.trainer_checkpoint import atomic_checkpoint_save, restricted_checkpoint_load


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 0.001:
        raise ValueError("opacity threshold must be finite and in [0, 0.001]")


def make_inference_state(state: dict, threshold: float = 0.001) -> tuple[dict, dict]:
    """Return a separate state; never modify source tensors, mappings or IDs."""
    validate_threshold(threshold)
    if state.get("inference_only", False):
        raise ValueError("source must be an original training checkpoint, not an inference-only derivative")
    pipeline = state.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("checkpoint pipeline must be a state mapping")
    means_keys = [key for key in pipeline if key.endswith("gauss_params.means")]
    if len(means_keys) != 1:
        raise ValueError("require exactly one Gaussian parameter set")
    prefix = means_keys[0][:-len("gauss_params.means")]
    gaussian_prefix = prefix + "gauss_params."
    means = pipeline[means_keys[0]]
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3 or len(means) == 0:
        raise ValueError("Gaussian means must be a nonempty Nx3 tensor")
    count = len(means)
    gaussian_keys = [key for key in pipeline if key.startswith(gaussian_prefix)]
    required = {"means": 3, "scales": 3, "quats": 4, "features_dc": 3,
                "features_rest": None, "opacities": 1, "id": 1}
    for name, width in required.items():
        value = pipeline.get(gaussian_prefix + name)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or (width is not None and value.shape[1] != width):
            raise ValueError(f"Gaussian {name} shape is invalid")
    for key in gaussian_keys:
        value = pipeline[key]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.ndim == 0
                or len(value) != count or not torch.isfinite(value).all()):
            raise ValueError(f"Gaussian parameter is not finite, CPU, or row-aligned: {key}")
    sizes = pipeline.get(prefix + "dynamic_actors.actor_sizes")
    if (not isinstance(sizes, torch.Tensor) or sizes.device.type != "cpu" or sizes.ndim != 2
            or sizes.shape[1] != 3 or not torch.isfinite(sizes).all() or (sizes <= 0).any()):
        raise ValueError("actor_sizes must be a finite positive CPU Ax3 tensor")
    actor_count = len(sizes)
    ids = pipeline[gaussian_prefix + "id"].reshape(-1)
    if ids.dtype == torch.bool or torch.any(ids != ids.round()) or torch.any((ids < 0) | (ids > actor_count)):
        raise ValueError("Gaussian IDs must be integral in [0, actor_count]; static ID equals actor_count")
    logits = pipeline[gaussian_prefix + "opacities"].reshape(-1)
    if not logits.is_floating_point():
        raise ValueError("opacity logits must be floating point")
    probabilities = torch.sigmoid(logits)
    remove = (ids == actor_count) & (probabilities < threshold)
    keep = ~remove
    if not keep.any():
        raise ValueError("filter would remove every Gaussian")
    filtered = copy.copy(pipeline)  # Preserve state_dict metadata and non-Gaussian state.
    shape_report = {}
    for key in gaussian_keys:
        filtered[key] = pipeline[key][keep]
        shape_report[key] = {"before": list(pipeline[key].shape), "after": list(filtered[key].shape),
                             "dtype": str(pipeline[key].dtype)}
    groups = []
    for group_id in range(actor_count + 1):
        selected = ids == group_id
        before, after = int(selected.sum()), int((selected & keep).sum())
        if group_id < actor_count and before != after:
            raise AssertionError("an actor Gaussian was removed")
        groups.append({"id": group_id, "kind": "static" if group_id == actor_count else "actor",
                       "before": before, "after": after, "removed": before - after})
    report = {
        "mode": "static_opacity_filter_inference_only", "threshold": threshold,
        "predicate": "(id == len(dynamic_actors.actor_sizes)) AND (sigmoid(opacities) < threshold)",
        "comparison": "strictly less; sigmoid evaluated in the source tensor dtype on CPU",
        "source_opacity_dtype": str(logits.dtype), "model_prefix": prefix,
        "static_id": actor_count, "actor_count": actor_count,
        "before": count, "after": int(keep.sum()), "removed": int(remove.sum()),
        "groups": groups, "gaussian_parameter_shapes": shape_report,
        "keep_mask_sha256": hashlib.sha256(keep.numpy().tobytes()).hexdigest(),
        "row_order": "original order retained, same mask on every gauss_params tensor",
        "all_actors_retained": True,
        "non_gaussian_pipeline_state": "unchanged, including actor trajectories and decoders",
        "statistics_used": "raw learned opacity only; no visibility or ground-truth hit attribution",
        "quality_status": "UNVERIFIED: requires paired real rendering; no losslessness or speed claim",
    }
    derivative = {"step": state["step"], "pipeline": filtered, "optimizers": {}, "schedulers": {}, "scalers": {},
                  "inference_only": True, "splatad_drive_inference": {"filter": report}}
    return derivative, report


def verify_roundtrip(expected: dict, actual: dict) -> None:
    if actual.get("inference_only") is not True or actual["step"] != expected["step"]:
        raise ValueError("inference checkpoint marker or step did not survive saving")
    if any(actual[key] != {} for key in ("optimizers", "schedulers", "scalers")):
        raise ValueError("inference checkpoint unexpectedly contains training optimizer state")
    if actual["pipeline"].keys() != expected["pipeline"].keys():
        raise ValueError("saved pipeline keys differ")
    for key, value in expected["pipeline"].items():
        restored = actual["pipeline"][key]
        equal = torch.equal(value, restored) if isinstance(value, torch.Tensor) else value == restored
        if not equal:
            raise ValueError(f"saved pipeline state differs: {key}")
    if actual.get("splatad_drive_inference") != expected["splatad_drive_inference"]:
        raise ValueError("inference provenance did not survive saving")


def prune_checkpoint(checkpoint: Path, config: Path, output: Path, threshold: float = 0.001,
                     experiment_name: str = "pruned_baseline_alpha001") -> dict:
    validate_threshold(threshold)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_name):
        raise ValueError("experiment_name must contain only letters, numbers, underscores or hyphens")
    checkpoint, config, output = checkpoint.resolve(), config.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty or new")
    source_sha = sha256_file(checkpoint)
    source_config = config.read_bytes()
    config_text = source_config.decode("utf-8")
    target_config, matches = re.subn(r"(?m)^experiment_name:[^\r\n]*", "experiment_name: " + experiment_name, config_text)
    if matches != 1:
        raise ValueError("source config must contain exactly one top-level experiment_name")
    original = restricted_checkpoint_load(checkpoint)
    derivative, report = make_inference_state(original, threshold)
    step = original["step"]
    del original
    source = {"checkpoint": str(checkpoint), "checkpoint_sha256": source_sha,
              "checkpoint_bytes": checkpoint.stat().st_size, "step": step,
              "config": str(config), "config_sha256": hashlib.sha256(source_config).hexdigest()}
    derivative["splatad_drive_inference"]["source"] = source
    destination = output / "nerfstudio_models" / f"step-{step:09d}.ckpt"
    if destination == checkpoint or output / "config.yml" == config:
        raise ValueError("source checkpoint and config cannot be overwritten")
    atomic_checkpoint_save(derivative, destination)
    reread = restricted_checkpoint_load(destination)
    verify_roundtrip(derivative, reread)
    del reread
    if sha256_file(checkpoint) != source_sha or config.read_bytes() != source_config:
        raise ValueError("source checkpoint or config changed during export")
    (output / "config.yml").write_bytes(target_config.encode("utf-8"))
    manifest = {
        "schema_version": 1, "inference_only": True, "status": "EXPORTED_AND_VERIFIED",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu",
        "source": source, "filter": report,
        "checkpoint": {"path": str(destination), "bytes": destination.stat().st_size,
                       "sha256": sha256_file(destination), "step": step},
        "config": {"path": str(output / "config.yml"), "sha256": sha256_file(output / "config.yml"),
                   "change": "experiment_name only", "experiment_name": experiment_name},
        "checks": {"pipeline_roundtrip_exact": True, "all_actors_retained": True,
                   "source_checkpoint_sha256_unchanged": True, "source_config_bytes_unchanged": True},
        "storage_caveat": "Source includes optimizer/scheduler/scaler state; output omits it. Total checkpoint size reduction is NOT solely attributable to pruning.",
        "training": "Not resumable; ResumableTrainerMixin rejects inference_only before loading model state.",
        "exporter_sha256": sha256_file(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--experiment-name", default="pruned_baseline_alpha001")
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = prune_checkpoint(args.checkpoint, args.config, args.output, args.threshold, args.experiment_name)
    summary = manifest["filter"]
    print(f"CPU inference-only checkpoint: {summary['before']:,} -> {summary['after']:,}; removed {summary['removed']:,} static Gaussians; all actors retained.")
    print(manifest["checkpoint"]["path"])


if __name__ == "__main__":
    main()
