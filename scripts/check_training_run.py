"""Inspect one official run; filenames alone are not success evidence."""
import argparse
import json
import math
from pathlib import Path
import re
import sys


REQUIRED_LOSSES = ("main_loss", "depth_loss", "intensity_loss", "ray_drop_loss")


def checkpoint_evidence(path):
    """Read tensor-only checkpoint contents without unrestricted pickle loading."""
    import torch

    from splatad_drive.checkpoint import numpy_scalar_checkpoint_context
    with numpy_scalar_checkpoint_context():
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not {"step", "pipeline", "optimizers", "schedulers", "scalers"} <= set(state):
        raise ValueError("checkpoint lacks official trainer state fields")
    step = state["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step is invalid")
    filename_match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
    if filename_match is None or int(filename_match[1]) != step:
        raise ValueError("checkpoint filename and stored step disagree")
    pipeline = state["pipeline"]
    if not isinstance(pipeline, dict) or not pipeline:
        raise ValueError("checkpoint pipeline is empty")
    tensor_count = 0

    def check_finite(value, prefix):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise ValueError(f"nonfinite checkpoint tensor: {prefix}")
        elif isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, f"{prefix}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check_finite(item, f"{prefix}.{index}")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"nonfinite checkpoint scalar: {prefix}")

    check_finite(state, "checkpoint")
    means = [tensor for key, tensor in pipeline.items() if key.endswith("gauss_params.means")]
    if len(means) != 1 or not isinstance(means[0], torch.Tensor) or means[0].ndim != 2 or means[0].shape[1] != 3 or means[0].shape[0] < 1:
        raise ValueError("checkpoint has no unique nonempty SplatAD Gaussian means tensor")
    prefix = next(key[:-len("means")] for key in pipeline if key.endswith("gauss_params.means"))
    for name, width in (("scales", 3), ("quats", 4), ("features_dc", 3), ("features_rest", None), ("opacities", 1), ("id", 1)):
        tensor = pipeline.get(prefix + name)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != means[0].shape[0] or (width is not None and tensor.shape[1] != width):
            raise ValueError(f"checkpoint Gaussian {name} shape is inconsistent")
    if not isinstance(state["optimizers"], dict) or not state["optimizers"]:
        raise ValueError("checkpoint lacks optimizer state")
    for name, optimizer in state["optimizers"].items():
        if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]:
            raise ValueError(f"invalid optimizer state: {name}")
    return {"path": str(path), "stored_step": step, "completed_iterations": step + 1,
            "gaussian_count": means[0].shape[0], "finite_tensors_checked": tensor_count}


def inspect_run(directory, expected_iterations=None):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    directory = Path(directory).resolve()
    summary = {"directory": str(directory), "checkpoints": [], "scalars": {}, "blockers": []}
    launches = sorted(directory.rglob("launch.json")) if not (directory / "launch.json").exists() else [directory / "launch.json"]
    if len(launches) > 1:
        summary["blockers"].append("Multiple launches found; inspect one exact run directory")
        summary["run_directories"] = [str(path.parent) for path in launches]
        return _finish(summary)
    if launches:
        directory = launches[0].parent
        summary["directory"] = str(directory)
        launch = json.loads(launches[0].read_text(encoding="utf-8"))
        configured = launch.get("expected_iterations", launch.get("config", {}).get("iterations"))
        if expected_iterations is not None and configured != expected_iterations:
            summary["blockers"].append("Expected iterations disagree with launch manifest")
        expected_iterations = configured if expected_iterations is None else expected_iterations
        completion_path = directory / "completion.json"
        if not completion_path.exists():
            summary["blockers"].append("Run has no completion record (still running, interrupted, or legacy run)")
        else:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            summary["completion"] = completion
            if completion.get("returncode") != 0 or completion.get("status") != "exited_zero":
                summary["blockers"].append("Trainer did not exit successfully")
    if isinstance(expected_iterations, bool) or not isinstance(expected_iterations, int) or expected_iterations <= 0:
        summary["blockers"].append("Expected iteration budget unknown; provide launch.json or --expected-iterations")
        expected_iterations = None
    summary["expected_iterations"] = expected_iterations
    checkpoints = sorted(directory.rglob("step-*.ckpt"))
    if len({path.parent for path in checkpoints}) > 1:
        summary["blockers"].append("Checkpoints from multiple run directories; select one run")
        return _finish(summary)
    if not checkpoints:
        summary["blockers"].append("No official checkpoint found")
    for path in checkpoints:
        try:
            summary["checkpoints"].append(checkpoint_evidence(path))
        except Exception as exc:
            summary["blockers"].append(f"Invalid checkpoint {path.name}: {type(exc).__name__}: {exc}")
    if expected_iterations and summary["checkpoints"]:
        final_step = max(item["stored_step"] for item in summary["checkpoints"])
        if final_step != expected_iterations - 1:
            summary["blockers"].append(f"Checkpoint ends at step {final_step}; expected {expected_iterations - 1} (zero-based)")
    events = sorted(directory.rglob("events.out.tfevents.*"))
    if not events:
        summary["blockers"].append("No TensorBoard event file found")
    observed_tags = set()
    total_loss_last_step = -1
    for event in events:
        try:
            accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                values = accumulator.Scalars(tag)
                if not values:
                    continue
                observed_tags.add(tag)
                finite = all(math.isfinite(value.value) for value in values)
                steps = [value.step for value in values]
                summary["scalars"][f"{event.relative_to(directory).as_posix()}/{tag}"] = {
                    "finite": finite, "count": len(values), "first_step": min(steps), "last_step": max(steps),
                    "first_value": values[0].value if math.isfinite(values[0].value) else None,
                    "last_value": values[-1].value if math.isfinite(values[-1].value) else None}
                if tag == "Train Loss":
                    total_loss_last_step = max(total_loss_last_step, max(steps))
                if not finite:
                    summary["blockers"].append(f"Nonfinite scalar: {tag}")
                if tag.endswith("gaussian_count") and any(value.value <= 0 for value in values):
                    summary["blockers"].append("Nonpositive Gaussian count recorded")
        except Exception as exc:
            summary["blockers"].append(f"Unreadable TensorBoard event: {type(exc).__name__}: {exc}")
    for loss in REQUIRED_LOSSES:
        if f"Train Loss Dict/{loss}" not in observed_tags:
            summary["blockers"].append(f"Missing training loss evidence: {loss}")
    if "Train Loss" not in observed_tags:
        summary["blockers"].append("Missing total Train Loss evidence")
    if expected_iterations and total_loss_last_step < ((expected_iterations - 1) // 10) * 10:
        summary["blockers"].append("Total loss logging does not reach the final scheduled logging step (default interval 10)")
    if not any(tag.endswith("gaussian_count") for tag in observed_tags):
        summary["blockers"].append("Missing Gaussian count evidence")
    return _finish(summary)


def _finish(summary):
    summary["status"] = "PASS" if not summary["blockers"] else "NO-GO"
    summary["scope"] = "Checkpoint tensor integrity, configured iteration completion and recorded joint-loss finiteness; not image quality, all unlogged iterations, held-out evaluation or paper reproduction"
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output", default="outputs/training_check.json")
    parser.add_argument("--expected-iterations", type=int, default=None)
    args = parser.parse_args()
    try:
        result = inspect_run(args.run, args.expected_iterations)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "PASS" else 2)
    except ImportError as exc:
        parser.exit(2, f"TensorBoard and PyTorch are required to inspect a training run: {exc}\n")
