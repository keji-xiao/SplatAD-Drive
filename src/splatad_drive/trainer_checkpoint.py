"""Checkpoint/resume behavior for the pinned official trainer, testable on CPU.

This mixin changes training orchestration only. Model losses, Gaussian strategy,
sampling ratios and optimizer schedules remain those of the upstream config.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import torch

from .checkpoint import numpy_scalar_checkpoint_context


def restricted_checkpoint_load(path: Path) -> dict[str, Any]:
    """Keep tensor-only deserialization with the official NumPy scalar allowlist."""
    with numpy_scalar_checkpoint_context():
        state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"step", "pipeline", "optimizers", "schedulers", "scalers"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("resume checkpoint lacks official trainer state fields")
    if isinstance(state["step"], bool) or not isinstance(state["step"], int) or state["step"] < 0:
        raise ValueError("resume checkpoint has an invalid step")
    return state


def atomic_checkpoint_save(state: dict, destination: Path) -> None:
    """Publish only a fully written checkpoint; leave previous checkpoints intact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class ResumableTrainerMixin:
    """Model restoration precedes optimizer creation, state restoration follows it.

    Upstream setup calls ``_load_checkpoint()`` before ``setup_optimizers()``.
    The mixin preserves that order for Gaussian parameter resizing and defers
    optimizer/scheduler state loading until optimizers reference the new tensors.
    ``max_num_iterations`` is a total target, including already completed steps.
    """

    def _load_checkpoint(self):
        self._pending_checkpoint = None
        self._resume_execution_state = None
        config = self.config
        load_path = getattr(config, "load_checkpoint", None)
        load_dir = getattr(config, "load_dir", None)
        if load_path is not None and load_dir is not None:
            raise ValueError("supply only one of load_checkpoint and load_dir")
        if getattr(config, "training_start_step", None) is not None:
            raise ValueError("training_start_step overrides are unsupported for a faithful resume")
        if load_dir is not None:
            folder = Path(load_dir)
            step = getattr(config, "load_step", None)
            if step is None:
                steps = [int(match[1]) for path in folder.glob("step-*.ckpt")
                         if (match := re.fullmatch(r"step-(\d+)\.ckpt", path.name))]
                if not steps:
                    raise ValueError(f"No resumable checkpoint in {folder}")
                step = max(steps)
            load_path = folder / f"step-{step:09d}.ckpt"
        if load_path is None:
            print("No checkpoint requested; training from scratch.", flush=True)
            return
        if not getattr(config, "load_optimizer", True) or not getattr(config, "load_scheduler", True):
            raise ValueError("resume requires optimizer and scheduler state; fine-tuning is a separate experiment")
        load_path = Path(load_path)
        state = restricted_checkpoint_load(load_path)
        if state.get("inference_only", False):
            raise ValueError("inference-only checkpoint cannot be resumed; use the original training checkpoint")
        next_step = state["step"] + 1
        if next_step >= config.max_num_iterations:
            raise ValueError(f"Checkpoint already completed {next_step} iterations; total target must be larger")
        self.pipeline.load_pipeline(state["pipeline"], state["step"])
        self._start_step = next_step
        self.grad_scaler.load_state_dict(state["scalers"])
        # Pipeline tensors are already copied; do not keep a second CPU model.
        self._pending_checkpoint = {"optimizers": state["optimizers"], "schedulers": state["schedulers"]}
        self._resume_execution_state = state.get("splatad_drive")
        self.resume_source = str(load_path.resolve())
        self.resume_source_step = state["step"]
        if hasattr(self, "base_dir"):
            Path(self.base_dir, "resume.json").write_text(json.dumps({
                "checkpoint": self.resume_source, "checkpoint_step": state["step"],
                "next_step": next_step, "total_iteration_target": config.max_num_iterations,
                "execution_state_available": self._resume_execution_state is not None,
                "replay_scope": "optimizer/scheduler continuation; CUDA bitwise determinism is not guaranteed",
            }, indent=2), encoding="utf-8")
        print(f"Loaded model at step {state['step']}; optimizer restoration deferred until parameter registration.", flush=True)
        if self._resume_execution_state is None:
            print("Legacy checkpoint has no RNG/sampler state: optimizer continuation is supported; bitwise replay is not promised.", flush=True)

    def setup_optimizers(self):
        optimizers = super().setup_optimizers()
        pending = getattr(self, "_pending_checkpoint", None)
        if pending is not None:
            optimizers.load_optimizers(pending["optimizers"])
            optimizers.load_schedulers(pending["schedulers"])
            self._pending_checkpoint = None
            print("Restored optimizer and scheduler states against loaded Gaussian parameters.", flush=True)
        return optimizers

    def _capture_execution_state(self):
        numpy_state = np.random.get_state()
        datamanager = self.pipeline.datamanager
        queues = {name: list(getattr(datamanager, name)) for name in ("train_unseen_cameras", "train_unseen_lidars", "eval_unseen_cameras", "eval_unseen_lidars")
                  if hasattr(datamanager, name)}
        return {
            "format_version": 1,
            "total_iteration_target": getattr(self, "_total_iteration_target", self.config.max_num_iterations),
            "python_rng": random.getstate(),
            "numpy_rng": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                          "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler_queues": queues,
        }

    def _restore_execution_state(self):
        execution = getattr(self, "_resume_execution_state", None)
        if execution is None:
            return
        if execution.get("format_version") != 1:
            raise ValueError("unsupported SplatAD-Drive checkpoint execution-state version")
        datamanager = self.pipeline.datamanager
        for name, queue in execution["sampler_queues"].items():
            if name not in ("train_unseen_cameras", "train_unseen_lidars", "eval_unseen_cameras", "eval_unseen_lidars") or not hasattr(datamanager, name):
                raise ValueError(f"Resume sampler queue is incompatible: {name}")
            split = name.split("_", 1)[0]
            dataset_name = split + ("_dataset" if name.endswith("cameras") else "_lidar_dataset")
            dataset = getattr(datamanager, dataset_name, None)
            if dataset is not None and any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(dataset) for index in queue):
                raise ValueError(f"Resume sampler queue is outside current dataset: {name}")
            setattr(datamanager, name, list(queue))
        cuda_states = execution["cuda_rng"]
        if cuda_states:
            if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
                raise ValueError("Resume CUDA RNG device count differs from the checkpoint")
            torch.cuda.set_rng_state_all(cuda_states)
        numpy_state = execution["numpy_rng"]
        np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                             numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
        random.setstate(execution["python_rng"])
        torch.set_rng_state(execution["torch_rng"])
        self._resume_execution_state = None

    def train(self):
        target = self.config.max_num_iterations
        start = getattr(self, "_start_step", 0)
        if target <= start:
            raise ValueError("total iteration target must exceed completed checkpoint iterations")
        self._total_iteration_target = target
        self._restore_execution_state()  # After setup has finished consuming RNG.
        # The pinned Trainer loops range(start, start + max_num_iterations).
        # Config saved by the entrypoint retains the user-facing total target.
        self.config.max_num_iterations = target - start
        try:
            return super().train()
        finally:
            self.config.max_num_iterations = target

    def save_checkpoint(self, step):
        if getattr(self, "local_rank", 0) != 0:
            return
        # Periodic recovery checkpoints must not be skipped by quality trackers.
        # Evaluation quality remains a separate report from recoverability.
        pipeline = self.pipeline.module if hasattr(self.pipeline, "module") else self.pipeline
        state = {
            "step": step, "pipeline": pipeline.state_dict(),
            "optimizers": {key: value.state_dict() for key, value in self.optimizers.optimizers.items()},
            "schedulers": {key: value.state_dict() for key, value in self.optimizers.schedulers.items()},
            "scalers": self.grad_scaler.state_dict(),
            "splatad_drive": self._capture_execution_state(),
        }
        destination = Path(self.checkpoint_dir) / f"step-{step:09d}.ckpt"
        atomic_checkpoint_save(state, destination)
        if getattr(self.config, "save_only_latest_checkpoint", True):
            # Keep a previous recovery point until a subsequent save succeeds.
            checkpoints = sorted((path for path in destination.parent.glob("step-*.ckpt")
                                  if re.fullmatch(r"step-\d+\.ckpt", path.name)),
                                 key=lambda path: int(path.stem.split("-")[1]), reverse=True)
            for previous in checkpoints[2:]:
                previous.unlink()
