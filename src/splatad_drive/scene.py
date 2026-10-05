"""A compact, trainable Gaussian scene graph for the reference backend.

Static Gaussians (actor_id=-1) live in world coordinates. Every other
Gaussian lives in its actor's local frame. A trajectory maps that frame to
world at a requested timestamp. Actor poses clamp outside their keyframe
range; timestamps are seconds, positions metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .geometry import (as_tensor, interpolate_pose, matrix_to_quaternion,
                       normalize_quaternion, quaternion_multiply,
                       transform_points, validate_transform)


@dataclass
class ActorTrajectory:
    timestamps: Tensor
    poses: Tensor

    def __post_init__(self):
        self.poses = as_tensor(self.poses)
        # Keep time in float64: Unix timestamps lose subsecond motion in float32.
        self.timestamps = torch.as_tensor(self.timestamps, device=self.poses.device, dtype=torch.float64)
        if self.timestamps.ndim != 1 or len(self.timestamps) == 0:
            raise ValueError("trajectory timestamps must be a nonempty 1D array")
        if self.poses.shape != (len(self.timestamps), 4, 4):
            raise ValueError("trajectory poses must have shape [keyframes,4,4]")
        if not torch.isfinite(self.timestamps).all() or torch.any(torch.diff(self.timestamps) <= 0):
            raise ValueError("trajectory timestamps must be finite and strictly increasing")
        for pose in self.poses:
            validate_transform(pose, "actor trajectory pose")

    def at(self, timestamp: float | Tensor, *, like: Tensor | None = None) -> Tensor:
        poses = self.poses if like is None else self.poses.to(device=like.device, dtype=like.dtype)
        times = self.timestamps.to(device=poses.device)
        timestamp = torch.as_tensor(timestamp, device=poses.device, dtype=torch.float64)
        if len(times) == 1:
            return poses[0]
        right = int(torch.searchsorted(times, timestamp.detach()).clamp(1, len(times)-1))
        left = right-1
        fraction = ((timestamp-times[left])/(times[right]-times[left])).clamp(0., 1.)
        return interpolate_pose(poses[left], poses[right], fraction)


@dataclass
class SceneState:
    timestamp: float
    actor_poses: Mapping[int, Tensor] = field(default_factory=dict)
    actor_visibility: Mapping[int, bool] = field(default_factory=dict)
    actor_trajectories: Mapping[int, ActorTrajectory] = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.timestamp):
            raise ValueError("scene timestamp must be finite seconds")


def _logit(values: Tensor) -> Tensor:
    return torch.logit(values.clamp(1e-6, 1-1e-6))


class GaussianScene(nn.Module):
    """Physical input parameters; optimization uses positive/bounded transforms.

    ``means``, ``quats`` and ``log_scales``, ``opacity_logits``,
    ``color_logits``, ``intensity_logits`` are trainable parameters. Colors
    are direct RGB, not the official SplatAD learned feature/CNN model.
    Bounded input endpoints are initialized at 1e-6 and 1-1e-6 to retain
    finite trainable logits.
    """

    def __init__(self, means, quats, scales, opacities, colors, actor_ids=None,
                 trajectories: Mapping[int, ActorTrajectory] | None = None, intensities=None):
        super().__init__()
        means = as_tensor(means)
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError("means must have shape [N,3]")
        quats, scales, opacities, colors = [as_tensor(v, like=means) for v in
                                            (quats, scales, opacities, colors)]
        opacities = opacities.reshape(-1)
        size = len(means)
        if quats.shape != (size, 4) or scales.shape != (size, 3) or colors.shape != (size, 3):
            raise ValueError("quats, scales, colors must have shape [N,4], [N,3], [N,3]")
        if opacities.shape != (size,):
            raise ValueError("opacities must have shape [N] or [N,1]")
        if any(not torch.isfinite(v).all() for v in (means, quats, scales, opacities, colors)):
            raise ValueError("Gaussian attributes must be finite")
        if torch.any(scales <= 0):
            raise ValueError("scales must be positive physical standard deviations")
        if torch.any((opacities < 0) | (opacities > 1)) or torch.any((colors < 0) | (colors > 1)):
            raise ValueError("opacities and colors must be in [0,1]")
        intensities = colors.mean(-1) if intensities is None else as_tensor(intensities, like=means).reshape(-1)
        if intensities.shape != (size,) or not torch.isfinite(intensities).all() or torch.any((intensities < 0) | (intensities > 1)):
            raise ValueError("intensities must be finite [N] values in [0,1]")
        if actor_ids is None:
            actor_ids = torch.full((size,), -1, dtype=torch.long, device=means.device)
        else:
            ids_input = torch.as_tensor(actor_ids, device=means.device)
            if ids_input.is_floating_point() and torch.any(ids_input != ids_input.round()):
                raise ValueError("actor_ids must be integers")
            actor_ids = ids_input.to(torch.long)
        if actor_ids.shape != (size,) or torch.any(actor_ids < -1):
            raise ValueError("actor_ids must have shape [N], with -1 for static and nonnegative actor ids")
        self.means = nn.Parameter(means.detach().clone())
        self.quats = nn.Parameter(normalize_quaternion(quats).detach().clone())
        self.log_scales = nn.Parameter(scales.log().detach().clone())
        self.opacity_logits = nn.Parameter(_logit(opacities).detach().clone())
        self.color_logits = nn.Parameter(_logit(colors).detach().clone())
        self.intensity_logits = nn.Parameter(_logit(intensities).detach().clone())
        self.register_buffer("actor_ids", actor_ids.detach().clone())
        self.trajectories = dict(trajectories or {})
        if any(int(key) < 0 for key in self.trajectories):
            raise ValueError("only dynamic actor IDs can have trajectories")

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp()

    @property
    def opacities(self) -> Tensor:
        return self.opacity_logits.sigmoid()

    @property
    def colors(self) -> Tensor:
        return self.color_logits.sigmoid()

    @property
    def intensities(self) -> Tensor:
        return self.intensity_logits.sigmoid()

    def world_gaussians(self, state: SceneState, time_offset: float = 0.) -> dict[str, Tensor]:
        means, quats = self.means, normalize_quaternion(self.quats)
        visible = torch.ones(len(means), dtype=torch.bool, device=means.device)
        # Stack transformed subsets using where to retain gradients to local geometry.
        for actor in torch.unique(self.actor_ids).tolist():
            if actor == -1:
                continue
            selection = self.actor_ids == actor
            if not state.actor_visibility.get(actor, True):
                visible = visible & ~selection
                continue
            if actor in state.actor_poses:
                pose = as_tensor(state.actor_poses[actor], like=means)
                validate_transform(pose, f"actor {actor} pose")
            else:
                trajectory = state.actor_trajectories.get(actor, self.trajectories.get(actor))
                pose = (torch.eye(4, device=means.device, dtype=means.dtype) if trajectory is None
                        else trajectory.at(state.timestamp+time_offset, like=means))
            new_means = transform_points(pose, self.means)
            new_quats = quaternion_multiply(matrix_to_quaternion(pose[:3, :3]), normalize_quaternion(self.quats))
            means = torch.where(selection[:, None], new_means, means)
            quats = torch.where(selection[:, None], new_quats, quats)
        return {"means": means[visible], "quats": quats[visible], "scales": self.scales[visible],
                "opacities": self.opacities[visible], "colors": self.colors[visible],
                "intensities": self.intensities[visible], "actor_ids": self.actor_ids[visible]}

    def export(self) -> dict:
        """Portable physical tensors suitable for JSON/NPZ adapters."""
        result = {name: getattr(self, name).detach().cpu() for name in
                  ("means", "quats", "scales", "opacities", "colors", "intensities", "actor_ids")}
        result["trajectories"] = {key: {"timestamps": item.timestamps.detach().cpu(),
                                         "poses": item.poses.detach().cpu()}
                                    for key, item in self.trajectories.items()}
        return result
