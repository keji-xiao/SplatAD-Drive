"""Differentiable spherical Gaussian projection on explicit LiDAR beams.

This reference renderer supports nonuniform elevation/azimuth samples and
wraps angular differences at the +/-pi seam. Return range is the conditional
expected radial distance under alpha compositing, not first-hit range.
Intensity uses a per-Gaussian scalar; ray_drop=1-hit_probability is a
geometric proxy, not SplatAD's learned intensity/ray-drop MLP.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .geometry import (as_tensor, composite, covariance_from_quaternion_scale,
                       invert_transform, transform_points, validate_transform, wrap_angle)
from .scene import GaussianScene, SceneState


@dataclass
class LiDARRequest:
    lidar_id: str
    timestamp: float
    T_world_lidar: Tensor
    azimuth: Tensor
    elevation: Tensor
    near: float = .1
    far: float = 200.
    beam_divergence: float = .003
    scan_duration: float = 0.
    linear_velocity: Tensor | None = None
    pixel_chunk_size: int = 4096
    time_offsets: Tensor | None = None
    # Native sensor-local radians per second; None = zero.
    angular_velocity: Tensor | None = None

    def __post_init__(self):
        if not (math.isfinite(self.timestamp) and 0 < self.near < self.far < float("inf")):
            raise ValueError("LiDAR timestamp/near/far must be finite and 0 < near < far")
        if not math.isfinite(self.scan_duration) or self.scan_duration < 0:
            raise ValueError("scan_duration must be nonnegative seconds")
        if not math.isfinite(self.beam_divergence) or self.beam_divergence <= 0 or self.pixel_chunk_size < 1 or int(self.pixel_chunk_size) != self.pixel_chunk_size:
            raise ValueError("beam_divergence and pixel_chunk_size must be positive")
        self.pixel_chunk_size = int(self.pixel_chunk_size)
        if self.angular_velocity is not None:
            angular = as_tensor(self.angular_velocity)
            if angular.shape != (3,) or not torch.isfinite(angular).all():
                raise ValueError("angular_velocity must be a finite sensor-local [3] vector in radians per second")
        if self.time_offsets is not None:
            azimuth, elevation = torch.as_tensor(self.azimuth), torch.as_tensor(self.elevation)
            if azimuth.ndim != 1 or elevation.ndim != 1:
                raise ValueError("time_offsets requires 1D azimuth and elevation arrays")
            expand_time_offsets(self.time_offsets, len(elevation), len(azimuth))


def expand_time_offsets(time_offsets, elevation_count: int, azimuth_count: int, *, like=None) -> Tensor:
    """Validate seconds relative to the request timestamp and expand to [E,A].

    Capture order need not match angle order. Values are not recentered, and
    explicit times override the synthetic schedule even when scan_duration=0.
    """
    offsets = as_tensor(time_offsets, like=like)
    if offsets.shape == (azimuth_count,):
        offsets = offsets[None].expand(elevation_count, -1)
    elif offsets.shape != (elevation_count, azimuth_count):
        raise ValueError("time_offsets must have shape [A] or [E,A] matching the LiDAR angle arrays")
    if not torch.isfinite(offsets).all():
        raise ValueError("time_offsets must contain finite seconds relative to timestamp")
    return offsets


def _project(scene: GaussianScene, request: LiDARRequest, state: SceneState, offset: float):
    attributes = scene.world_gaussians(state, offset)
    pose = as_tensor(request.T_world_lidar, like=scene.means)
    validate_transform(pose, "T_world_lidar")
    if request.linear_velocity is not None:
        velocity = as_tensor(request.linear_velocity, like=scene.means)
        if velocity.shape != (3,) or not torch.isfinite(velocity).all():
            raise ValueError("linear_velocity must be a finite sensor-local [3] vector")
        delta = pose[:3, :3] @ velocity * offset
        pose = torch.cat((torch.cat((pose[:3, :3], (pose[:3, 3]+delta)[:, None]), -1), pose[3:]), 0)
    world_to_lidar = invert_transform(pose)
    means = transform_points(world_to_lidar, attributes["means"])
    ranges = torch.linalg.vector_norm(means, dim=-1)
    valid = (ranges > request.near) & (ranges < request.far)
    means, ranges = means[valid], ranges[valid]
    x, y, z = means.unbind(-1)
    # Azimuth itself is singular at poles. These floors keep the reference
    # approximation finite; automotive beams should remain away from +/-90deg.
    horizontal_squared = (x.square()+y.square()).clamp_min(1e-8)
    horizontal = horizontal_squared.sqrt()
    range_squared = ranges.square()
    azimuth, elevation = torch.atan2(y, x), torch.atan2(z, horizontal)
    jacobian = torch.stack((-y/horizontal_squared, x/horizontal_squared, torch.zeros_like(x),
                            -x*z/(range_squared*horizontal), -y*z/(range_squared*horizontal),
                            horizontal/range_squared), -1).reshape(-1, 2, 3)
    covariance = covariance_from_quaternion_scale(attributes["quats"][valid], attributes["scales"][valid])
    rotation = world_to_lidar[:3, :3]
    covariance = rotation @ covariance @ rotation.T
    angular_covariance = jacobian @ covariance @ jacobian.transpose(-1, -2)
    angular_covariance = angular_covariance + torch.eye(2, device=means.device, dtype=means.dtype)*request.beam_divergence**2
    order = ranges.argsort()
    return (torch.stack((azimuth, elevation), -1)[order], torch.linalg.inv(angular_covariance)[order],
            ranges[order], attributes["opacities"][valid][order], attributes["intensities"][valid][order, None])


def _render_beams(scene: GaussianScene, request: LiDARRequest, state: SceneState,
                  angles: Tensor, offset: float) -> dict[str, Tensor]:
    projected, inverse_covariance, ranges, opacities, intensities = _project(scene, request, state, offset)
    range_parts, intensity_parts, hit_parts = [], [], []
    for start in range(0, len(angles), request.pixel_chunk_size):
        delta = angles[start:start+request.pixel_chunk_size][None] - projected[:, None]
        delta = torch.stack((wrap_angle(delta[..., 0]), delta[..., 1]), -1)
        mahalanobis = torch.einsum("npi,nij,npj->np", delta, inverse_covariance, delta)
        alpha = opacities[:, None]*torch.exp(-.5*mahalanobis)
        alpha = torch.where(mahalanobis <= 9., alpha, torch.zeros_like(alpha))
        weighted_intensity, depth, accumulated = composite(alpha, intensities, ranges)
        intensity = weighted_intensity[:, 0]/accumulated.clamp_min(1e-8)
        intensity = torch.where(accumulated > 1e-8, intensity, torch.zeros_like(intensity))
        range_parts.append(depth)
        intensity_parts.append(intensity)
        hit_parts.append(accumulated)
    return {"range": torch.cat(range_parts), "intensity": torch.cat(intensity_parts),
            "hit_probability": torch.cat(hit_parts)}


def render_lidar(scene: GaussianScene, request: LiDARRequest, state: SceneState | None = None) -> dict[str, Tensor]:
    """Return [E,A] range/intensity/hit_probability/ray_drop and [E,A,3] points.

    Azimuth and elevation are independent, possibly nonuniform 1D arrays in
    radians, with azimuth zero along local +x and elevation towards local +z.
    T_world_lidar defines the physical sensor basis. Explicit time_offsets
    [A] or [E,A] are seconds relative to timestamp. Without them, columns use
    a synthetic increasing-order schedule centered on timestamp. Each point
    is in its instantaneous local frame (no implicit deskew); no-hit values
    are zero.
    """
    if request.angular_velocity is not None:
        angular = as_tensor(request.angular_velocity, like=scene.means)
        if angular.shape != (3,) or not torch.isfinite(angular).all():
            raise ValueError("angular_velocity must be a finite sensor-local [3] vector in radians per second")
        if torch.any(angular != 0):
            raise NotImplementedError("Reference LiDAR renderer does not support nonzero angular_velocity; use the official backend")
    state = state or SceneState(request.timestamp)
    azimuth, elevation = [as_tensor(value, like=scene.means) for value in (request.azimuth, request.elevation)]
    if any(value.ndim != 1 or len(value) == 0 or not torch.isfinite(value).all() for value in (azimuth, elevation)):
        raise ValueError("azimuth and elevation must be finite nonempty 1D angle arrays in radians")
    if torch.any(elevation.abs() >= math.pi/2):
        raise ValueError("elevations must lie strictly between -pi/2 and pi/2")
    e, a = torch.meshgrid(elevation, azimuth, indexing="ij")
    angles = torch.stack((a, e), -1)
    if request.time_offsets is not None:
        offsets = expand_time_offsets(request.time_offsets, len(elevation), len(azimuth), like=scene.means)
        unique_offsets, groups = torch.unique(offsets.reshape(-1), return_inverse=True)
        flat_angles = angles.reshape(-1, 2)
        flat = {key: scene.means.new_zeros(len(flat_angles)) for key in ("range", "intensity", "hit_probability")}
        # Common column times share one scene projection. Distinct per-beam
        # hardware times are also supported; this reference path is explicit.
        for group, offset in enumerate(unique_offsets):
            mask = groups == group
            part = _render_beams(scene, request, state, flat_angles[mask], float(offset))
            for key, value in part.items():
                flat[key][mask] = value
        result = {key: value.reshape(len(elevation), len(azimuth)) for key, value in flat.items()}
    elif request.scan_duration > 0:
        columns = []
        for column in range(len(azimuth)):
            offset = ((column+.5)/len(azimuth)-.5)*request.scan_duration
            columns.append(_render_beams(scene, request, state, angles[:, column], offset))
        result = {key: torch.stack([part[key] for part in columns], dim=1) for key in columns[0]}
    else:
        flat = _render_beams(scene, request, state, angles.reshape(-1, 2), 0.)
        result = {key: value.reshape(len(elevation), len(azimuth)) for key, value in flat.items()}
    result["ray_drop"] = 1-result["hit_probability"]
    direction = torch.stack((torch.cos(e)*torch.cos(a), torch.cos(e)*torch.sin(a), torch.sin(e)), -1)
    result["points"] = direction * result["range"][..., None]
    return result
