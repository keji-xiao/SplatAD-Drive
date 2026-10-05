"""CPU/GPU PyTorch reference pinhole Gaussian rasterizer.

Uses the first-order projection of full anisotropic covariance and sorted
front-to-back alpha compositing. This is a correctness/experimentation tool,
not the official CUDA renderer or SplatAD's learned appearance CNN. Rolling
shutter samples each row at its midpoint, with linear sensor translation
(velocity in the camera's local frame) and interpolated actor trajectories.
Sensor rotation during readout and exposure-time integration are not modeled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .geometry import (as_tensor, composite, covariance_from_quaternion_scale,
                       invert_transform, transform_points, validate_transform)
from .scene import GaussianScene, SceneState


@dataclass
class CameraRequest:
    camera_id: str
    timestamp: float
    T_world_camera: Tensor
    K: Tensor
    width: int
    height: int
    near: float = .05
    # SplatAD represents sky and distant geometry with far Gaussians. A LiDAR
    # range limit (e.g. 200 m) would incorrectly clip their camera contribution.
    far: float = 1e10
    background: tuple[float, float, float] = (0., 0., 0.)
    rolling_shutter_duration: float = 0.
    linear_velocity: Tensor | None = None
    pixel_variance: float = .3
    pixel_chunk_size: int = 4096
    # Sensor-local OpenCV (right/down/forward) radians per second; None = zero.
    angular_velocity: Tensor | None = None

    def __post_init__(self):
        if self.width < 1 or self.height < 1 or int(self.width) != self.width or int(self.height) != self.height:
            raise ValueError("camera width and height must be positive integers")
        self.width, self.height = int(self.width), int(self.height)
        if not (math.isfinite(self.timestamp) and 0 < self.near < self.far < float("inf")):
            raise ValueError("camera timestamp/near/far must be finite and 0 < near < far")
        if not math.isfinite(self.rolling_shutter_duration) or self.rolling_shutter_duration < 0:
            raise ValueError("rolling_shutter_duration must be nonnegative seconds")
        if not math.isfinite(self.pixel_variance) or self.pixel_variance <= 0 or self.pixel_chunk_size < 1 or int(self.pixel_chunk_size) != self.pixel_chunk_size:
            raise ValueError("pixel_variance and pixel_chunk_size must be positive")
        self.pixel_chunk_size = int(self.pixel_chunk_size)
        if self.angular_velocity is not None:
            angular = as_tensor(self.angular_velocity)
            if angular.shape != (3,) or not torch.isfinite(angular).all():
                raise ValueError("angular_velocity must be a finite sensor-local [3] vector in radians per second")


def _project(scene: GaussianScene, request: CameraRequest, state: SceneState, offset: float):
    attributes = scene.world_gaussians(state, offset)
    pose = as_tensor(request.T_world_camera, like=scene.means)
    validate_transform(pose, "T_world_camera")
    if request.linear_velocity is not None:
        velocity = as_tensor(request.linear_velocity, like=scene.means)
        if velocity.shape != (3,) or not torch.isfinite(velocity).all():
            raise ValueError("linear_velocity must be a finite sensor-local [3] vector")
        delta = pose[:3, :3] @ velocity * offset
        pose = torch.cat((torch.cat((pose[:3, :3], (pose[:3, 3]+delta)[:, None]), -1), pose[3:]), 0)
    world_to_camera = invert_transform(pose)
    means = transform_points(world_to_camera, attributes["means"])
    depths = means[:, 2]
    valid = (depths > request.near) & (depths < request.far)
    means, depths = means[valid], depths[valid]
    k = as_tensor(request.K, like=means)
    if k.shape != (3, 3) or not torch.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("K must be finite [3,3] with positive focal lengths")
    if not torch.allclose(k[2], k.new_tensor([0., 0., 1.]), atol=1e-6):
        raise ValueError("K must use homogeneous pinhole intrinsics [0,0,1] in its final row")
    projected = (means @ k.T)[:, :2]/depths[:, None]
    covariances = covariance_from_quaternion_scale(attributes["quats"][valid], attributes["scales"][valid])
    rotation = world_to_camera[:3, :3]
    covariances = rotation @ covariances @ rotation.T
    numerator_u = means[:, 0]*k[0, 0]+means[:, 1]*k[0, 1]
    numerator_v = means[:, 0]*k[1, 0]+means[:, 1]*k[1, 1]
    jacobian = torch.stack((k[0, 0]/depths, k[0, 1]/depths, -numerator_u/depths.square(),
                            k[1, 0]/depths, k[1, 1]/depths, -numerator_v/depths.square()), -1).reshape(-1, 2, 3)
    covariance_2d = jacobian @ covariances @ jacobian.transpose(-1, -2)
    covariance_2d = covariance_2d + torch.eye(2, device=means.device, dtype=means.dtype)*request.pixel_variance
    order = depths.argsort()
    return (projected[order], torch.linalg.inv(covariance_2d)[order],
            depths[order], attributes["opacities"][valid][order], attributes["colors"][valid][order])


def _render_pixels(scene: GaussianScene, request: CameraRequest, state: SceneState,
                   pixels: Tensor, offset: float) -> dict[str, Tensor]:
    projected, inverse_covariance, depths, opacities, colors = _project(scene, request, state, offset)
    background = as_tensor(request.background, like=scene.means)
    if background.shape != (3,) or not torch.isfinite(background).all() or torch.any((background < 0) | (background > 1)):
        raise ValueError("background must contain three finite RGB values in [0,1]")
    rgb_parts, depth_parts, alpha_parts = [], [], []
    for start in range(0, len(pixels), request.pixel_chunk_size):
        delta = pixels[start:start+request.pixel_chunk_size][None] - projected[:, None]
        mahalanobis = torch.einsum("npi,nij,npj->np", delta, inverse_covariance, delta)
        alpha = opacities[:, None] * torch.exp(-.5*mahalanobis)
        alpha = torch.where(mahalanobis <= 9., alpha, torch.zeros_like(alpha))
        rgb, depth, accumulated = composite(alpha, colors, depths)
        rgb_parts.append(rgb+(1-accumulated[:, None])*background)
        depth_parts.append(depth)
        alpha_parts.append(accumulated)
    return {"rgb": torch.cat(rgb_parts), "depth": torch.cat(depth_parts), "alpha": torch.cat(alpha_parts)}


def render_camera(scene: GaussianScene, request: CameraRequest, state: SceneState | None = None) -> dict[str, Tensor]:
    """Return RGB [H,W,3], camera-z conditional expected depth [H,W], alpha.

    Pixel coordinates are sampled at (column+0.5,row+0.5); timestamp denotes
    the middle of readout. With zero alpha depth is zero. Three-sigma splats
    and center-based near/far rejection give piecewise differentiability.
    """
    if request.angular_velocity is not None:
        angular = as_tensor(request.angular_velocity, like=scene.means)
        if angular.shape != (3,) or not torch.isfinite(angular).all():
            raise ValueError("angular_velocity must be a finite sensor-local [3] vector in radians per second")
        if torch.any(angular != 0):
            raise NotImplementedError("Reference camera renderer does not support nonzero angular_velocity; use the official backend")
    state = state or SceneState(request.timestamp)
    y, x = torch.meshgrid(torch.arange(request.height, device=scene.means.device, dtype=scene.means.dtype)+.5,
                          torch.arange(request.width, device=scene.means.device, dtype=scene.means.dtype)+.5,
                          indexing="ij")
    pixels = torch.stack((x, y), -1)
    if request.rolling_shutter_duration > 0:
        rows = []
        for row in range(request.height):
            offset = ((row+.5)/request.height-.5)*request.rolling_shutter_duration
            rows.append(_render_pixels(scene, request, state, pixels[row], offset))
        return {key: torch.stack([result[key] for result in rows]) for key in rows[0]}
    result = _render_pixels(scene, request, state, pixels.reshape(-1, 2), 0.)
    return {"rgb": result["rgb"].reshape(request.height, request.width, 3),
            "depth": result["depth"].reshape(request.height, request.width),
            "alpha": result["alpha"].reshape(request.height, request.width)}
