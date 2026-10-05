"""Differentiable geometry. Transforms map local coordinates to parent/world.

Quaternions are ``(w, x, y, z)``. Camera coordinates are OpenCV (right,
down, forward). LiDAR coordinates are sensor-local, defined by T_world_lidar:
azimuth zero is local +x and positive elevation points towards local +z.
Reference examples use FLU, whereas native PandaSet forward is local -y.
Angles use radians and distance uses metres throughout this package.
"""
from __future__ import annotations

import torch
from torch import Tensor


def as_tensor(value, *, like: Tensor | None = None) -> Tensor:
    if like is not None:
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    result = torch.as_tensor(value)
    return result if result.is_floating_point() else result.to(torch.float32)


def normalize_quaternion(q: Tensor) -> Tensor:
    if q.shape[-1] != 4:
        raise ValueError("quaternions must end in 4 components (w,x,y,z)")
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if torch.any(norm.detach() < 1e-12):
        raise ValueError("zero quaternion has no rotation")
    return q / norm.clamp_min(1e-12)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    w, x, y, z = normalize_quaternion(q).unbind(-1)
    return torch.stack((
        1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Stable conversion, choosing the best conditioned of four candidates."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in (3,3)")
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    squared = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                           1-m00+m11-m22, 1-m00-m11+m22), -1)
    # Clamp before sqrt avoids infinite derivatives on unused zero candidates.
    magnitudes = torch.sqrt(squared.clamp_min(1e-12))
    candidates = torch.stack((
        torch.stack((squared[..., 0], m21-m12, m02-m20, m10-m01), -1),
        torch.stack((m21-m12, squared[..., 1], m10+m01, m02+m20), -1),
        torch.stack((m02-m20, m10+m01, squared[..., 2], m12+m21), -1),
        torch.stack((m10-m01, m20+m02, m21+m12, squared[..., 3]), -1),
    ), -2) / (2 * magnitudes[..., :, None])
    choice = magnitudes.argmax(-1)
    q = candidates.gather(-2, choice[..., None, None].expand(choice.shape+(1, 4))).squeeze(-2)
    return normalize_quaternion(q)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def quaternion_slerp(a: Tensor, b: Tensor, fraction: Tensor) -> Tensor:
    a, b = normalize_quaternion(a), normalize_quaternion(b)
    dot = (a*b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(max=1.0)
    # Use a safe interior angle even in the unused near-identical branch.
    theta = torch.acos(dot.clamp(max=1.0-1e-7))
    f = as_tensor(fraction, like=a)
    while f.ndim < a.ndim:
        f = f.unsqueeze(-1)
    spherical = (torch.sin((1-f)*theta)*a + torch.sin(f*theta)*b)/torch.sin(theta)
    linear = (1-f)*a + f*b
    return normalize_quaternion(torch.where(dot > .9995, linear, spherical))


def validate_transform(transform: Tensor, name: str = "transform") -> None:
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 rigid transform")
    expected = transform.new_tensor([0., 0., 0., 1.])
    rotation = transform[:3, :3]
    if (not torch.allclose(transform[3], expected, atol=1e-5, rtol=1e-5)
            or not torch.allclose(rotation.T @ rotation, torch.eye(3, device=rotation.device,
                                  dtype=rotation.dtype), atol=1e-4, rtol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation), rotation.new_tensor(1.), atol=1e-4)):
        raise ValueError(f"{name} must contain a proper orthonormal rotation and homogeneous last row")


def invert_transform(transform: Tensor) -> Tensor:
    rotation = transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation @ transform[..., :3, 3, None])
    top = torch.cat((rotation, translation), dim=-1)
    bottom = transform.new_tensor([0., 0., 0., 1.]).expand(transform.shape[:-2]+(1,4))
    return torch.cat((top, bottom), -2)


def transform_points(transform: Tensor, points: Tensor) -> Tensor:
    return points @ transform[:3, :3].T + transform[:3, 3]


def covariance_from_quaternion_scale(quats: Tensor, scales: Tensor) -> Tensor:
    rotation = quaternion_to_matrix(quats)
    scaled = rotation * scales[..., None, :]
    return scaled @ scaled.transpose(-1, -2)


def interpolate_pose(first: Tensor, second: Tensor, fraction: Tensor) -> Tensor:
    f = as_tensor(fraction, like=first)
    quaternion = quaternion_slerp(matrix_to_quaternion(first[:3, :3]),
                                  matrix_to_quaternion(second[:3, :3]), f)
    rotation = quaternion_to_matrix(quaternion)
    translation = (1-f)*first[:3, 3] + f*second[:3, 3]
    return torch.cat((torch.cat((rotation, translation[:, None]), -1),
                      first.new_tensor([[0., 0., 0., 1.]])), 0)


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def composite(alpha: Tensor, values: Tensor, depth: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Front-to-back composite sorted [N,P] alpha; conditional expected depth."""
    if alpha.shape[0] == 0:
        size = alpha.shape[1]
        return (values.new_zeros((size, values.shape[-1])), alpha.new_zeros(size),
                alpha.new_zeros(size))
    alpha = alpha.clamp(0., .999)
    transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1-alpha), 0), 0)[:-1]
    weights = alpha * transmittance
    accumulated = weights.sum(0)
    color = weights.T @ values
    expected = (weights*depth[:, None]).sum(0)/accumulated.clamp_min(1e-8)
    expected = torch.where(accumulated > 1e-8, expected, torch.zeros_like(expected))
    return color, expected, accumulated
