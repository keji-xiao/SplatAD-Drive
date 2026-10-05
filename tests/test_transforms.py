import math

import pytest
import torch

from splatad_drive.geometry import (covariance_from_quaternion_scale, interpolate_pose,
                                    invert_transform, matrix_to_quaternion,
                                    quaternion_to_matrix, transform_points, validate_transform)


def test_transform_round_trip_and_quaternion_convention():
    quaternion = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    rotation = quaternion_to_matrix(quaternion)
    assert torch.allclose(rotation @ torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), atol=1e-6)
    transform = torch.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = torch.tensor([2., -1., 4.])
    points = torch.tensor([[1., 2., 3.], [0., 0., 0.]])
    assert torch.allclose(transform_points(invert_transform(transform), transform_points(transform, points)), points, atol=1e-6)
    assert torch.allclose(quaternion_to_matrix(matrix_to_quaternion(rotation)), rotation, atol=1e-6)


def test_covariance_rotates_anisotropy():
    q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
    covariance = covariance_from_quaternion_scale(q, torch.tensor([[3., 1., 2.]]))
    assert torch.allclose(covariance[0], torch.diag(torch.tensor([1., 9., 4.])), atol=1e-5)


def test_pose_interpolation_shortest_path_and_finite_gradients():
    first, second = torch.eye(4), torch.eye(4)
    second[:3, :3] = quaternion_to_matrix(torch.tensor([0., 0., 0., 1.]))
    second[0, 3] = 4.
    middle = interpolate_pose(first, second, torch.tensor(.5))
    assert middle[0, 3] == 2.
    assert torch.allclose(middle[:3, :3].T @ middle[:3, :3], torch.eye(3), atol=1e-6)
    first = first.double().requires_grad_()
    result = interpolate_pose(first, first, torch.tensor(.3, dtype=torch.float64))
    result.sum().backward()
    assert torch.isfinite(first.grad).all()


def test_transform_validation_rejects_scale_and_reflection():
    bad = torch.eye(4)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="proper orthonormal"):
        validate_transform(bad)
