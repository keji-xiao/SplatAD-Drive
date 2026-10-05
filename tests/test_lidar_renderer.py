from dataclasses import replace
import math

import pytest
import torch

from splatad_drive.lidar_renderer import LiDARRequest, render_lidar
from splatad_drive.geometry import quaternion_to_matrix
from splatad_drive.scene import GaussianScene


def request(**kwargs):
    result = LiDARRequest("lidar", 0., torch.eye(4), torch.tensor([0.]), torch.tensor([0.]))
    return replace(result, **kwargs)


def scene(means, opacities=None, scales=None):
    n = len(means)
    return GaussianScene(means, [[1., 0., 0., 0.]]*n, scales or [[.2, .2, .2]]*n,
                         opacities or [.8]*n, [[.5, .5, .5]]*n, intensities=[.6]*n)


def test_range_compositing_and_flu_points():
    result = render_lidar(scene([[2., 0., 0.], [4., 0., 0.]], [.5, .8]), request())
    expected = torch.tensor((.5*2+.4*4)/.9)
    assert torch.allclose(result["range"][0, 0], expected, atol=1e-6)
    assert torch.allclose(result["intensity"][0, 0], torch.tensor(.6), atol=1e-6)
    assert torch.allclose(result["points"][0, 0], torch.stack((expected, expected*0, expected*0)))
    assert torch.allclose(result["ray_drop"], 1-result["hit_probability"])


def test_azimuth_seam_equal_on_both_sides():
    model = scene([[-5., 0., 0.]])
    result = render_lidar(model, request(azimuth=torch.tensor([-math.pi+.005, math.pi-.005, 0.])))
    assert torch.allclose(result["hit_probability"][0, 0], result["hit_probability"][0, 1], atol=1e-5)
    assert result["hit_probability"][0, 0] > .7
    assert result["hit_probability"][0, 2] == 0


def test_nonuniform_elevation_beams_use_actual_angles_and_gradients():
    elevation = .18
    model = scene([[5*math.cos(elevation), 0., 5*math.sin(elevation)]], scales=[[.1, .2, .08]])
    result = render_lidar(model, request(elevation=torch.tensor([-.3, -.01, .18, .5])))
    assert result["hit_probability"][:, 0].argmax() == 2
    assert torch.allclose(result["range"][2, 0], torch.tensor(5.), atol=1e-5)
    (result["range"].sum()+result["intensity"].sum()+result["hit_probability"].sum()).backward()
    for parameter in (model.means, model.log_scales, model.opacity_logits, model.intensity_logits):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_scan_motion_and_local_velocity_frame():
    model = scene([[5., 0., 0.]])
    static = request(azimuth=torch.tensor([-.01, .01]))
    moving = replace(static, scan_duration=1., linear_velocity=torch.tensor([1., 0., 0.]))
    result = render_lidar(model, moving)
    assert result["range"][0, 0] > result["range"][0, 1]
    assert torch.allclose(render_lidar(model, static)["range"], torch.tensor([[5., 5.]]), atol=1e-5)
    rotated_pose = torch.eye(4)
    rotated_pose[:3, :3] = quaternion_to_matrix(torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)]))
    rotated = render_lidar(scene([[0., 5., 0.]]), replace(moving, T_world_lidar=rotated_pose))
    assert torch.allclose(rotated["range"], result["range"], atol=1e-5)


def test_explicit_per_beam_times_override_scan_order_and_keep_gradients():
    model = scene([[5., 0., 0.]])
    offsets = torch.tensor([[.2, -.2], [.1, -.1]])
    capture = request(azimuth=torch.tensor([-.001, .001]), elevation=torch.tensor([0., .001]),
                      linear_velocity=torch.tensor([1., 0., 0.]), time_offsets=offsets,
                      scan_duration=100.)
    result = render_lidar(model, capture)
    # Local +x sensor motion shortens range at positive time. Per-row times
    # differ, and reversed capture order must not be sorted with azimuth.
    assert torch.allclose(result["range"], 5.-offsets, atol=1e-5)
    assert torch.equal(offsets, torch.tensor([[.2, -.2], [.1, -.1]]))
    (result["range"].sum()+result["hit_probability"].sum()+result["intensity"].sum()).backward()
    for parameter in (model.means, model.log_scales, model.opacity_logits, model.intensity_logits):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_explicit_column_times_apply_with_zero_duration_and_without_recentering():
    model = scene([[5., 0., 0.]])
    base = request(azimuth=torch.tensor([-.001, .001]), elevation=torch.tensor([0., .001]),
                   linear_velocity=torch.tensor([1., 0., 0.]))
    constant = render_lidar(model, replace(base, time_offsets=torch.tensor([.2, .2])))
    assert torch.allclose(constant["range"], torch.full((2, 2), 4.8), atol=1e-5)
    zero = render_lidar(model, replace(base, time_offsets=torch.zeros(2), scan_duration=1.))
    assert torch.equal(zero["range"], render_lidar(model, base)["range"])
    columns = render_lidar(model, replace(base, time_offsets=torch.tensor([.1, -.1])))
    assert torch.allclose(columns["range"], torch.tensor([[4.9, 5.1], [4.9, 5.1]]), atol=1e-5)


@pytest.mark.parametrize("offsets", [torch.zeros(3), torch.zeros(2, 1), torch.tensor([float("nan"), 0.]),
                                    torch.tensor([[0., float("inf")], [0., 0.]])])
def test_explicit_times_reject_bad_shape_or_nonfinite(offsets):
    with pytest.raises(ValueError, match="time_offsets"):
        request(azimuth=torch.tensor([-.01, .01]), elevation=torch.tensor([0., .01]), time_offsets=offsets)
