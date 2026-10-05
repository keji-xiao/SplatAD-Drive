from dataclasses import replace

import pytest
import torch

from splatad_drive.camera_renderer import CameraRequest
from splatad_drive.demo import overlay
from splatad_drive.lidar_renderer import LiDARRequest


def fixture():
    camera = CameraRequest("camera", 12., torch.eye(4),
                           torch.tensor([[5., 0., 5.], [0., 5., 4.], [0., 0., 1.]]), 11, 9)
    lidar = LiDARRequest("lidar", 10., torch.eye(4), torch.tensor([-.1, .1]), torch.tensor([0.]),
                         linear_velocity=torch.tensor([10., 0., 0.]))
    return torch.zeros(9, 11, 3), camera, lidar


def colored_pixels(rgb):
    return {tuple(index) for index in (rgb.sum(-1) > 0).nonzero().tolist()}


def test_default_preserves_nominal_projection_despite_motion_and_different_timestamps():
    rgb, camera, lidar = fixture()
    lidar = replace(lidar, time_offsets=torch.tensor([-.2, .2]))
    points = torch.tensor([[[0., 0., 5.], [0., 0., 5.]]])
    assert colored_pixels(overlay(rgb, points, camera, lidar)) == {(4, 5)}
    compensated = overlay(rgb, points, camera, lidar, compensate_lidar_translation=True)
    assert colored_pixels(compensated) == {(4, 3), (4, 7)}
    # Changing labels alone does not move geometry: poses already correspond
    # to their respective timestamps, so adding the intersensor delta is wrong.
    shifted_labels = overlay(rgb, points, replace(camera, timestamp=99.), lidar, compensate_lidar_translation=True)
    torch.testing.assert_close(compensated, shifted_labels)


def test_reference_scan_duration_column_midpoints_and_explicit_priority():
    rgb, camera, lidar = fixture()
    points = torch.tensor([[[0., 0., 5.], [0., 0., 5.]]])
    synthetic = replace(lidar, scan_duration=1.)
    assert colored_pixels(overlay(rgb, points, camera, synthetic, compensate_lidar_translation=True)) == {(4, 2), (4, 7)}
    explicit = replace(synthetic, time_offsets=torch.tensor([.1, .2]))
    # Explicit non-centered times override scan_duration and preserve origin.
    assert colored_pixels(overlay(rgb, points, camera, explicit, compensate_lidar_translation=True)) == {(4, 6), (4, 7)}


@pytest.mark.parametrize("times,first_pixel", [([-.2, .2], (4, 3)), ([.2, -.2], (4, 7))])
def test_explicit_times_preserve_forward_or_reverse_scan_at_azimuth_seam(times, first_pixel):
    rgb, camera, lidar = fixture()
    # The user-provided schedule defines scan direction, even across +/-pi.
    # Projection must not sort/recenter times or infer time from azimuth.
    lidar = replace(lidar, azimuth=torch.tensor([3.13, -3.13]), time_offsets=torch.tensor(times))
    points = torch.tensor([[[0., 0., 5.], [0., 0., 5.]]])
    result = overlay(rgb, points, camera, lidar, torch.tensor([[True, False]]),
                     compensate_lidar_translation=True)
    assert colored_pixels(result) == {first_pixel}


def test_velocity_is_sensor_local_and_camera_pose_is_nominal_center():
    rgb, camera, lidar = fixture()
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    lidar = replace(lidar, T_world_lidar=pose)
    points = torch.tensor([[0., 0., 5.]])
    moved = overlay(rgb, points, camera, lidar, compensate_lidar_translation=True, point_time_offsets=[.2])
    assert colored_pixels(moved) == {(6, 5)}  # local +x rotates to world +y.
    camera_pose = camera.T_world_camera.clone()
    camera_pose[1, 3] = 2.
    camera = replace(camera, T_world_camera=camera_pose, rolling_shutter_duration=.03,
                     linear_velocity=torch.tensor([100., 100., 0.]))
    centered = overlay(rgb, points, camera, lidar, compensate_lidar_translation=True, point_time_offsets=[.2])
    assert colored_pixels(centered) == {(4, 5)}  # no camera row-time compensation.


def test_full_grid_per_ray_timing_and_valid_mask_share_flatten_order():
    rgb, camera, lidar = fixture()
    lidar = replace(lidar, elevation=torch.tensor([-.1, .1]),
                    time_offsets=torch.tensor([[-.2, -.1], [.1, .2]]))
    points = torch.tensor([[[0., 0., 5.], [0., 0., 5.]], [[0., 0., 5.], [0., 0., 5.]]])
    valid = torch.tensor([[True, False], [False, True]])
    result = overlay(rgb, points, camera, lidar, valid, compensate_lidar_translation=True)
    assert colored_pixels(result) == {(4, 3), (4, 7)}


def test_filtered_cloud_requires_corresponding_point_times():
    rgb, camera, lidar = fixture()
    lidar = replace(lidar, scan_duration=.1)
    points = torch.tensor([[0., 0., 5.]])
    with pytest.raises(ValueError, match="filtered cloud needs point_time_offsets"):
        overlay(rgb, points, camera, lidar, compensate_lidar_translation=True)
    result = overlay(rgb, points, camera, lidar, compensate_lidar_translation=True, point_time_offsets=[.2])
    assert colored_pixels(result) == {(4, 7)}


def test_invalid_timing_and_velocity_are_rejected():
    rgb, camera, lidar = fixture()
    points = torch.tensor([[0., 0., 5.]])
    with pytest.raises(ValueError, match="requires compensate"):
        overlay(rgb, points, camera, lidar, point_time_offsets=[.1])
    with pytest.raises(ValueError, match="finite seconds"):
        overlay(rgb, points, camera, lidar, compensate_lidar_translation=True, point_time_offsets=[float("nan")])
    with pytest.raises(ValueError, match="match the input point"):
        overlay(rgb, points, camera, lidar, compensate_lidar_translation=True, point_time_offsets=[.1, .2])
    with pytest.raises(ValueError, match="finite sensor-local"):
        overlay(rgb, points, camera, replace(lidar, linear_velocity=torch.tensor([1., 2.])),
                compensate_lidar_translation=True, point_time_offsets=[.1])


def test_missing_velocity_and_zero_scan_are_identity_even_for_unstructured_cloud():
    rgb, camera, lidar = fixture()
    points = torch.tensor([[0., 0., 5.]])
    result = overlay(rgb, points, camera, lidar, compensate_lidar_translation=True)
    assert colored_pixels(result) == {(4, 5)}
    result = overlay(rgb, points, camera, replace(lidar, linear_velocity=None),
                     compensate_lidar_translation=True, point_time_offsets=[.2])
    assert colored_pixels(result) == {(4, 5)}
