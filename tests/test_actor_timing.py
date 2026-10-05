import json
from pathlib import Path

import numpy as np
import pytest

from splatad_drive.data.actor_timing import estimate_actor_capture_times


def estimate(centers, points, rotations=None, dims=None, pose=None, front=100., period=.1):
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    return estimate_actor_capture_times(centers, np.tile(np.eye(3), (len(centers), 1, 1)) if rotations is None else rotations,
                                        np.ones_like(centers)*2 if dims is None else dims,
                                        np.asarray(points).reshape(-1, 6), np.eye(4) if pose is None else pose, front, period)


def test_median_preserves_unix_subsecond_precision_and_diagnostics():
    base = 1700000000.
    points = [[0., -10., 0., 10., base+.01, 0.], [0., -10., 0., 20., base+.03, 0.],
              [0., -10., 0., 30., base+.07, 0.]]
    times, metadata = estimate([[0., -10., 0.]], points, front=base)
    assert times.dtype == np.float64 and times[0] == base+.03
    assert metadata[0]["source"] == "lidar_bbox_median" and metadata[0]["point_count"] == 3
    assert metadata[0]["low_support"] is True
    assert metadata[0]["iqr_seconds"] == pytest.approx(.03, abs=3e-7)


def test_sensor_and_nonfinite_filter_do_not_contaminate_time():
    points = [[0., -10., 0., 1., 100.001, 0.], [0., -10., 0., 1., 101., 1.],
              [0., -10., 0., np.nan, 102., 0.], [0., -10., 0., 1., np.inf, 0.],
              [np.inf, -10., 0., 1., 103., 0.], [30., -10., 0., 1., 104., 0.]]
    times, metadata = estimate([[0., -10., 0.]], points)
    assert times[0] == 100.001
    assert metadata[0]["point_count"] == 1


def test_oriented_actor_box_and_sensor_pose_are_distinct_transforms():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    center = np.array([10., -10., 0.])
    local_inside = np.array([[.4, 1.9, .1], [-.4, -1.8, .2]])
    inside = local_inside @ rotation.T+center
    points = np.column_stack((inside, [1., 1.], [10.01, 10.03], [0., 0.]))
    points = np.concatenate((points, [[11.8, -10., 0., 1., 10.7, 0.]]))
    # Last point is also in this rotated long-axis box; place it outside width.
    points[-1, :3] = center+np.array([0., 1.8, 0.])
    times, metadata = estimate([center], points, rotations=rotation[None], dims=[[1., 4., 2.]])
    assert times[0] == pytest.approx(10.02)
    assert metadata[0]["point_count"] == 2
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, [7., -3., 1.]
    local = np.array([1., -10., 0.])
    world = local@rotation.T+pose[:3, 3]
    predicted, _ = estimate([world], [], pose=pose)
    assert predicted[0] == pytest.approx(100.+np.arctan2(-1., 10.)/(2*np.pi)*.1)


def test_fallback_native_scan_direction_and_center_phase():
    centers = [[0., -10., 0.], [10., 0., 0.], [-10., 0., 0.]]
    times, metadata = estimate(centers, [])
    np.testing.assert_allclose(times, [100., 99.975, 100.025], atol=1e-10)
    assert all(item["source"] == "angle_fallback" and item["iqr_seconds"] is None for item in metadata)


def test_scan_seam_is_bounded_and_circularly_continuous():
    times, _ = estimate([[1e-5, 10., 0.], [-1e-5, 10., 0.]], [])
    assert np.all(np.abs(times-100.) <= .05)
    assert times[0] < 100. and times[1] > 100.
    assert abs((times[1]-times[0])-.1) < 1e-7


@pytest.mark.parametrize("invalid", ["dims", "rotation", "period", "undefined_azimuth"])
def test_rejects_invalid_geometry_or_timing(invalid):
    kwargs = {}
    centers = [[0., -10., 0.]]
    if invalid == "dims":
        kwargs["dims"] = [[1., -1., 1.]]
    elif invalid == "rotation":
        kwargs["rotations"] = np.array([np.diag([-1., 1., 1.])])
    elif invalid == "period":
        kwargs["period"] = 0
    else:
        centers = [[0., 0., 2.]]
    with pytest.raises(ValueError):
        estimate(centers, [], **kwargs)


def test_empty_batch_is_well_defined():
    times, diagnostics = estimate([], [])
    assert times.shape == (0,) and times.dtype == np.float64 and diagnostics == []


@pytest.mark.upstream
def test_original_pandaset_028_bus_frame79_measurement():
    root = Path(__file__).resolve().parents[1]/"data"/"pandaset"/"028"
    if not (root/"lidar/79.pkl.gz").exists():
        pytest.skip("Original PandaSet 028 data is unavailable")
    pd = pytest.importorskip("pandas")
    table = pd.read_pickle(root/"annotations/cuboids/79.pkl.gz")
    bus = table[table.uuid == "53a31f42-ec8d-49a1-aa73-543bce729cbd"].iloc[0]
    yaw = float(bus.yaw)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    center = [[bus[f"position.{k}"] for k in "xyz"]]
    dims = [[bus[f"dimensions.{k}"] for k in "xyz"]]
    points = pd.read_pickle(root/"lidar/79.pkl.gz")[["x", "y", "z", "i", "t", "d"]].to_numpy()
    front = json.loads((root/"camera/front_camera/timestamps.json").read_text())[79]
    times, metadata = estimate(center, points, rotations=rotation[None], dims=dims, front=front)
    assert metadata[0]["point_count"] == 47
    assert (times[0]-front)*1000 == pytest.approx(2.9654502868652344, abs=1e-6)
