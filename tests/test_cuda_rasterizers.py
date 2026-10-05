"""Real custom-gsplat integration checks; no mocked kernels or silent fallback.

Run with ``SPLATAD_REQUIRE_CUDA=1 python -m pytest -m cuda -v`` in the
official CUDA environment. Without CUDA these are explicitly skipped;
SPLATAD_REQUIRE_CUDA turns a missing device/dependency into a test failure.
Compilation/runtime errors are always failures once CUDA is available.
"""
from __future__ import annotations

import copy
import importlib
import math
import os
from pathlib import Path

import pytest
import torch

from splatad_drive.camera_renderer import CameraRequest, render_camera
from splatad_drive.backends.neurad import lidar_raster_grid
from splatad_drive.lidar_renderer import LiDARRequest, render_lidar
from splatad_drive.scene import GaussianScene


pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        message = "CUDA integration not executed: torch.cuda.is_available() is false"
        if os.environ.get("SPLATAD_REQUIRE_CUDA") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return torch.device("cuda")


@pytest.fixture(scope="module")
def custom_gsplat(cuda_device):
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(root / "outputs" / "torch_extensions"))
    os.environ.setdefault("MAX_JOBS", "4")
    try:
        module = importlib.import_module("gsplat.rendering")
    except ImportError as error:
        pytest.fail(f"CUDA integration dependency missing: {error}")
    assert hasattr(module, "lidar_rasterization"), "Install carlinds/splatad gsplat, not generic gsplat"
    return module


def _tensor(values, device, requires_grad=True):
    return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=requires_grad)


@pytest.mark.parametrize("render_mode", ["RGB", "D", "ED", "RGB+D", "RGB+ED"])
@pytest.mark.parametrize("direction", [1, 2, 3, 4])
@pytest.mark.parametrize("moving", [False, True], ids=["static", "moving"])
def test_official_camera_modes_rolling_shutter_forward_backward(custom_gsplat, cuda_device, render_mode, direction, moving):
    means = _tensor([[.1, -.2, 3.], [-.3, .2, 5.]], cuda_device)
    quats = _tensor([[1., .05, .1, 0.], [1., -.1, 0., .08]], cuda_device)
    scales = _tensor([[.3, .15, .2], [.25, .35, .18]], cuda_device)
    opacities = _tensor([.7, .8], cuda_device)
    colors = _tensor([[.8, .2, .1], [.1, .3, .9]], cuda_device)
    velocities = _tensor([[1.8, -.4, .1], [-.4, .8, -.1]], cuda_device, False) if moving else None
    kwargs = dict(means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                  velocities=velocities, viewmats=torch.eye(4, device=cuda_device)[None],
                  Ks=_tensor([[[35., 0., 16.], [0., 35., 16.], [0., 0., 1.]]], cuda_device, False),
                  width=32, height=32, packed=False, render_mode=render_mode,
                  linear_velocity=_tensor([[.3, -.2, .1]] if moving else [[0., 0., 0.]], cuda_device, False),
                  angular_velocity=_tensor([[0., .01, 0.]] if moving else [[0., 0., 0.]], cuda_device, False),
                  rolling_shutter_time=_tensor([.1], cuda_device, False), rolling_shutter_direction=direction)
    rendered, alpha, meta = custom_gsplat.rasterization(**kwargs)
    channels = {"RGB": 3, "D": 1, "ED": 1, "RGB+D": 4, "RGB+ED": 4}[render_mode]
    assert rendered.shape == (1, 32, 32, channels)
    assert alpha.shape == (1, 32, 32, 1)
    assert torch.isfinite(rendered).all() and torch.isfinite(alpha).all()
    assert alpha.max() > .3 and (alpha >= 0).all() and (alpha <= 1).all()
    if moving:
        with torch.no_grad():
            _, global_alpha, _ = custom_gsplat.rasterization(**{**kwargs, "rolling_shutter_time": torch.zeros(1, device=cuda_device)})
        assert not torch.allclose(alpha, global_alpha, atol=1e-5), "Nonzero readout/motion must affect alpha"
    (rendered.square().mean()+alpha.mean()).backward()
    for value in (means, quats, scales, opacities):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    if "RGB" in render_mode:
        assert colors.grad is not None and torch.isfinite(colors.grad).all()
    assert means.grad.abs().sum() > 0 and opacities.grad.abs().sum() > 0
    torch.cuda.synchronize()


@pytest.mark.parametrize("moving", [False, True], ids=["static", "moving"])
def test_official_lidar_nonuniform_seam_range_forward_backward(custom_gsplat, cuda_device, moving):
    means = _tensor([[-5., 0., 0.]], cuda_device)
    quats = _tensor([[1., 0., 0., 0.]], cuda_device)
    scales = _tensor([[.2, .3, .2]], cuda_device)
    opacities = _tensor([.8], cuda_device)
    features = _tensor([[[.6, .2]]], cuda_device)
    # Upstream angles are degrees. Sorted nonuniform channels define 2 tiles.
    elevations = _tensor([-8., -4., -1., 0., 1., 3., 6., 10.], cuda_device, False)
    azimuths = torch.arange(360, device=cuda_device, dtype=torch.float32)-179.5
    elevation, azimuth = torch.meshgrid(elevations, azimuths, indexing="ij")
    offsets = (torch.arange(360, device=cuda_device, dtype=torch.float32)/360-.5)*(.1 if moving else 0.)
    raster_points = torch.stack((azimuth, elevation, torch.full_like(azimuth, 10.),
                                 offsets[None].expand_as(azimuth)), -1)[None].contiguous()
    rendered, alpha, alpha_until, meta = custom_gsplat.lidar_rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, lidar_features=features,
        velocities=_tensor([[0., .4, 0.]], cuda_device, False) if moving else None,
        viewmats=torch.eye(4, device=cuda_device)[None], raster_pts=raster_points,
        tile_elevation_boundaries=_tensor([-10., .5, 12.], cuda_device, False),
        linear_velocity=_tensor([[.2, 0., 0.]] if moving else [[0., 0., 0.]], cuda_device, False),
        rolling_shutter_time=_tensor([.1 if moving else 0.], cuda_device, False),
        min_elevation=-10., max_elevation=12., min_azimuth=-180., max_azimuth=180.,
        n_elevation_channels=8, azimuth_resolution=1., tile_width=32, tile_height=4,
        packed=False, use_depth_compensation=False)
    assert rendered.shape == (1, 8, 360, 3) and alpha.shape == (1, 8, 360, 1)
    assert torch.isfinite(rendered).all() and torch.isfinite(alpha).all()
    assert alpha[0, 3, 0, 0] > .5 and alpha[0, 3, -1, 0] > .5, "Both sides of seam must be rendered"
    # Despite the upstream docstring, its kernel outputs sum(w*range).
    expected_range = rendered[..., -1]/alpha[..., 0].clamp_min(1e-8)
    if not moving:
        assert torch.allclose(alpha[0, 3, 0], alpha[0, 3, -1], atol=1e-5)
        hits = alpha[..., 0] > .1
        assert torch.allclose(expected_range[hits], torch.full_like(expected_range[hits], 5.), atol=2e-4)
    assert torch.isfinite(alpha_until).all() and alpha_until.max() > .5
    (rendered.square().mean()+alpha.mean()).backward()
    for value in (means, quats, scales, opacities, features):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert means.grad.abs().sum() > 0 and features.grad.abs().sum() > 0
    torch.cuda.synchronize()


@pytest.mark.parametrize("time_sign", [-1., 1.])
def test_official_lidar_explicit_per_beam_capture_times(custom_gsplat, cuda_device, time_sign):
    """A real CUDA range must move by -sensor_speed * the supplied ray time."""
    azimuths = torch.arange(64, device=cuda_device)*2*math.pi/64-math.pi
    elevations = torch.arange(-4, 4, device=cuda_device, dtype=torch.float32).deg2rad()
    rows, columns = torch.meshgrid(torch.arange(8, device=cuda_device),
                                  torch.arange(64, device=cuda_device), indexing="ij")
    # Capture order deliberately alternates inside each tile and across beams.
    offsets = torch.where((rows+columns) % 2 == 0, .02, -.02)*time_sign
    raster, boundaries, resolution = lidar_raster_grid(azimuths, elevations, time_offsets=offsets)
    rendered, alpha, _, _ = custom_gsplat.lidar_rasterization(
        means=_tensor([[5., 0., 0.]], cuda_device, False),
        quats=_tensor([[1., 0., 0., 0.]], cuda_device, False),
        scales=_tensor([[.2, .2, .2]], cuda_device, False),
        opacities=_tensor([.8], cuda_device, False),
        lidar_features=_tensor([[[.6, .2]]], cuda_device, False),
        velocities=None,
        viewmats=torch.eye(4, device=cuda_device)[None],
        raster_pts=raster[..., :4].contiguous(), tile_elevation_boundaries=boundaries,
        linear_velocity=_tensor([[1., 0., 0.]], cuda_device, False),
        rolling_shutter_time=_tensor([.04], cuda_device, False),
        min_elevation=float(boundaries[0]), max_elevation=float(boundaries[-1]),
        min_azimuth=-180., max_azimuth=180., n_elevation_channels=8,
        azimuth_resolution=resolution, tile_width=32, tile_height=8,
        packed=False, use_depth_compensation=False)
    assert torch.isfinite(rendered).all() and torch.isfinite(alpha).all()
    assert alpha[0, 4, 32, 0] > .7
    distance = rendered[0, 4, 32, -1]/alpha[0, 4, 32, 0]
    assert float(distance) == pytest.approx(5.-time_sign*.02, abs=2e-4)
    torch.cuda.synchronize()


@pytest.mark.parametrize("kind", ["camera", "lidar"])
def test_reference_cpu_cuda_agree_and_backward(cuda_device, kind):
    local_means = [[.1, -.1, 4.]] if kind == "camera" else [[4., .1, -.1]]
    cpu_scene = GaussianScene(local_means, [[1., .1, .05, 0.]], [[.2, .3, .15]], [.8], [[.7, .3, .1]])
    gpu_scene = copy.deepcopy(cpu_scene).to(cuda_device)
    if kind == "camera":
        request = CameraRequest("camera", 0., torch.eye(4), torch.tensor([[20., 0., 8.], [0., 20., 8.], [0., 0., 1.]]), 16, 16)
        cpu, gpu = render_camera(cpu_scene, request), render_camera(gpu_scene, request)
        loss = gpu["rgb"].sum()+gpu["depth"].sum()
    else:
        request = LiDARRequest("lidar", 0., torch.eye(4), torch.linspace(-.15, .15, 16), torch.tensor([-.1, -.02, .04, .1]))
        cpu, gpu = render_lidar(cpu_scene, request), render_lidar(gpu_scene, request)
        loss = gpu["range"].sum()+gpu["hit_probability"].sum()
    for key in cpu:
        assert torch.allclose(cpu[key], gpu[key].cpu(), atol=2e-5, rtol=2e-5), key
    loss.backward()
    assert gpu_scene.means.grad is not None and torch.isfinite(gpu_scene.means.grad).all()


def test_camera_positive_y_angular_velocity_moves_late_rows_left(custom_gsplat, cuda_device):
    """Independent geometric sign check: rotating the camera right moves a static point left."""
    kwargs = dict(means=_tensor([[0., 1.5, 5.]], cuda_device, False),
        quats=_tensor([[1., 0., 0., 0.]], cuda_device, False),
        scales=_tensor([[.12, .12, .12]], cuda_device, False),
        opacities=_tensor([.8], cuda_device, False), colors=_tensor([[1., .2, .1]], cuda_device, False),
        viewmats=torch.eye(4, device=cuda_device)[None],
        Ks=_tensor([[[40., 0., 32.], [0., 40., 32.], [0., 0., 1.]]], cuda_device, False),
        width=64, height=64, packed=False, render_mode="RGB", velocities=None,
        linear_velocity=torch.zeros(1, 3, device=cuda_device), rolling_shutter_direction=1)
    def alpha(omega, duration):
        _, value, _ = custom_gsplat.rasterization(**kwargs,
            angular_velocity=_tensor([[0., omega, 0.]], cuda_device, False),
            rolling_shutter_time=_tensor([duration], cuda_device, False))
        assert torch.isfinite(value).all()
        return value[0, ..., 0]
    zero = alpha(0., .2)
    positive, negative = alpha(.5, .2), alpha(-.5, .2)
    x = torch.arange(64, device=cuda_device)[None] + .5
    center = lambda a: float((a*x).sum()/a.sum())
    assert center(positive) < center(zero)-.4
    assert center(negative) > center(zero)+.4
    torch.testing.assert_close(alpha(.5, 0.), alpha(0., 0.), atol=0, rtol=0)


@pytest.mark.parametrize("offset", [-.04, .04])
def test_lidar_z_angular_velocity_shifts_azimuth_by_minus_omega_time(custom_gsplat, cuda_device, offset):
    """A static target's local bearing changes by -omega*dt, in real CUDA output."""
    azimuth = torch.arange(1440, device=cuda_device)*2*math.pi/1440-math.pi
    elevation = torch.arange(-4, 4, device=cuda_device, dtype=torch.float32).deg2rad()
    raster, boundaries, resolution = lidar_raster_grid(azimuth, elevation,
        time_offsets=torch.full((1440,), offset, device=cuda_device))
    kwargs = dict(means=_tensor([[5., 5., 0.]], cuda_device, False),
        quats=_tensor([[1., 0., 0., 0.]], cuda_device, False),
        scales=_tensor([[.2, .2, .2]], cuda_device, False),
        opacities=_tensor([.8], cuda_device, False),
        lidar_features=_tensor([[[.6, .2]]], cuda_device, False), velocities=None,
        viewmats=torch.eye(4, device=cuda_device)[None],
        raster_pts=raster[..., :4].contiguous(), tile_elevation_boundaries=boundaries,
        linear_velocity=torch.zeros(1, 3, device=cuda_device),
        rolling_shutter_time=_tensor([.08], cuda_device, False),
        min_elevation=float(boundaries[0]), max_elevation=float(boundaries[-1]),
        min_azimuth=-180., max_azimuth=180., n_elevation_channels=8,
        azimuth_resolution=resolution, tile_width=32, tile_height=8,
        packed=False, use_depth_compensation=False)
    centers = []
    for omega in (0., .5):
        _, alpha, _, _ = custom_gsplat.lidar_rasterization(**kwargs,
            angular_velocity=_tensor([[0., 0., omega]], cuda_device, False))
        assert torch.isfinite(alpha).all() and alpha.max() > .5
        weights = alpha[0, 4, :1440, 0]
        centers.append(float((weights*azimuth.rad2deg()).sum()/weights.sum()))
    assert centers[1]-centers[0] == pytest.approx(-.5*offset*180/math.pi, abs=.08)
