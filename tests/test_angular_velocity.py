"""CPU public-request contracts; mocked sensor classes are not CUDA evidence."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from splatad_drive.backends.neurad import NeuradBackend
from splatad_drive.camera_renderer import CameraRequest
from splatad_drive.cli import _load_requests
from splatad_drive.io import save_json
from splatad_drive.lidar_renderer import LiDARRequest
from splatad_drive.reference import ReferenceBackend
from splatad_drive.scene import GaussianScene


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU angular contract attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def request(kind, **kwargs):
    if kind == "camera":
        base = CameraRequest("front", 0., torch.eye(4),
                             torch.tensor([[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]]), 16, 16)
    else:
        base = LiDARRequest("Pandar64", 0., torch.eye(4),
                            torch.arange(64)*2*math.pi/64-math.pi, torch.linspace(-.2, .2, 8))
    return replace(base, **kwargs)


@pytest.mark.parametrize("kind", ["camera", "lidar"])
@pytest.mark.parametrize("angular", [[float("nan"), 0., 0.], [0., float("inf"), 0.], [1., 2.], [[1., 2., 3.]]])
def test_angular_request_rejects_nonfinite_or_wrong_shape(kind, angular):
    with pytest.raises(ValueError, match="angular_velocity"):
        request(kind, angular_velocity=angular)


@pytest.mark.parametrize("kind", ["camera", "lidar"])
def test_reference_explicit_zero_matches_default_and_nonzero_is_rejected(kind):
    scene = GaussianScene([[2., 0., 3.]], [[1., 0., 0., 0.]], [[.3, .3, .3]], [.8], [[.6, .2, .1]])
    backend = ReferenceBackend(scene)
    render = getattr(backend, "render_" + kind)
    default, zero = render(request(kind)), render(request(kind, angular_velocity=torch.zeros(3)))
    for name in default:
        torch.testing.assert_close(default[name], zero[name], rtol=0, atol=0)
    # Reject explicitly even when duration is zero, rather than implying support.
    with pytest.raises(NotImplementedError, match="nonzero angular_velocity"):
        render(request(kind, angular_velocity=[0., .1, 0.]))


def test_cli_loads_optional_sensor_local_angular_tensors_and_legacy_requests(tmp_path):
    path = tmp_path / "requests.json"
    values = {kind: asdict(request(kind, angular_velocity=[.1, -.2, .3])) for kind in ("camera", "lidar")}
    save_json(path, values)
    loaded = _load_requests(path, "cpu")
    for sensor in loaded.values():
        assert isinstance(sensor.angular_velocity, torch.Tensor)
        torch.testing.assert_close(sensor.angular_velocity, torch.tensor([.1, -.2, .3]))
    for sensor in values.values():
        del sensor["angular_velocity"]
    save_json(path, values)
    assert all(sensor.angular_velocity is None for sensor in _load_requests(path, "cpu").values())


class FakeSensor(SimpleNamespace):
    def __getitem__(self, index):
        return self

    def to(self, device):
        return self

    def get_intrinsics_matrices(self):
        return torch.tensor([[[30., 0., 16.], [0., 30., 8.], [0., 0., 1.]]])


def fake_sensor_modules(monkeypatch):
    for name in ("nerfstudio", "nerfstudio.cameras"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    cameras = ModuleType("nerfstudio.cameras.cameras")
    cameras.Cameras = FakeSensor
    cameras.CameraType = SimpleNamespace(PERSPECTIVE=1)
    lidars = ModuleType("nerfstudio.cameras.lidars")
    lidars.Lidars = FakeSensor
    lidars.LidarType = type("FakeLidarType", (), {"PANDAR64": 0, "__new__": lambda cls, value: value})
    lidars.get_lidar_elevation_mapping = lambda value: {i: float(i-3) for i in range(8)}
    monkeypatch.setitem(sys.modules, cameras.__name__, cameras)
    monkeypatch.setitem(sys.modules, lidars.__name__, lidars)


def capturing_backend(monkeypatch):
    fake_sensor_modules(monkeypatch)
    backend = object.__new__(NeuradBackend)
    backend.model = SimpleNamespace(gauss_params={"means": torch.zeros(1, 3)},
                                    appearance_embedding=SimpleNamespace(num_embeddings=2),
                                    get_camera_outputs=lambda camera: None, get_lidar_outputs=lambda lidar: None)
    backend.sensor_idx_to_name = {0: "front", 1: "Pandar64"}
    backend._lidar_sensor_ids = [1]
    backend._edit_context = lambda scene: nullcontext()
    backend._request_config = lambda **kwargs: nullcontext()
    captured = {}

    def render(method, sensor, rasterizer_name, near, far):
        captured[rasterizer_name] = sensor
        if rasterizer_name == "rasterization":
            plane = torch.ones(sensor.height, sensor.width, 1)
            return {"rgb": plane.expand(-1, -1, 3), "depth": plane, "accumulation": plane}
        shape = sensor.metadata["raster_pts"].shape[:-1] + (1,)
        return {name: torch.ones(shape) for name in ("accumulation", "depth", "ray_drop_prob", "intensity", "median_depth")}

    monkeypatch.setattr("splatad_drive.backends.neurad._with_clip_planes", render)
    return backend, captured


def test_official_camera_converts_cv_angular_to_gl_metadata_without_world_rotation(monkeypatch):
    backend, captured = capturing_backend(monkeypatch)
    angular = torch.tensor([.2, -.3, .4])
    original = angular.clone()
    pose = torch.tensor([[0., -1., 0., 3.], [1., 0., 0., 4.], [0., 0., 1., 1.], [0., 0., 0., 1.]])
    backend.render_camera(request("camera", T_world_camera=pose, angular_velocity=angular,
                                  linear_velocity=[1., 2., 3.], rolling_shutter_duration=.03))
    metadata = captured["rasterization"].metadata
    torch.testing.assert_close(metadata["angular_velocities_local"], torch.tensor([[.2, .3, -.4]]))
    torch.testing.assert_close(metadata["linear_velocities_local"], torch.tensor([[1., -2., -3.]]))
    # The upstream camera model multiplies GL metadata by this same rotation.
    torch.testing.assert_close(metadata["angular_velocities_local"] * torch.tensor([1., -1., -1.]), angular[None])
    assert torch.equal(angular, original)


def test_official_lidar_keeps_native_local_angular_axes_and_zero_default(monkeypatch):
    backend, captured = capturing_backend(monkeypatch)
    angular = torch.tensor([.2, -.3, .4])
    backend.render_lidar(request("lidar", angular_velocity=angular, scan_duration=.1))
    torch.testing.assert_close(captured["lidar_rasterization"].metadata["angular_velocities_local"], angular[None])
    for kind, key in (("camera", "rasterization"), ("lidar", "lidar_rasterization")):
        getattr(backend, "render_" + kind)(request(kind))
        assert torch.equal(captured[key].metadata["angular_velocities_local"], torch.zeros(1, 3))


@pytest.mark.parametrize("angular", [[float("nan"), 0., 0.], [[0., 0., 0.]], [1., 2.]])
def test_official_metadata_rechecks_mutable_request_values(monkeypatch, angular):
    backend, _ = capturing_backend(monkeypatch)
    with pytest.raises(ValueError, match="angular_velocity"):
        backend._metadata(0, None, angular)


@pytest.mark.parametrize("rolling_shutter", [False, True])
def test_checkpoint_requests_use_native_angular_metadata_and_cv_conversion(monkeypatch, rolling_shutter):
    fake_sensor_modules(monkeypatch)
    path = Path(__file__).parents[1] / "scripts/validate_checkpoint.py"
    spec = importlib.util.spec_from_file_location("validate_angular_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_pandaset_scan_period", lambda backend: .1)
    camera = FakeSensor(device=torch.device("cpu"), camera_to_worlds=torch.eye(4)[None, :3],
                        times=torch.tensor([[4.]]), width=torch.tensor([32]), height=torch.tensor([16]),
                        metadata={"sensor_idxs": torch.tensor([[0]]), "time_to_center_pixel": torch.tensor([[.01]]),
                                  "rolling_shutter_time": torch.tensor([[.03]]),
                                  "linear_velocities_local": torch.tensor([[1., 2., 3.]]),
                                  "angular_velocities_local": torch.tensor([[.2, .3, .4]])})
    lidar = FakeSensor(device=torch.device("cpu"), lidar_to_worlds=torch.eye(4)[None, :3],
                       times=torch.tensor([[4.]]), lidar_type=torch.tensor([0]),
                       metadata={"sensor_idxs": torch.tensor([[1]]),
                                 "linear_velocities_local": torch.tensor([[4., 5., 6.]]),
                                 "angular_velocities_local": torch.tensor([[.5, -.6, .7]])})
    image_data = SimpleNamespace(cameras=camera, image_filenames=[Path("frame.png")])
    lidar_data = SimpleNamespace(lidars=lidar, point_clouds=[torch.tensor([[5., 0., 0., .8, -.02], [5., 0., 0., .8, .02]])])
    backend = SimpleNamespace(device=torch.device("cpu"), sensor_idx_to_name={0: "front", 1: "Pandar64"})
    args = SimpleNamespace(width=32, height=16, azimuth_count=64, rolling_shutter=rolling_shutter)
    camera_request, lidar_request, manifest = module.make_requests(backend, image_data, lidar_data, 0, args)
    torch.testing.assert_close(camera_request.angular_velocity, torch.tensor([.2, -.3, -.4]))
    torch.testing.assert_close(lidar_request.angular_velocity, torch.tensor([.5, -.6, .7]))
    torch.testing.assert_close(camera.metadata["angular_velocities_local"], torch.tensor([[.2, .3, .4]]))
    assert "camera_angular_velocity_local_ignored" not in manifest
    assert "camera_angular_velocity_local_cv_rad_s" in manifest
    assert camera_request.rolling_shutter_duration == pytest.approx(.03 if rolling_shutter else 0.)
