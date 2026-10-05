"""CPU contract tests: these do not claim a CUDA checkpoint was executed."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from splatad_drive.backends.neurad import (NeuradBackend, cv_to_nerfstudio,
                                          lidar_raster_grid, _with_clip_planes)
from splatad_drive.data.pandaset import export_parser_audit, project_lidar_points, se3_statistics


def test_camera_conversion_preserves_world_and_positive_cv_depth():
    T_world_camera = torch.eye(4)
    T_world_camera[:3, 3] = torch.tensor([3., 4., 1.])
    gl = cv_to_nerfstudio(T_world_camera)
    assert torch.equal(cv_to_nerfstudio(gl), T_world_camera)
    point = np.array([[0., 0., 5.]])
    K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    uv, depth, valid = project_lidar_points(point, T_world_camera, gl, K, 100, 100)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(uv, [[50., 50.]])
    np.testing.assert_allclose(depth, [5.])
    assert se3_statistics(gl)["status"] == "PASS"


def test_se3_check_rejects_affine_scale_even_though_matrix_inverse_exists():
    transform = np.eye(4)
    transform[0, 0] = 2
    assert se3_statistics(transform)["status"] == "FAIL"


def test_se3_float32_roundoff_at_driving_scale():
    transform = np.eye(4, dtype=np.float32)
    angle = .731
    transform[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    transform[:3, 3] = [80., -120., 2.]
    result = se3_statistics(transform)
    assert result["status"] == "PASS"
    assert result["input_dtype"] == "float32"
    transform[0, 0] *= 1.01
    assert se3_statistics(transform)["status"] == "FAIL"


def test_lidar_tile_padding_and_nonzero_point_capture_times():
    azimuth = torch.arange(1800)*2*math.pi/1800-math.pi
    elevation = torch.linspace(-.25, .15, 64)
    raster, boundaries, resolution = lidar_raster_grid(azimuth, elevation, .1)
    assert raster.shape == (1, 64, 1824, 5)
    assert boundaries.shape == (9,)
    assert resolution == pytest.approx(.2, abs=2e-5)
    assert float(raster[0, :, :1800, 3].mean()) == pytest.approx(0, abs=1e-6)
    assert float(raster[..., 3].max()-raster[..., 3].min()) > .099
    assert torch.all(boundaries.diff() > 0)
    with pytest.raises(ValueError, match="full 360"):
        lidar_raster_grid(torch.linspace(-1., 1., 64), elevation)


@pytest.mark.parametrize("per_beam", [False, True])
def test_lidar_explicit_times_preserve_origin_order_and_pad_last_ray(per_beam):
    azimuth = torch.arange(36)*2*math.pi/36-math.pi
    elevation = torch.linspace(-.25, .15, 8)
    offsets = torch.linspace(.03, .01, 36)
    if per_beam:
        offsets = offsets[None].repeat(8, 1)+torch.arange(8)[:, None]*.0001
    original = offsets.clone()
    raster, _, _ = lidar_raster_grid(azimuth, elevation, 100., time_offsets=offsets)
    expected = offsets if per_beam else offsets[None].expand(8, -1)
    assert torch.equal(raster[0, :, :36, 3], expected)
    assert torch.equal(raster[0, :, 36:, 3], expected[:, -1:].expand(-1, 28))
    assert torch.equal(offsets, original)
    assert raster[..., 3].min() > 0, "Explicit times must not be recentered before the upstream pose/time shift"


@pytest.mark.parametrize("offsets", [torch.zeros(35), torch.zeros(8, 1), torch.full((36,), float("nan")),
                                    torch.full((8, 36), float("inf"))])
def test_lidar_raster_explicit_times_reject_bad_shape_or_nonfinite(offsets):
    azimuth = torch.arange(36)*2*math.pi/36-math.pi
    with pytest.raises(ValueError, match="time_offsets"):
        lidar_raster_grid(azimuth, torch.linspace(-.2, .1, 8), time_offsets=offsets)


def test_pandaset_native_phase_forward_zero_and_positive_x_early():
    path = Path(__file__).resolve().parents[1]/"scripts"/"validate_checkpoint.py"
    spec = importlib.util.spec_from_file_location("validate_checkpoint_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    angles = torch.tensor([-math.pi, -math.pi/2, 0., math.pi/2-.01, math.pi/2+.01])
    offsets = module.pandaset_native_time_offsets(angles, .1)
    assert offsets[0] == pytest.approx(.025, abs=1e-8)
    assert offsets[1] == pytest.approx(0., abs=1e-8)  # native -Y forward
    assert offsets[2] == pytest.approx(-.025, abs=1e-8)  # native +X
    assert offsets[3] < -.049 and offsets[4] > .049  # physical wrap is at native +Y
    with pytest.raises(ValueError, match="scan period"):
        module.pandaset_native_time_offsets(angles, float("nan"))


def rasterization(**kwargs):
    return kwargs


class ClipModel:
    def render(self, sensor):
        return rasterization(sensor=sensor, near_plane=.5, far_plane=1e10)


def test_clip_plane_override_never_modifies_upstream_module_globals():
    model = ClipModel()
    assert _with_clip_planes(model.render, "camera", "rasterization", .1, 25)["far_plane"] == 25
    assert model.render("camera")["far_plane"] == 1e10
    assert ClipModel.render.__globals__["rasterization"] is rasterization


class FakeActors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_actors = 2

    def get_boxes2world(self, times, flatten=False):
        transforms = torch.eye(4).repeat(times.numel(), 2, 1, 1)
        transforms[:, 0, 0, 3] = times.reshape(-1)
        return transforms, torch.ones(times.numel(), 2, dtype=torch.bool)

    def get_velocities(self, times):
        velocities = torch.zeros(times.numel(), 2, 6)
        velocities[:, 0, 0] = 1
        return velocities


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamic_actors = FakeActors()
        self.gauss_params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.zeros(3, 3)),
            "id": torch.nn.Parameter(torch.tensor([[0.], [1.], [2.]]), requires_grad=False),
            "quats": torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)),
            "opacities": torch.nn.Parameter(torch.ones(3, 1))})

    def get_camera_outputs(self, camera):
        return {}

    def get_lidar_outputs(self, lidar):
        return {}


def test_actor_edit_is_anchored_moves_velocity_and_restores_after_exception():
    model = FakeModel()
    backend = NeuradBackend(SimpleNamespace(model=model))
    target = torch.tensor([[0., -1., 0., 10.], [1., 0., 0., 20.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
    backend.set_actor_pose(0, 2., target)
    backend.set_actor_visible(1, False)
    before = {name: value.detach().clone() for name, value in model.gauss_params.items()}
    with pytest.raises(RuntimeError, match="raster failure"):
        with backend._edit_context():
            anchored, _ = model.dynamic_actors.get_boxes2world(torch.tensor([2.]), flatten=False)
            assert torch.allclose(anchored[0, 0], target)
            assert torch.allclose(model.dynamic_actors.get_velocities(torch.tensor([2.]))[0, 0, :3], torch.tensor([0., 1., 0.]))
            assert model.gauss_params["opacities"][1].sigmoid() == 0
            assert torch.equal(model.gauss_params["opacities"][2], before["opacities"][2])
            assert not torch.equal(model.gauss_params["quats"][0], before["quats"][0])
            raise RuntimeError("raster failure")
    assert "get_boxes2world" not in model.dynamic_actors.__dict__
    assert "get_velocities" not in model.dynamic_actors.__dict__
    for key, value in before.items():
        assert torch.equal(model.gauss_params[key], value)
    with pytest.raises(ValueError, match="Unknown dynamic actor"):
        backend.set_actor_visible(2, False)


def test_audit_does_not_turn_unreviewed_overlays_into_pass(tmp_path):
    from PIL import Image
    Image.new("RGB", (16, 16)).save(tmp_path / "frame.png")
    camera_gl = torch.eye(4)[:3].clone()
    camera_gl[:, 1:3] *= -1
    cameras = SimpleNamespace(camera_to_worlds=camera_gl[None], times=torch.tensor([[0.]]),
        metadata={"sensor_idxs": torch.tensor([[0]])}, width=torch.tensor([[16]]), height=torch.tensor([[16]]),
        fx=torch.tensor([[10.]]), fy=torch.tensor([[10.]]), cx=torch.tensor([[8.]]), cy=torch.tensor([[8.]]))
    lidars = SimpleNamespace(lidar_to_worlds=torch.eye(4)[None, :3], times=torch.tensor([[0.]]),
        metadata={"sensor_idxs": torch.tensor([[1]])}, valid_lidar_distance_threshold=1000.)
    outputs = SimpleNamespace(cameras=cameras, image_filenames=[tmp_path / "frame.png"], time_offset=100.,
        dataparser_transform=torch.eye(4)[:3], dataparser_scale=1., metadata={"lidars": lidars,
        "point_clouds": [torch.tensor([[0., 0., 5., .5, -.02], [0., 0., 1500., 0., .03]])],
        "trajectories": [], "sensor_idx_to_name": {0: "front", 1: "Pandar64"}})
    report = export_parser_audit(outputs, tmp_path / "audit", max_frames=1)
    assert report["numeric_status"] == "PASS"
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["decision"] == "NO-GO"
    assert report["overlay_count"] == 1
    assert (tmp_path / "audit" / "lidar_camera_overlay" / "front_0000.png").is_file()
    points = np.load(tmp_path / "audit" / "lidar_0000.npz")
    np.testing.assert_allclose(points["points"][:, 4], [-.02, .03])


def test_audit_reads_batched_column_intrinsics():
    from splatad_drive.data.pandaset import camera_intrinsics
    cameras = SimpleNamespace(camera_to_worlds=torch.eye(4)[None, :3].repeat(2, 1, 1),
        fx=torch.tensor([[10.], [20.]]), fy=torch.tensor([[11.], [21.]]),
        cx=torch.tensor([[8.], [9.]]), cy=torch.tensor([[6.], [7.]]))
    np.testing.assert_allclose(camera_intrinsics(cameras),
        [[[10., 0., 8.], [0., 11., 6.], [0., 0., 1.]],
         [[20., 0., 9.], [0., 21., 7.], [0., 0., 1.]]])


def test_pinned_official_source_contract_when_checkout_is_present():
    root = Path(__file__).resolve().parents[1] / "third_party" / "neurad-studio" / "nerfstudio"
    model_path = root / "models" / "splatad.py"
    if not model_path.is_file():
        pytest.skip("Pinned upstream source has not been fetched; fake contract tests are not GPU integration tests")
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SplatADModel")
    methods = {node.name for node in model.body if isinstance(node, ast.FunctionDef)}
    assert {"get_camera_outputs", "get_lidar_outputs", "_get_actor_adjusted_means"} <= methods
    assert "lidar_rasterization" in model_path.read_text(encoding="utf-8")
    parser = ast.parse((root / "data" / "dataparsers" / "pandaset_dataparser.py").read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.ClassDef) and node.name == "PandaSetDataParserConfig" for node in parser.body)
