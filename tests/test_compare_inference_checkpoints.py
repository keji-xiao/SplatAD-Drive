"""CPU tests for comparison decisions and reporting, with no official model load."""
import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "compare_inference_checkpoints", Path(__file__).parents[1] / "scripts/compare_inference_checkpoints.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("comparison unit test attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


def test_exact_and_tolerance_pass_are_distinct():
    reference = torch.tensor([0., 1., 100.])
    exact = compare.compare_tensor(reference, reference.clone())
    assert exact["status"] == "PASS" and exact["exact_equal"] and exact["unequal_count"] == 0
    close = compare.compare_tensor(reference, reference+torch.tensor([0., 1e-6, 1e-4]))
    assert close["status"] == "PASS" and not close["exact_equal"] and close["unequal_count"] == 2
    assert close["outside_tolerance_count"] == 0


def test_tolerance_uses_reference_and_absolute_bands_do_not_override_fail():
    reference, candidate = torch.tensor([1.], dtype=torch.float64), torch.tensor([1.11], dtype=torch.float64)
    assert compare.compare_tensor(reference, candidate, atol=0., rtol=.1)["status"] == "FAIL"
    assert compare.compare_tensor(candidate, reference, atol=0., rtol=.1)["status"] == "PASS"
    result = compare.compare_tensor(torch.zeros(1), torch.tensor([1e-4]))
    assert result["status"] == "FAIL" and result["absolute_only_bands"]["0.001"]["within_count"] == 1


def test_matching_nonfinite_values_always_fail_and_are_excluded_from_metrics():
    values = torch.tensor([float("nan"), float("inf"), 2.])
    result = compare.compare_tensor(values, values.clone())
    assert result["status"] == "FAIL" and result["nonfinite_pair_count"] == 2
    assert result["outside_tolerance_count"] == 2 and result["finite_pair_count"] == 1
    assert result["max_absolute_error"] == result["mean_absolute_error"] == 0
    only_invalid = compare.compare_tensor(values[:2], values[:2])
    assert only_invalid["max_absolute_error"] is None


def test_shape_dtype_and_key_mismatches_fail():
    value = torch.ones(2)
    assert compare.compare_tensor(value, torch.ones(3))["shape_equal"] is False
    assert compare.compare_tensor(value, value.double())["status"] == "FAIL"
    extra = compare.compare_outputs({"rgb": value}, {"rgb": value, "extra": value}, sensor="camera", atol=1e-5, rtol=1e-5)
    missing = compare.compare_outputs({"rgb": value}, {}, sensor="camera", atol=1e-5, rtol=1e-5)
    assert extra["status"] == missing["status"] == "FAIL"
    assert extra["extra_candidate_keys"] == ["extra"] and missing["missing_candidate_keys"] == ["rgb"]


def test_distance_component_and_point_vector_metrics_are_separate():
    result = compare.compare_tensor(torch.zeros(2, 3), torch.tensor([[.03, .04, 0.], [0., 0., 2.]]), distance=True, points=True)
    assert result["distance_error_metres"]["finite_pairs_over_1m"] == 1
    assert result["point_l2_error_metres"]["mean"] == pytest.approx(1.025)
    assert result["point_l2_error_metres"]["finite_points_over_1cm"] == 2


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_tolerances_rejected(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare.compare_tensor(torch.ones(1), torch.ones(1), atol=bad)


@pytest.fixture
def args(tmp_path):
    requests_root = tmp_path / "requests"
    for index in range(2):
        destination = requests_root / f"sample_{index:03d}"
        destination.mkdir(parents=True)
        pair = {"camera": {"camera_id": 0, "timestamp": index+.01,
                           "T_world_camera": torch.eye(4).tolist(), "K": torch.eye(3).tolist(),
                           "width": 2, "height": 2, "rolling_shutter_duration": .03},
                "lidar": {"lidar_id": 0, "timestamp": index+.02, "T_world_lidar": torch.eye(4).tolist(),
                          "azimuth": [0., 1.], "elevation": [0.], "scan_duration": .1, "time_offsets": [-.04, .06]}}
        (destination / "requests.json").write_text(json.dumps(pair), encoding="utf-8")
    files = {}
    for phase in ("reference", "candidate"):
        destination = tmp_path / phase
        destination.mkdir()
        for suffix, name in (("config", "config.yml"), ("checkpoint", "step-000029999.ckpt")):
            path = destination / name
            path.write_text("CPU fixture", encoding="utf-8")
            files[phase+"_"+suffix] = path
    return SimpleNamespace(**files, requests_root=requests_root, expected_native_count=2,
                           output=tmp_path / "report.json", atol=1e-5, rtol=1e-5)


def test_discovery_and_zero_rs_preserve_timestamp_and_other_requests(args):
    cases = compare.discover_cases(args.requests_root, expected_native_count=2)
    assert len(cases) == 4 and [case["zero_rs"] for case in cases] == [False, False, True, True]
    native, zero = compare.case_requests(cases[0], "cpu"), compare.case_requests(cases[2], "cpu")
    assert native["camera"].timestamp == zero["camera"].timestamp
    assert native["lidar"].timestamp == zero["lidar"].timestamp
    assert zero["camera"].rolling_shutter_duration == zero["lidar"].scan_duration == 0
    assert zero["lidar"].time_offsets is None and native["lidar"].time_offsets is not None
    with pytest.raises(ValueError, match="Expected 20"):
        compare.discover_cases(args.requests_root)


@pytest.mark.parametrize("mode,expected_status,failed,errors", [
    ("pass", "PASS", 0, 0), ("fail", "FAIL", 4, 0), ("error", "FAIL", 0, 4), ("wrong_checkpoint", "FAIL", 0, 8)])
def test_phases_release_models_and_save_complete_failure_reports(args, mode, expected_status, failed, errors):
    references = []

    class Backend:
        def __init__(self, phase):
            self.phase, self.step, self.device = phase, 29999, "cpu"
            self.checkpoint_path = getattr(args, phase+"_checkpoint")
            if phase == "candidate" and mode == "wrong_checkpoint":
                self.checkpoint_path = self.checkpoint_path.with_name("step-000000001.ckpt")
            self.model = SimpleNamespace(gauss_params={"means": torch.zeros(2, 3)})

        def render_camera(self, request):
            if self.phase == "candidate" and mode == "error":
                raise RuntimeError("intentional CPU render failure")
            delta = (1. if mode == "fail" else 1e-6) if self.phase == "candidate" else 0.
            return {key: torch.tensor([1.+delta]) for key in ("rgb", "alpha", "depth")}

        def render_lidar(self, request):
            return {key: torch.ones(1) for key in ("range", "alpha", "ray_drop", "median_range", "intensity", "points")}

    def factory(config, load_dir):
        gc.collect()
        assert all(reference() is None for reference in references), "two models simultaneously alive"
        backend = Backend(Path(config).parent.name)
        references.append(weakref.ref(backend))
        return backend

    report = compare.run_comparison(args, factory)
    assert report["status"] == expected_status and report["summary"]["sensor_comparisons"] == 8
    assert report["summary"]["failed"] == failed and report["summary"]["errors"] == errors
    assert all(reference() is None for reference in references)
    assert json.loads(args.output.read_text(encoding="utf-8"))["status"] == expected_status
    if mode == "pass":
        assert report["summary"]["all_compared_tensors_exact_equal"] is False
    if mode == "wrong_checkpoint":
        assert "Loaded" in report["models"]["candidate"]["traceback"]


def test_report_cannot_overwrite_input(args):
    before = args.reference_config.read_bytes()
    args.output = args.reference_config
    with pytest.raises(ValueError, match="must not overwrite"):
        compare.run_comparison(args)
    assert args.reference_config.read_bytes() == before


def test_main_returns_nonzero_and_saves_setup_failure(args, monkeypatch):
    arguments = ["compare_inference_checkpoints.py"]
    for name in ("reference_config", "candidate_config", "reference_checkpoint", "candidate_checkpoint", "requests_root", "output"):
        arguments.extend(["--"+name.replace("_", "-"), str(getattr(args, name))])
    # Deliberately leave default expected-native-count=20 for this two-case fixture.
    monkeypatch.setattr(compare.sys, "argv", arguments)
    assert compare.main() == 1
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["status"] == "ERROR" and "Expected 20" in report["errors"][0]
