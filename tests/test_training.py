import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch

from splatad_drive.training import build_train_command, load_training_config, run_training, validate_review_scope


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("check_training_run", ROOT / "scripts/check_training_run.py")
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


def config():
    return load_training_config(ROOT / "configs/smoke.yaml")


@pytest.mark.parametrize("key,value", [("iterations", True), ("seed", -1), ("sequence", 28), ("dataset_end_fraction", float("nan")), ("dataset_end_fraction", True), ("cameras", "front"), ("cameras", ["front", "front"]), ("name", "../old_run"), ("data", None), ("vis", "viewer")])
def test_config_invalid_types_fail_before_launch(key, value):
    candidate = config()
    candidate[key] = value
    with pytest.raises(ValueError):
        build_train_command(candidate)


def test_explicit_timestamp_is_trainer_option():
    command = build_train_command(config(), root=ROOT, timestamp="test_run")
    assert command.index("--timestamp") < command.index("pandaset-data")
    assert command[command.index("--timestamp") + 1] == "test_run"


def test_smoke_review_scope_allows_smoke_but_not_larger_experiments():
    review = {"training_scope": {"iterations_max": 300, "cameras": ["front"], "dataset_end_fraction_max": 0.2}}
    validate_review_scope(config(), review)
    validate_review_scope(config(), {})
    for key, value in (("iterations", 301), ("cameras", ["front", "back"]), ("dataset_end_fraction", 1.0)):
        candidate = config()
        candidate[key] = value
        with pytest.raises(ValueError, match="training_scope"):
            validate_review_scope(candidate, review)


@pytest.mark.parametrize("scope", [False, {"iterations_max": True}, {"cameras": "front"}, {"dataset_end_fraction_max": float("nan")}])
def test_invalid_review_scope_fails_closed(scope):
    with pytest.raises(ValueError):
        validate_review_scope(config(), {"training_scope": scope})


def make_audit(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"data": "data/pandaset", "sequence": "028", "numeric_checks_passed": True}))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(), "approved": True, "reviewer": "test reviewer"}))
    return audit, review


def test_launches_are_isolated_and_failure_is_recorded(tmp_path, monkeypatch):
    import splatad_drive.environment as environment
    import splatad_drive.training as training

    monkeypatch.setattr(environment, "environment_report", lambda *args, **kwargs: {"official_training_ready": True})

    class Process:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO("fixture subprocess log\n")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def wait(self):
            return 3

    monkeypatch.setattr(training.subprocess, "Popen", Process)
    audit, review = make_audit(tmp_path)
    for _ in range(2):
        assert run_training(config(), root=tmp_path, audit_path=audit, alignment_review=review) == 3
    launches = sorted(tmp_path.rglob("launch.json"))
    assert len(launches) == 2 and launches[0].parent != launches[1].parent
    for launch_path in launches:
        launch = json.loads(launch_path.read_text())
        completion = json.loads((launch_path.parent / "completion.json").read_text())
        assert launch["expected_final_step"] == 299
        assert launch["command"][launch["command"].index("--timestamp") + 1] == launch_path.parent.name
        assert completion["returncode"] == 3
        assert completion["status"] == "failed"
        assert "fixture" in (launch_path.parent / "training.log").read_text()


def make_checkpoint(directory, step=19, *, nonfinite=False):
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = {"_model.gauss_params." + name: torch.zeros(2, width) for name, width in (
        ("means", 3), ("scales", 3), ("quats", 4), ("features_dc", 3), ("features_rest", 13), ("opacities", 1), ("id", 1))}
    if nonfinite:
        pipeline["_model.gauss_params.means"][0, 0] = float("nan")
    state = {"step": step, "pipeline": pipeline, "optimizers": {"means": {"state": {}, "param_groups": [{"params": [0]}]}}, "schedulers": {}, "scalers": {}}
    path = directory / f"step-{step:09d}.ckpt"
    torch.save(state, path)
    return path


def test_checkpoint_payload_and_filename_are_checked(tmp_path):
    good = make_checkpoint(tmp_path, 19)
    assert checker.checkpoint_evidence(good)["completed_iterations"] == 20
    renamed = good.with_name("step-000000020.ckpt")
    good.rename(renamed)
    with pytest.raises(ValueError, match="filename"):
        checker.checkpoint_evidence(renamed)
    bad = make_checkpoint(tmp_path, 21, nonfinite=True)
    with pytest.raises(ValueError, match="nonfinite"):
        checker.checkpoint_evidence(bad)


def test_checkpoint_accepts_official_numpy_scheduler_scalars(tmp_path):
    import numpy as np
    good = make_checkpoint(tmp_path, 19)
    state = torch.load(good, weights_only=True)
    state["schedulers"] = {"means": {"last_lr": np.float64(0.001)}}
    torch.save(state, good)
    assert checker.checkpoint_evidence(good)["completed_iterations"] == 20


def test_log_forwarding_survives_gbk_emoji(tmp_path, monkeypatch):
    import splatad_drive.environment as environment
    import splatad_drive.training as training
    monkeypatch.setattr(environment, "environment_report", lambda *args, **kwargs: {"official_training_ready": True})
    class Process:
        def __init__(self, *args, **kwargs): self.stdout = io.StringIO("Training complete \U0001f389\n")
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def wait(self): return 0
    stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
    monkeypatch.setattr(training.subprocess, "Popen", Process)
    monkeypatch.setattr(training.sys, "stdout", stream)
    audit, review = make_audit(tmp_path)
    assert run_training(config(), root=tmp_path, audit_path=audit, alignment_review=review) == 0
    run = next(tmp_path.rglob("launch.json")).parent
    assert "\U0001f389" in (run / "training.log").read_text(encoding="utf-8")
    assert json.loads((run / "completion.json").read_text())["returncode"] == 0


def make_events(directory, *, missing=None, last_step=10):
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(log_dir=directory) as writer:
        for name in checker.REQUIRED_LOSSES:
            if name != missing:
                writer.add_scalar("Train Loss Dict/" + name, 0.1, last_step)
        writer.add_scalar("Train Loss", 0.4, last_step)
        writer.add_scalar("Train Metrics Dict/gaussian_count", 2, last_step)


def test_checker_needs_complete_checkpoint_and_joint_losses(tmp_path):
    make_checkpoint(tmp_path / "nerfstudio_models", step=19)
    make_events(tmp_path)
    report = checker.inspect_run(tmp_path, expected_iterations=20)
    assert report["status"] == "PASS", report
    incomplete = checker.inspect_run(tmp_path, expected_iterations=30)
    assert incomplete["status"] == "NO-GO"
    assert any("expected 29" in blocker for blocker in incomplete["blockers"])
    unknown = checker.inspect_run(tmp_path)
    assert unknown["status"] == "NO-GO"


def test_checker_cannot_use_one_loss_or_failed_completion_as_success(tmp_path):
    make_checkpoint(tmp_path / "nerfstudio_models")
    make_events(tmp_path, missing="ray_drop_loss")
    (tmp_path / "launch.json").write_text(json.dumps({"expected_iterations": 20}))
    (tmp_path / "completion.json").write_text(json.dumps({"returncode": 2, "status": "failed"}))
    result = checker.inspect_run(tmp_path)
    assert result["status"] == "NO-GO"
    assert any("ray_drop_loss" in blocker for blocker in result["blockers"])
    assert "Trainer did not exit successfully" in result["blockers"]


def test_checker_refuses_mixing_launches(tmp_path):
    pytest.importorskip("tensorboard")
    for name in ("first", "second"):
        run = tmp_path / name
        run.mkdir()
        (run / "launch.json").write_text("{}")
    report = checker.inspect_run(tmp_path, expected_iterations=20)
    assert report["status"] == "NO-GO"
    assert "Multiple launches" in report["blockers"][0]
