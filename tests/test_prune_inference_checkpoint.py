"""CPU artifact tests; do not import an official model or initialize CUDA."""
from collections import OrderedDict
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from splatad_drive.trainer_checkpoint import ResumableTrainerMixin, restricted_checkpoint_load


spec = importlib.util.spec_from_file_location(
    "prune_inference_checkpoint", Path(__file__).parents[1] / "scripts/prune_inference_checkpoint.py")
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inference export attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def source_state():
    pipeline = OrderedDict()
    widths = {"means": 3, "scales": 3, "quats": 4, "features_dc": 3,
              "features_rest": 5, "opacities": 1, "id": 1}
    for name, width in widths.items():
        pipeline["_model.gauss_params." + name] = torch.arange(7 * width, dtype=torch.float64).reshape(7, width)
    pipeline["_model.gauss_params.id"] = torch.tensor([3, 0, 3, 1, 3, 2, 3], dtype=torch.float64).reshape(-1, 1)
    pipeline["_model.gauss_params.opacities"] = torch.tensor([-8, -20, -7, -20, -6, -20, 0], dtype=torch.float64).reshape(-1, 1)
    pipeline["_model.dynamic_actors.actor_sizes"] = torch.ones(3, 3)
    pipeline["_model.dynamic_actors.actor_positions"] = torch.arange(18).reshape(2, 3, 3)
    pipeline["_model.rgb_decoder.weights"] = torch.tensor([42.0])
    pipeline._metadata = OrderedDict({"_model": {"version": 1}})
    return {"step": 3, "pipeline": pipeline, "optimizers": {"means": {"state": torch.tensor([77.])}},
            "schedulers": {"lr": 0.123}, "scalers": {"scale": 1024}, "splatad_drive": {"rng": "source"}}


def test_same_mask_original_order_all_actors_and_non_gaussian_state_preserved():
    source = source_state()
    before = deepcopy(source)
    result, report = prune.make_inference_state(source)
    keep = torch.tensor([False, True, False, True, True, True, True])
    assert result["inference_only"] is True
    assert result["step"] == source["step"]
    assert all(result[name] == {} for name in ("optimizers", "schedulers", "scalers"))
    assert "splatad_drive" not in result
    for key, tensor in source["pipeline"].items():
        torch.testing.assert_close(tensor, before["pipeline"][key], rtol=0, atol=0)
        expected = tensor[keep] if ".gauss_params." in key else tensor
        torch.testing.assert_close(result["pipeline"][key], expected, rtol=0, atol=0)
    assert result["pipeline"]._metadata == source["pipeline"]._metadata
    assert source["optimizers"]["means"]["state"].item() == 77
    assert source["splatad_drive"] == {"rng": "source"}
    assert report["before"] == 7 and report["after"] == 5 and report["removed"] == 2
    assert report["static_id"] == 3
    assert [group["removed"] for group in report["groups"]] == [0, 0, 0, 2]
    assert result["pipeline"]["_model.gauss_params.id"].ravel().tolist() == [0, 1, 3, 2, 3]


def test_strict_threshold_keeps_exact_boundary():
    source = source_state()
    boundary = torch.sigmoid(source["pipeline"]["_model.gauss_params.opacities"])[2].item()
    result, report = prune.make_inference_state(source, threshold=boundary)
    assert 0 < boundary < 0.001
    assert report["removed"] == 1
    torch.testing.assert_close(result["pipeline"]["_model.gauss_params.means"],
                               source["pipeline"]["_model.gauss_params.means"][1:], rtol=0, atol=0)


@pytest.mark.parametrize("threshold", [0.00101, 1.0, -0.1, float("nan"), float("inf"), True])
def test_threshold_cannot_exceed_authorized_limit(threshold):
    with pytest.raises(ValueError, match="threshold"):
        prune.make_inference_state(source_state(), threshold)


def test_zero_threshold_retains_every_row_and_extra_gaussian_parameter_uses_same_mask():
    source = source_state()
    source["pipeline"]["_model.gauss_params.extra_feature"] = torch.arange(14).reshape(7, 2)
    result, report = prune.make_inference_state(source, threshold=0)
    assert report["removed"] == 0
    for key, value in source["pipeline"].items():
        torch.testing.assert_close(value, result["pipeline"][key], rtol=0, atol=0)
    pruned, _ = prune.make_inference_state(source)
    torch.testing.assert_close(pruned["pipeline"]["_model.gauss_params.extra_feature"],
                               source["pipeline"]["_model.gauss_params.extra_feature"][[1, 3, 4, 5, 6]], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["id", "shape", "nan"])
def test_rejects_inconsistent_source_without_mutation(invalid):
    source = source_state()
    if invalid == "id":
        source["pipeline"]["_model.gauss_params.id"][0, 0] = -1
    elif invalid == "shape":
        source["pipeline"]["_model.gauss_params.scales"] = torch.zeros(6, 3)
    else:
        source["pipeline"]["_model.gauss_params.means"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        prune.make_inference_state(source)


def test_real_serialization_roundtrip_provenance_and_source_files_unchanged(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    config = tmp_path / "config.yml"
    torch.save(source_state(), checkpoint)
    config.write_bytes(b"!!python/object:TrainerConfig\r\nexperiment_name: baseline_30000\r\nmax_num_iterations: 30000\r\n")
    source_bytes, config_bytes = checkpoint.read_bytes(), config.read_bytes()
    output = tmp_path / "pruned"
    manifest = prune.prune_checkpoint(checkpoint, config, output)
    restored = restricted_checkpoint_load(output / "nerfstudio_models/step-000000003.ckpt")
    assert restored["inference_only"] is True and len(restored["pipeline"]["_model.gauss_params.means"]) == 5
    assert restored["pipeline"]._metadata == source_state()["pipeline"]._metadata
    assert checkpoint.read_bytes() == source_bytes and config.read_bytes() == config_bytes
    assert (output / "config.yml").read_bytes() == config_bytes.replace(b"baseline_30000", b"pruned_baseline_alpha001")
    assert restored["splatad_drive_inference"]["source"]["checkpoint_sha256"] == prune.sha256_file(checkpoint)
    assert all(manifest["checks"].values())
    assert "NOT solely" in manifest["storage_caveat"]
    with pytest.raises(FileExistsError):
        prune.prune_checkpoint(checkpoint, config, output)
    with pytest.raises(ValueError, match="inference-only derivative"):
        prune.make_inference_state(restored)


@pytest.mark.parametrize("load_via_directory", [False, True])
def test_resume_rejects_inference_artifact_before_model_or_scaler_mutation(tmp_path, load_via_directory):
    state, _ = prune.make_inference_state(source_state())
    checkpoint = tmp_path / "step-000000003.ckpt"
    torch.save(state, checkpoint)
    calls = []
    subject = SimpleNamespace(
        config=SimpleNamespace(load_checkpoint=None if load_via_directory else checkpoint,
                               load_dir=tmp_path if load_via_directory else None,
                               max_num_iterations=4, load_optimizer=True, load_scheduler=True),
        pipeline=SimpleNamespace(load_pipeline=lambda *args: calls.append("model")),
        grad_scaler=SimpleNamespace(load_state_dict=lambda *args: calls.append("scaler")),
    )
    with pytest.raises(ValueError, match="inference-only checkpoint cannot be resumed"):
        ResumableTrainerMixin._load_checkpoint(subject)
    assert calls == [] and not hasattr(subject, "_start_step")


@pytest.mark.parametrize("marker", [None, False])
def test_original_training_state_remains_resumable(tmp_path, marker):
    state = source_state()
    if marker is not None:
        state["inference_only"] = marker
    checkpoint = tmp_path / "step-000000003.ckpt"
    torch.save(state, checkpoint)
    calls = []
    subject = SimpleNamespace(
        config=SimpleNamespace(load_checkpoint=checkpoint, load_dir=None, max_num_iterations=5),
        pipeline=SimpleNamespace(load_pipeline=lambda *args: calls.append("model")),
        grad_scaler=SimpleNamespace(load_state_dict=lambda *args: calls.append("scaler")),
    )
    ResumableTrainerMixin._load_checkpoint(subject)
    assert calls == ["model", "scaler"] and subject._start_step == 4
    assert subject._pending_checkpoint["schedulers"] == state["schedulers"]
