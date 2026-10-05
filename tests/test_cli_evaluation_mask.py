"""The NPZ command must distinguish observed drops from unobserved padding."""
import json

import numpy as np
import pytest
import torch

from splatad_drive.cli import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("NPZ evaluation attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def arrays(padded_x=500.0):
    pred = {"range": np.array([[11., 0., padded_x, 19.]]),
            "ray_drop": np.array([[0., 1., 0., 0.]]),
            "points": np.array([[[11., 0., 0.], [0., 0., 0.], [padded_x, 0., 0.], [19., 0., 0.]]])}
    gt = {"range": np.array([[10., 0., 0., 20.]]),
          "hit": np.array([[True, False, False, True]]),
          "points": np.array([[[10., 0., 0.], [0., 0., 0.], [0., 0., 0.], [20., 0., 0.]]]),
          "evaluation_mask": np.array([[True, True, False, True]])}
    return pred, gt


def run_evaluation(tmp_path, pred, gt):
    np.savez(tmp_path / "prediction.npz", **pred)
    np.savez(tmp_path / "target.npz", **gt)
    output = tmp_path / "result.json"
    status = main(["evaluate", "--prediction", str(tmp_path / "prediction.npz"),
                   "--target", str(tmp_path / "target.npz"), "--output", str(output)])
    return status, json.loads(output.read_text()) if output.exists() else None


@pytest.mark.parametrize("padded_x", [500., 1e9, float("nan")])
def test_unobserved_padding_does_not_change_accuracy_or_chamfer(tmp_path, padded_x):
    pred, gt = arrays(padded_x)
    # Only the target defines observation support; prediction cannot hide errors.
    pred["evaluation_mask"] = np.zeros_like(gt["evaluation_mask"])
    status, result = run_evaluation(tmp_path, pred, gt)
    assert status == 0
    assert result["lidar"]["evaluated_rays"] == 3
    assert result["lidar"]["target_returns"] == 2
    assert result["lidar"]["ray_drop_accuracy"] == 1
    assert result["lidar"]["ray_drop_tp"] == 1  # An observed no-return remains evaluated.
    assert result["lidar"]["range_rmse_m"] == 1
    assert result["chamfer_bidirectional_sum_m2"] == 2


def test_without_mask_existing_all_rays_policy_is_preserved(tmp_path):
    pred, gt = arrays()
    del gt["evaluation_mask"]
    status, result = run_evaluation(tmp_path, pred, gt)
    assert status == 0
    assert result["lidar"]["evaluated_rays"] == 4
    assert result["lidar"]["ray_drop_accuracy"] == .75
    assert result["chamfer_bidirectional_sum_m2"] > 1000


@pytest.mark.parametrize("mask", [np.array([True, True, False, True]),
                                  np.array([[1., 1., 2., 1.]]),
                                  np.array([[1., 1., np.nan, 1.]]),
                                  np.array([["1", "1", "0", "1"]])])
def test_rejects_wrong_mask_shape_or_values(tmp_path, mask, capsys):
    pred, gt = arrays()
    gt["evaluation_mask"] = mask
    status, result = run_evaluation(tmp_path, pred, gt)
    assert status == 2 and result is None
    assert "GT evaluation_mask" in capsys.readouterr().err


def test_masked_chamfer_requires_corresponding_point_count(tmp_path, capsys):
    pred, gt = arrays()
    pred["points"] = pred["points"][:, :3]
    status, result = run_evaluation(tmp_path, pred, gt)
    assert status == 2 and result is None
    assert "one corresponding predicted and target point per ray" in capsys.readouterr().err


def test_points_only_mask_and_finite_binary_values(tmp_path):
    pred, gt = arrays()
    del pred["range"], gt["range"]
    gt["evaluation_mask"] = gt["evaluation_mask"].astype(np.int8)
    status, result = run_evaluation(tmp_path, pred, gt)
    assert status == 0 and result == {"chamfer_bidirectional_sum_m2": 2}


def test_range_grid_mask_supports_row_major_flattened_points(tmp_path):
    pred, gt = arrays()
    pred["points"] = pred["points"].reshape(-1, 3)
    gt["points"] = gt["points"].reshape(-1, 3)
    status, result = run_evaluation(tmp_path, pred, gt)
    assert status == 0 and result["chamfer_bidirectional_sum_m2"] == 2
