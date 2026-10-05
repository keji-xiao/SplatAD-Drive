import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from splatad_drive.evaluator import camera_metrics, chamfer_distance, lidar_metrics
from splatad_drive.io import save_json, save_ply, save_png, to_jsonable


def test_camera_identical_and_known_psnr():
    target = torch.full((13, 15, 3), 0.5)
    identical = camera_metrics(target, target)
    assert math.isinf(identical["psnr_db"])
    assert identical["ssim"] == pytest.approx(1)
    assert identical["lpips"] is None
    prediction = target + 0.25
    measured = camera_metrics(prediction, target)
    assert measured["mse"] == pytest.approx(0.0625)
    assert measured["psnr_db"] == pytest.approx(12.041199826559248)
    assert measured["ssim"] == pytest.approx((2 * 0.75 * 0.5 + 0.01**2) / (0.75**2 + 0.5**2 + 0.01**2))


def test_camera_mask_excludes_invalid_pixels_and_no_valid_windows():
    prediction = torch.zeros(5, 5, 3)
    target = prediction.clone()
    prediction[0, 0] = float("nan")
    mask = torch.ones(5, 5, dtype=torch.bool)
    mask[0, 0] = False
    result = camera_metrics(prediction, target, mask=mask)
    assert math.isinf(result["psnr_db"])
    assert result["ssim"] is None
    assert result["ssim_status"] == "no_fully_valid_windows"
    assert camera_metrics(prediction, target, mask=torch.zeros_like(mask))["status"] == "no_valid_pixels"
    with pytest.raises(ValueError, match="finite"):
        camera_metrics(prediction, target)


def test_camera_tiny_image_and_real_lpips_hook():
    observed = []

    def supplied_model(prediction, target):
        observed.append((prediction.shape, float(prediction.min()), float(target.max())))
        return torch.tensor(0.125)

    result = camera_metrics(torch.zeros(2, 2, 3), torch.ones(2, 2, 3), lpips_model=supplied_model)
    assert result["lpips"] == 0.125
    assert result["ssim_window_size"] == 1
    assert observed == [(torch.Size((1, 3, 2, 2)), -1, 1)]


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf")])
def test_camera_invalid_data_range(invalid):
    with pytest.raises(ValueError):
        camera_metrics(torch.zeros(2, 2, 3), torch.zeros(2, 2, 3), data_range=invalid)


def test_lidar_definition_and_missing_returns():
    # GT drops: [False, False, True, True], predicted drops [False, True, False, True].
    result = lidar_metrics(
        [12, 21, 3, 0], [10, 20, 0, 0], pred_drop_probability=[0.1, 0.9, 0.1, 0.9],
        pred_intensity=[0.6, 0.8, float("nan"), float("nan")], target_intensity=[0.4, 0.7, float("nan"), float("nan")],
    )
    assert result["target_returns"] == 2
    assert result["common_valid_returns"] == 1
    assert result["return_coverage"] == 0.5
    assert result["missing_target_returns"] == 1
    assert result["range_rmse_m"] == 2
    assert result["range_mean_relative_error"] == 0.2
    assert result["intensity_rmse"] == pytest.approx(0.2)
    assert result["ray_drop_accuracy"] == 0.5
    assert result["ray_drop_precision"] == 0.5
    assert result["ray_drop_recall"] == 0.5
    assert all(result[f"ray_drop_{key}"] == 1 for key in ("tp", "fp", "tn", "fn"))


def test_lidar_even_median_and_empty_returns():
    result = lidar_metrics([11, 24], [10, 20])
    assert result["range_median_absolute_error_m"] == 2.5
    assert result["range_rmse_m"] == pytest.approx(math.sqrt(8.5))
    assert result["ray_drop_precision"] is None
    assert result["ray_drop_recall"] is None
    dropped = lidar_metrics([0, float("nan")], [10, 20])
    assert dropped["range_rmse_m"] is None
    assert dropped["return_coverage"] == 0
    empty = lidar_metrics([], [])
    assert empty["ray_drop_accuracy"] is None
    assert empty["range_rmse_m"] is None


def test_lidar_invalid_prediction_cannot_hide_as_success():
    result = lidar_metrics([float("nan")], [10], pred_drop_probability=[0.1])
    assert result["invalid_predicted_hit_ranges"] == 1
    assert result["missing_target_returns"] == 1
    assert result["return_coverage"] == 0
    assert result["range_rmse_m"] is None


def test_lidar_evaluation_mask_and_validation():
    result = lidar_metrics([11, float("nan")], [10, float("nan")], evaluation_mask=[1, 0])
    assert result["evaluated_rays"] == 1
    assert result["ray_drop_accuracy"] == 1
    with pytest.raises(ValueError, match="invalid/nonpositive"):
        lidar_metrics([1], [0], target_hit=[1])
    with pytest.raises(ValueError, match="probabilities"):
        lidar_metrics([1], [1], pred_drop_probability=[1.1])
    with pytest.raises(ValueError, match="together"):
        lidar_metrics([1], [1], pred_intensity=[0.5])
    with pytest.raises(ValueError, match="mask"):
        lidar_metrics([1], [1], evaluation_mask=[2])


@pytest.mark.parametrize("backend", ["torch", "auto"])
def test_chamfer_units_and_sum_convention(backend):
    # Directed distances: [0, 2] and [0]. Sum of means is 1 m / 2 m².
    prediction = [[0, 0, 0], [2, 0, 0]]
    target = [[0, 0, 0]]
    assert chamfer_distance(prediction, target, chunk_size=1, backend=backend) == 2
    assert chamfer_distance(prediction, target, chunk_size=1, squared=False, backend=backend) == 1


def test_chamfer_chunk_equivalence_and_invalid():
    rng = np.random.default_rng(123)
    first, second = rng.normal(size=(9, 3)), rng.normal(size=(11, 3))
    all_distances = np.linalg.norm(first[:, None] - second[None], axis=-1)
    expected = np.square(all_distances.min(axis=1)).mean() + np.square(all_distances.min(axis=0)).mean()
    assert chamfer_distance(first, second, chunk_size=3, backend="torch") == pytest.approx(expected)
    with pytest.raises(ValueError, match="nonempty"):
        chamfer_distance(np.empty((0, 3)), second)
    with pytest.raises(ValueError, match="positive integer"):
        chamfer_distance(first, second, chunk_size=0)


def test_artifact_exports_roundtrip(tmp_path):
    path = save_json(tmp_path / "metadata.json", {"metric": torch.tensor([1, float("inf")]), "label": "测试"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"metric": [1, None], "label": "测试"}
    rgb = np.array([[[1.0, 0.0, 0.5]]], dtype=np.float32)
    png = save_png(tmp_path / "rgb.png", rgb)
    assert np.array(Image.open(png)).tolist() == [[[255, 0, 128]]]
    ply = save_ply(tmp_path / "cloud.ply", [[1, 2, 3]], rgb.reshape(-1, 3))
    assert "element vertex 1" in ply.read_text()
    assert ply.read_text().strip().endswith("1 2 3 255 0 128")
    assert "element vertex 0" in save_ply(tmp_path / "empty.ply", np.empty((0, 3))).read_text()
    with pytest.raises(ValueError):
        save_png(tmp_path / "invalid.png", rgb * 2)
    with pytest.raises(ValueError):
        save_ply(tmp_path / "invalid.ply", [[float("nan"), 0, 0]])
    with pytest.raises(TypeError):
        to_jsonable(object())
