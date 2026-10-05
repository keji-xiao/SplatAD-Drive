"""Explicit camera/LiDAR metrics; unmeasurable quantities are ``None``.

Camera inputs are HWC in [0, data_range]. LiDAR ranges/points are metres.
Range errors use the intersection of GT and predicted returns; coverage is
reported alongside them so dropping difficult rays cannot hide missing returns.
Ray-drop classification uses True/1 = *no return*, not a valid hit.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


def _tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.as_tensor(value, device=device).detach().to(dtype=torch.float64)


def _mask(value: Any, shape: torch.Size, device: torch.device, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if result.shape != shape or not torch.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must be a boolean/binary mask with shape {tuple(shape)}")
    return result.bool()


def _image(value: Any) -> torch.Tensor:
    result = _tensor(value)
    if result.ndim == 2:
        result = result[..., None]
    if result.ndim != 3 or result.shape[-1] not in (1, 3) or min(result.shape[:2]) < 1:
        raise ValueError("images must be nonempty HWC RGB or grayscale arrays")
    return result


def _ssim(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, data_range: float
) -> float | None:
    # Wang et al. Gaussian-window SSIM, valid convolution (no padded pixels).
    size = min(11, prediction.shape[0], prediction.shape[1])
    size -= 1 - size % 2
    coordinates = torch.arange(size, device=prediction.device, dtype=prediction.dtype) - size // 2
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    channels = prediction.shape[-1]
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(channels, 1, size, size)
    prediction = torch.where(valid[..., None], prediction, 0).permute(2, 0, 1)[None]
    target = torch.where(valid[..., None], target, 0).permute(2, 0, 1)[None]

    def convolution(values: torch.Tensor) -> torch.Tensor:
        return F.conv2d(values, kernel, groups=channels)

    mu_prediction, mu_target = convolution(prediction), convolution(target)
    var_prediction = (convolution(prediction.square()) - mu_prediction.square()).clamp_min(0)
    var_target = (convolution(target.square()) - mu_target.square()).clamp_min(0)
    covariance = convolution(prediction * target) - mu_prediction * mu_target
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    similarity = ((2 * mu_prediction * mu_target + c1) * (2 * covariance + c2)) / (
        (mu_prediction.square() + mu_target.square() + c1) * (var_prediction + var_target + c2)
    )
    window_valid = F.conv2d(
        valid[None, None].to(prediction.dtype),
        torch.ones((1, 1, size, size), dtype=prediction.dtype, device=prediction.device),
    ) == size * size
    selected = similarity[window_valid.expand_as(similarity)]
    return float(selected.mean()) if selected.numel() else None


def camera_metrics(
    pred_rgb: Any,
    target_rgb: Any,
    *,
    mask: Any = None,
    data_range: float = 1.0,
    lpips_model: Callable | None = None,
) -> dict[str, Any]:
    """Compute PSNR and Gaussian-window SSIM without silently clipping RGB.

    Optional LPIPS requires an explicitly supplied pretrained callable accepting
    NCHW RGB tensors in [-1, 1]. This function neither downloads weights nor
    substitutes another metric. For masked inputs LPIPS is unavailable.
    SSIM averages only completely valid windows; small images use the largest
    odd window that fits, with sigma 1.5. Infinite PSNR means identical images.
    """
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    prediction, target = _image(pred_rgb), _image(target_rgb)
    target = target.to(prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("pred_rgb and target_rgb must have equal shapes")
    valid = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    if mask is not None:
        valid = _mask(mask, valid.shape, valid.device, "mask")
    count = int(valid.sum())
    result: dict[str, Any] = {
        "valid_pixels": count, "mse": None, "psnr_db": None, "ssim": None,
        "psnr_status": "no_valid_pixels",
        "lpips": None, "lpips_status": "not_requested; supply a pretrained lpips_model",
    }
    size = min(11, min(prediction.shape[:2]))
    result["ssim_window_size"] = size - (1 - size % 2)
    if count == 0:
        result["status"] = "no_valid_pixels"
        result["lpips_status"] = "unavailable: no valid pixels"
        return result
    for name, values in (("pred_rgb", prediction[valid]), ("target_rgb", target[valid])):
        if not torch.isfinite(values).all() or torch.any((values < 0) | (values > data_range)):
            raise ValueError(f"{name} valid pixels must be finite and within [0, data_range]")
    mse = float((prediction[valid] - target[valid]).square().mean())
    result.update(mse=mse, psnr_db=math.inf if mse == 0 else 10 * math.log10(data_range**2 / mse))
    result["psnr_status"] = "perfect_match_infinite" if mse == 0 else "finite"
    result["ssim"] = _ssim(prediction, target, valid, data_range)
    result["ssim_status"] = "ok" if result["ssim"] is not None else "no_fully_valid_windows"
    result["status"] = "ok"
    if lpips_model is not None:
        if not valid.all() or prediction.shape[-1] != 3:
            result["lpips_status"] = "unavailable: LPIPS requires unmasked RGB images"
        else:
            # Caller supplies a model on the same device; no hidden device transfer.
            pred_input = prediction.permute(2, 0, 1)[None].float() * (2 / data_range) - 1
            target_input = target.permute(2, 0, 1)[None].float() * (2 / data_range) - 1
            with torch.inference_mode():
                value = torch.as_tensor(lpips_model(pred_input, target_input))
            if not value.numel() or not torch.isfinite(value).all():
                raise ValueError("lpips_model returned an empty or nonfinite result")
            result["lpips"] = float(value.mean())
            result["lpips_status"] = "computed_with_supplied_model"
    return result


def lidar_metrics(
    pred_range: Any,
    target_range: Any,
    *,
    target_hit: Any = None,
    pred_drop_probability: Any = None,
    pred_intensity: Any = None,
    target_intensity: Any = None,
    evaluation_mask: Any = None,
    drop_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate rays, with explicit missing-return counts and return coverage.

    If no GT hit mask is given, finite positive GT ranges define hits; other
    entries represent drops. An explicit hit mask cannot label invalid range
    values as hits. ``evaluation_mask`` excludes unobserved rays from *all*
    metrics. Undefined precision/recall and empty regression sets return None.
    Intensity units are inherited from inputs (e.g. normalized [0, 1]).
    """
    if not math.isfinite(drop_threshold) or not 0 <= drop_threshold <= 1:
        raise ValueError("drop_threshold must be in [0, 1]")
    prediction = _tensor(pred_range)
    target = _tensor(target_range, device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("pred_range and target_range must have equal shapes")
    evaluated = torch.ones_like(target, dtype=torch.bool)
    if evaluation_mask is not None:
        evaluated = _mask(evaluation_mask, target.shape, target.device, "evaluation_mask")
    target_valid = torch.isfinite(target) & (target > 0)
    gt_hit = target_valid if target_hit is None else _mask(target_hit, target.shape, target.device, "target_hit")
    if torch.any(evaluated & gt_hit & ~target_valid):
        raise ValueError("target_hit labels an invalid/nonpositive target range as a hit")
    prediction_valid = torch.isfinite(prediction) & (prediction > 0)
    if pred_drop_probability is None:
        pred_drop = ~prediction_valid
    else:
        probability = _tensor(pred_drop_probability, device=target.device)
        if probability.shape != target.shape:
            raise ValueError("pred_drop_probability must match range shape")
        active = probability[evaluated]
        if not torch.isfinite(active).all() or torch.any((active < 0) | (active > 1)):
            raise ValueError("evaluated drop probabilities must be finite in [0, 1]")
        pred_drop = probability >= drop_threshold
    returned = evaluated & gt_hit & ~pred_drop & prediction_valid
    gt_drop = ~gt_hit
    true_positive = int((evaluated & pred_drop & gt_drop).sum())
    false_positive = int((evaluated & pred_drop & gt_hit).sum())
    false_negative = int((evaluated & ~pred_drop & gt_drop).sum())
    true_negative = int((evaluated & ~pred_drop & gt_hit).sum())
    num_evaluated, num_gt_hits, num_common = int(evaluated.sum()), int((evaluated & gt_hit).sum()), int(returned.sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    result: dict[str, Any] = {
        "evaluated_rays": num_evaluated,
        "target_returns": num_gt_hits,
        "common_valid_returns": num_common,
        "missing_target_returns": num_gt_hits - num_common,
        "return_coverage": ratio(num_common, num_gt_hits),
        "invalid_predicted_hit_ranges": int((evaluated & ~pred_drop & ~prediction_valid).sum()),
        "range_rmse_m": None,
        "range_mae_m": None,
        "range_median_absolute_error_m": None,
        "range_mean_relative_error": None,
        "intensity_rmse": None,
        "ray_drop_accuracy": ratio(true_positive + true_negative, num_evaluated),
        "ray_drop_precision": ratio(true_positive, true_positive + false_positive),
        "ray_drop_recall": ratio(true_positive, true_positive + false_negative),
        "ray_drop_tp": true_positive, "ray_drop_fp": false_positive,
        "ray_drop_fn": false_negative, "ray_drop_tn": true_negative,
        "ray_drop_positive_class": "drop (no return)",
        "range_metric_policy": "common valid returns; consult return_coverage",
    }
    if num_common:
        error = (prediction[returned] - target[returned]).abs()
        result.update(
            range_rmse_m=float(error.square().mean().sqrt()),
            range_mae_m=float(error.mean()),
            range_median_absolute_error_m=float(torch.quantile(error, 0.5)),
            range_mean_relative_error=float((error / target[returned]).mean()),
        )
    if (pred_intensity is None) != (target_intensity is None):
        raise ValueError("pred_intensity and target_intensity must be supplied together")
    if pred_intensity is not None:
        intensity_prediction = _tensor(pred_intensity, device=target.device)
        intensity_target = _tensor(target_intensity, device=target.device)
        if intensity_prediction.shape != target.shape or intensity_target.shape != target.shape:
            raise ValueError("intensities must match range shape")
        if not torch.isfinite(intensity_prediction[returned]).all() or not torch.isfinite(intensity_target[returned]).all():
            raise ValueError("intensities on common valid returns must be finite")
        if num_common:
            result["intensity_rmse"] = float((intensity_prediction[returned] - intensity_target[returned]).square().mean().sqrt())
    return result


def chamfer_distance(
    pred_points: Any,
    target_points: Any,
    *,
    chunk_size: int = 1024,
    squared: bool = True,
    backend: str = "auto",
) -> float:
    """Sum of both directed mean nearest-neighbour distances (no factor 1/2).

    Inputs are Nx3 metres; output is m² when squared=True, metres otherwise.
    Empty/nonfinite point clouds are errors, never a zero distance. CPU ``auto``
    uses SciPy's exact KD-tree when installed; ``torch`` uses two-dimensional
    chunking with O(chunk_size²) distance memory and O(N*M) work. Explicit
    ``scipy`` requires CPU tensors. This is an evaluation metric, not a loss.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if backend not in ("auto", "torch", "scipy"):
        raise ValueError("backend must be auto, torch, or scipy")
    prediction, target = _tensor(pred_points), _tensor(target_points)
    target = target.to(prediction.device)
    for points in (prediction, target):
        if points.ndim != 2 or points.shape[1] != 3 or not points.shape[0] or not torch.isfinite(points).all():
            raise ValueError("point clouds must be nonempty finite Nx3 tensors")
    if backend == "scipy" and prediction.device.type != "cpu":
        raise ValueError("scipy backend requires CPU point clouds")
    if backend != "torch" and prediction.device.type == "cpu":
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            if backend == "scipy":
                raise ImportError("scipy backend requested but scipy is not installed") from None
        else:
            pred_array, target_array = prediction.numpy(), target.numpy()
            forward = cKDTree(target_array).query(pred_array, k=1)[0]
            backward = cKDTree(pred_array).query(target_array, k=1)[0]
            if squared:
                forward, backward = np.square(forward), np.square(backward)
            return float(forward.mean() + backward.mean())

    def directed(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        total = first.new_zeros(())
        for source in first.split(chunk_size):
            nearest = source.new_full((source.shape[0],), float("inf"))
            for destination in second.split(chunk_size):
                distances = torch.cdist(source, destination, compute_mode="donot_use_mm_for_euclid_dist")
                nearest = torch.minimum(nearest, distances.min(dim=1).values)
            total += (nearest.square() if squared else nearest).sum()
        return total / first.shape[0]

    return float(directed(prediction, target) + directed(target, prediction))
