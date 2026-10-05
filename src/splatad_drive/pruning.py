"""Reviewable actor-aware Gaussian pruning selection, not a training strategy.

No tensors, optimizers, MCMC state, or upstream models are modified here. Applying
a mask to a trained model requires backend-specific parameter/optimizer updates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class PruningConfig:
    camera_weight: float = 1.0
    lidar_weight: float = 1.0
    opacity_weight: float = 1.0
    score_threshold: float = 0.5
    opacity_threshold: float = 0.1
    strong_lidar_hit_threshold: int = 2
    min_actor_gaussians: int = 32
    static_id: int = -1

    def validate(self) -> None:
        for name in ("camera_weight", "lidar_weight", "opacity_weight", "score_threshold", "opacity_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.opacity_threshold > 1:
            raise ValueError("opacity_threshold must lie in [0, 1]")
        for name in ("strong_lidar_hit_threshold", "min_actor_gaussians"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.static_id, bool) or not isinstance(self.static_id, int):
            raise ValueError("static_id must be an integer")


@dataclass
class PruningResult:
    keep_mask: torch.Tensor
    prune_mask: torch.Tensor
    scores: torch.Tensor
    report: dict[str, Any]


def select_pruning_mask(
    opacities: Any,
    actor_ids: Any,
    camera_visibility_count: Any,
    lidar_visibility_count: Any,
    lidar_hit_count: Any,
    *,
    config: PruningConfig | None = None,
) -> PruningResult:
    """Score raw counts, protect LiDAR-supported points, enforce actor minima.

    Candidates satisfy score < threshold AND opacity < threshold. A count
    strictly above strong_lidar_hit_threshold always protects a Gaussian.
    Every dynamic actor retains at least min(original count, minimum); static
    Gaussians have no per-actor minimum. Restoration ranks highest scores first,
    breaking ties by original index, deterministically. Visibility counts must
    be nonnegative integers, with hits <= LiDAR visibility count.
    """
    config = config or PruningConfig()
    config.validate()
    opacity = torch.as_tensor(opacities).detach().to(dtype=torch.float64)
    if opacity.ndim != 1 or not torch.isfinite(opacity).all() or torch.any((opacity < 0) | (opacity > 1)):
        raise ValueError("opacities must be a finite one-dimensional array in [0, 1]")
    ids = torch.as_tensor(actor_ids, device=opacity.device)
    if ids.shape != opacity.shape or ids.dtype == torch.bool or not torch.isfinite(ids).all() or torch.any(ids != ids.round()):
        raise ValueError("actor_ids must be one-dimensional integer IDs matching opacities")
    ids = ids.long()

    def counts(value: Any, name: str) -> torch.Tensor:
        array = torch.as_tensor(value, device=opacity.device).detach().to(dtype=opacity.dtype)
        if array.shape != opacity.shape or not torch.isfinite(array).all() or torch.any(array < 0) or torch.any(array != array.round()):
            raise ValueError(f"{name} must contain matching nonnegative integer counts")
        return array

    camera_count = counts(camera_visibility_count, "camera_visibility_count")
    lidar_count = counts(lidar_visibility_count, "lidar_visibility_count")
    hit_count = counts(lidar_hit_count, "lidar_hit_count")
    if torch.any(hit_count > lidar_count):
        raise ValueError("lidar_hit_count cannot exceed lidar_visibility_count")
    scores = config.camera_weight * camera_count + config.lidar_weight * lidar_count + config.opacity_weight * opacity
    candidates = (scores < config.score_threshold) & (opacity < config.opacity_threshold)
    lidar_protected = hit_count > config.strong_lidar_hit_threshold
    keep = ~candidates | lidar_protected
    restored = torch.zeros_like(keep)
    per_actor: dict[str, dict[str, int]] = {}
    for actor_id in torch.unique(ids, sorted=True).tolist():
        if actor_id == config.static_id:
            continue
        actor_indices = torch.nonzero(ids == actor_id, as_tuple=False).flatten()
        before = actor_indices.numel()
        minimum = min(before, config.min_actor_gaussians)
        required = minimum - int(keep[actor_indices].sum())
        if required > 0:
            candidate_indices = actor_indices[~keep[actor_indices]]
            ranking = torch.argsort(scores[candidate_indices], descending=True, stable=True)
            restore_indices = candidate_indices[ranking[:required]]
            keep[restore_indices] = True
            restored[restore_indices] = True
        per_actor[str(actor_id)] = {"before": before, "after": int(keep[actor_indices].sum()), "minimum": minimum}
    total = opacity.numel()
    report = {
        "mode": "selection_only; no model or optimizer mutation",
        "config": asdict(config),
        "gaussians_before": total,
        "gaussians_after": int(keep.sum()),
        "pruned": int((~keep).sum()),
        "candidates": int(candidates.sum()),
        "protected_by_lidar": int((candidates & lidar_protected).sum()),
        "restored_for_actor_minimum": int(restored.sum()),
        "actors": per_actor,
    }
    return PruningResult(keep_mask=keep, prune_mask=~keep, scores=scores, report=report)
