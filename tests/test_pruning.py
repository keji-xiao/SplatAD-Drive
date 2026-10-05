import pytest
import torch

from splatad_drive.pruning import PruningConfig, select_pruning_mask


def test_actor_minimum_deterministic_ties_and_static_pruning():
    result = select_pruning_mask(
        [0.01] * 7, [-1, -1, 0, 0, 0, 1, 1], [0] * 7, [0] * 7, [0] * 7,
        config=PruningConfig(min_actor_gaussians=2),
    )
    assert result.keep_mask.tolist() == [False, False, True, True, False, True, True]
    assert result.report["gaussians_before"] == 7
    assert result.report["gaussians_after"] == 4
    assert result.report["pruned"] == 3
    assert result.report["restored_for_actor_minimum"] == 4
    assert result.report["actors"]["0"]["after"] == 2


def test_lidar_protection_independent_of_score_and_strict_threshold():
    result = select_pruning_mask(
        [0.01] * 3, [-1] * 3, [0] * 3, [3, 2, 0], [3, 2, 0],
        config=PruningConfig(lidar_weight=0, strong_lidar_hit_threshold=2),
    )
    assert result.keep_mask.tolist() == [True, False, False]
    assert result.report["protected_by_lidar"] == 1


def test_restore_highest_score_and_opacity_gate():
    result = select_pruning_mask(
        [0.01, 0.03, 0.02, 0.5], [0, 0, 0, -1], [0] * 4, [0] * 4, [0] * 4,
        config=PruningConfig(min_actor_gaussians=1),
    )
    assert result.keep_mask.tolist() == [False, True, False, True]


def test_no_input_mutation_and_empty_scene():
    opacity = torch.tensor([0.01], requires_grad=True)
    original = opacity.clone()
    result = select_pruning_mask(opacity, [0], [0], [0], [0])
    assert torch.equal(opacity, original)
    assert not result.scores.requires_grad
    assert result.keep_mask.tolist() == [True]
    empty = select_pruning_mask([], [], [], [], [])
    assert empty.report["gaussians_after"] == 0


@pytest.mark.parametrize(
    "camera,lidar,hits", [([-1], [0], [0]), ([0.5], [0], [0]), ([0], [1], [2]), ([float("nan")], [0], [0])]
)
def test_invalid_counts_rejected(camera, lidar, hits):
    with pytest.raises(ValueError):
        select_pruning_mask([0.01], [0], camera, lidar, hits)


@pytest.mark.parametrize("config", [PruningConfig(min_actor_gaussians=-1), PruningConfig(opacity_threshold=2), PruningConfig(camera_weight=float("nan"))])
def test_invalid_configuration_rejected(config):
    with pytest.raises(ValueError):
        select_pruning_mask([], [], [], [], [], config=config)


def test_invalid_opacity_id_and_shape_rejected():
    with pytest.raises(ValueError, match="opacities"):
        select_pruning_mask([2], [0], [0], [0], [0])
    with pytest.raises(ValueError, match="actor_ids"):
        select_pruning_mask([0.01], [0.1], [0], [0], [0])
    with pytest.raises(ValueError, match="actor_ids"):
        select_pruning_mask([0.01], [0, 1], [0], [0], [0])
