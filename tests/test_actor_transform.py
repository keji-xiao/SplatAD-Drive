import math

import pytest
import torch

from splatad_drive.actor_controller import ActorController
from splatad_drive.geometry import quaternion_to_matrix
from splatad_drive.scene import ActorTrajectory, GaussianScene, SceneState


def make_scene():
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 10.
    return GaussianScene([[1., 0., 0.], [2., 0., 0.]], [[1., 0., 0., 0.]]*2,
                         [[.1, .2, .3]]*2, [.8, .8], [[1., 0., 0.]]*2, [-1, 7],
                         trajectories={7: ActorTrajectory([0., 2.], poses)})


def test_actor_trajectory_only_moves_local_gaussians():
    scene = make_scene()
    means = scene.world_gaussians(SceneState(1.))["means"]
    assert torch.allclose(means, torch.tensor([[1., 0., 0.], [7., 0., 0.]]))
    assert scene.world_gaussians(SceneState(-1.))["means"][1, 0] == 2.
    assert scene.world_gaussians(SceneState(3.))["means"][1, 0] == 12.


def test_actor_rotation_updates_means_and_covariance_orientation():
    scene = make_scene()
    pose = torch.eye(4)
    pose[:3, :3] = quaternion_to_matrix(torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)]))
    result = scene.world_gaussians(SceneState(1., actor_poses={7: pose}))
    assert torch.allclose(result["means"][1], torch.tensor([0., 2., 0.]), atol=1e-6)
    assert torch.allclose(quaternion_to_matrix(result["quats"][1]), pose[:3, :3], atol=1e-6)


def test_controller_edit_visibility_and_reset_preserve_original_scene():
    scene = make_scene()
    controller = ActorController(scene)
    edited = torch.eye(4)
    edited[1, 3] = 3.
    controller.set_actor_pose(7, 1., edited)
    result = scene.world_gaussians(controller.state(1.))
    assert torch.allclose(result["means"][1], torch.tensor([2., 3., 0.]))
    controller.set_actor_visible(7, False)
    assert scene.world_gaussians(controller.state(1.))["actor_ids"].tolist() == [-1]
    controller.reset()
    assert scene.world_gaussians(controller.state(1.))["means"][1, 0] == 7.
    with pytest.raises(KeyError):
        controller.set_actor_visible(-1, False)


def test_actor_gradient_survives_interpolation_and_transform():
    scene = make_scene()
    scene.world_gaussians(SceneState(1.))["means"].sum().backward()
    assert torch.allclose(scene.means.grad, torch.ones_like(scene.means))


def test_absolute_unix_timestamps_preserve_subsecond_motion():
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 2.
    trajectory = ActorTrajectory([1700000000., 1700000000.2], poses)
    assert torch.allclose(trajectory.at(1700000000.1)[0, 3], torch.tensor(1.), atol=2e-6)
