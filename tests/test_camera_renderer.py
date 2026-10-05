from dataclasses import replace
import math

import torch

from splatad_drive.camera_renderer import CameraRequest, render_camera
from splatad_drive.reference import ReferenceBackend
from splatad_drive.scene import ActorTrajectory, GaussianScene, SceneState


def request(**kwargs):
    result = CameraRequest("front", 0., torch.eye(4),
                           torch.tensor([[10., 0., 2.5], [0., 10., 2.5], [0., 0., 1.]]), 5, 5)
    return replace(result, **kwargs)


def scene(means, colors, opacities, scales=None, actor_ids=None, trajectories=None):
    size = len(means)
    return GaussianScene(means, [[1., 0., 0., 0.]]*size,
                         scales or [[.3, .3, .3]]*size, opacities, colors, actor_ids,
                         trajectories=trajectories)


def test_exact_front_to_back_compositing_and_expected_depth():
    model = scene([[0., 0., 2.], [0., 0., 4.]], [[1., 0., 0.], [0., 0., 1.]], [.5, .8])
    result = render_camera(model, request())
    assert torch.allclose(result["alpha"][2, 2], torch.tensor(.9), atol=1e-6)
    assert torch.allclose(result["rgb"][2, 2], torch.tensor([.5, 0., .4]), atol=2e-6)
    assert torch.allclose(result["depth"][2, 2], torch.tensor((.5*2+.4*4)/.9), atol=1e-6)


def test_anisotropic_splat_and_gradient():
    model = scene([[0., 0., 3.]], [[.8, .2, .1]], [.7], scales=[[.8, .08, .1]])
    result = render_camera(model, request())
    assert result["alpha"][2, 0] > result["alpha"][0, 2]
    loss = result["rgb"].sum()+result["depth"].sum()
    loss.backward()
    for parameter in (model.means, model.quats, model.log_scales, model.opacity_logits, model.color_logits):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.log_scales.grad.abs().sum() > 0
    assert model.means.grad.abs().sum() > 0


def test_actor_removal_shared_backend_state_and_explicit_override():
    model = scene([[0., 0., 3.]], [[1., 0., 0.]], [.8], actor_ids=[3])
    backend = ReferenceBackend(model)
    backend.set_actor_visible(3, False)
    result = backend.render_camera(request(background=(.1, .2, .3)))
    assert result["alpha"].sum() == 0
    assert torch.allclose(result["rgb"][0, 0], torch.tensor([.1, .2, .3]))
    assert backend.render_camera(request(), SceneState(0., actor_visibility={3: True}))["alpha"].sum() > 0


def test_rolling_shutter_motion_changes_render_and_zero_duration_is_stable():
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[0, 0, 3], poses[1, 0, 3] = -1., 1.
    model = scene([[0., 0., 3.]], [[1., 0., 0.]], [.8], actor_ids=[3],
                  trajectories={3: ActorTrajectory([-1., 1.], poses)})
    global_result = render_camera(model, request())
    rolling_result = render_camera(model, request(rolling_shutter_duration=1.))
    assert not torch.allclose(global_result["alpha"], rolling_result["alpha"])
    assert torch.allclose(global_result["alpha"], render_camera(model, request(linear_velocity=torch.ones(3)))["alpha"])


def test_near_far_and_behind_camera_points_are_excluded():
    model = scene([[0., 0., -1.], [0., 0., .01], [0., 0., 201.]], [[1., 0., 0.]]*3, [.8]*3)
    result = render_camera(model, request(far=200.))
    assert result["alpha"].sum() == 0
    assert result["depth"].sum() == 0


def test_default_camera_keeps_far_sky_gaussians_but_explicit_clip_is_respected():
    model = scene([[0., 0., 5000.]], [[.2, .5, .9]], [.8], scales=[[300., 300., 300.]])
    rendered = render_camera(model, request())
    assert rendered["alpha"][2, 2] > .79
    assert torch.isclose(rendered["depth"][2, 2], torch.tensor(5000.))
    assert render_camera(model, request(far=200.))["alpha"].sum() == 0
