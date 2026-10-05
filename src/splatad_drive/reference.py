"""Executable PyTorch validation backend; not a reproduction of SplatAD.

The official learned appearance CNN, intensity/ray-drop MLP, CUDA tiling,
densification, full motion models and calibrated sensor response are absent.
"""
from __future__ import annotations

from .actor_controller import ActorController
from .camera_renderer import CameraRequest, render_camera
from .lidar_renderer import LiDARRequest, render_lidar
from .scene import GaussianScene, SceneState


class ReferenceBackend:
    name = "torch-reference"

    def __init__(self, scene: GaussianScene):
        self.scene = scene
        self.controller = ActorController(scene)

    def _state(self, timestamp: float, scene: SceneState | None) -> SceneState:
        controlled = self.controller.state(timestamp)
        if scene is None:
            return controlled
        return SceneState(scene.timestamp, actor_poses=dict(scene.actor_poses),
                          actor_visibility={**controlled.actor_visibility, **scene.actor_visibility},
                          actor_trajectories={**controlled.actor_trajectories, **scene.actor_trajectories})

    def render_camera(self, request: CameraRequest, scene: SceneState | None = None):
        return render_camera(self.scene, request, self._state(request.timestamp, scene))

    def render_lidar(self, request: LiDARRequest, scene: SceneState | None = None):
        return render_lidar(self.scene, request, self._state(request.timestamp, scene))

    def set_actor_pose(self, actor_id: int, timestamp: float, T_world_actor):
        self.controller.set_actor_pose(actor_id, timestamp, T_world_actor)

    def set_actor_visible(self, actor_id: int, visible: bool):
        self.controller.set_actor_visible(actor_id, visible)
