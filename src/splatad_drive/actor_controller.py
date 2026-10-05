"""Non-destructive dynamic actor edits shared by camera and LiDAR."""
from __future__ import annotations

import math
import torch

from .geometry import as_tensor, validate_transform
from .scene import ActorTrajectory, GaussianScene, SceneState


class ActorController:
    def __init__(self, scene: GaussianScene):
        self.scene = scene
        self._poses: dict[int, dict[float, torch.Tensor]] = {}
        self._visibility: dict[int, bool] = {}

    def _validate_actor(self, actor_id: int):
        if actor_id < 0 or not bool(torch.any(self.scene.actor_ids == actor_id)):
            raise KeyError(f"unknown dynamic actor {actor_id}")

    def set_actor_pose(self, actor_id: int, timestamp: float, T_world_actor):
        self._validate_actor(actor_id)
        if not math.isfinite(timestamp):
            raise ValueError("actor edit timestamp must be finite")
        pose = as_tensor(T_world_actor, like=self.scene.means)
        validate_transform(pose, "T_world_actor")
        # First edit starts from the existing timeline, preserving unedited keyframes.
        if actor_id not in self._poses:
            original = self.scene.trajectories.get(actor_id)
            self._poses[actor_id] = ({} if original is None else
                                     {float(t): p.detach().clone() for t, p in
                                      zip(original.timestamps, original.poses)})
        self._poses[actor_id][float(timestamp)] = pose.clone()

    def set_actor_visible(self, actor_id: int, visible: bool):
        self._validate_actor(actor_id)
        self._visibility[actor_id] = bool(visible)

    def reset(self):
        self._poses.clear()
        self._visibility.clear()

    def state(self, timestamp: float) -> SceneState:
        trajectories = {}
        for actor_id, keyframes in self._poses.items():
            times = sorted(keyframes)
            poses = torch.stack([keyframes[t].to(self.scene.means) for t in times])
            trajectories[actor_id] = ActorTrajectory(torch.tensor(times, device=poses.device, dtype=torch.float64), poses)
        return SceneState(timestamp, actor_visibility=dict(self._visibility), actor_trajectories=trajectories)
