"""Single entry point for the reference renderer or an official checkpoint."""
from pathlib import Path


class SplatADDrive:
    def __init__(self, backend):
        self.backend = backend

    @classmethod
    def from_scene(cls, scene):
        from .reference import ReferenceBackend
        return cls(ReferenceBackend(scene))

    @classmethod
    def from_checkpoint(cls, config: str | Path):
        from .backends.neurad import NeuradBackend
        return cls(NeuradBackend.from_config(Path(config)))

    def render_camera(self, request, scene=None):
        return self.backend.render_camera(request, scene) if scene is not None else self.backend.render_camera(request)

    def render_lidar(self, request, scene=None):
        return self.backend.render_lidar(request, scene) if scene is not None else self.backend.render_lidar(request)

    def set_actor_pose(self, actor_id, timestamp, T_world_actor):
        self.backend.set_actor_pose(actor_id, timestamp, T_world_actor)

    def set_actor_visible(self, actor_id, visible):
        self.backend.set_actor_visible(actor_id, visible)

    def remove_actor(self, actor_id):
        self.set_actor_visible(actor_id, False)
