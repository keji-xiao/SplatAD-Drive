"""Adapter for the pinned, official NeuRAD SplatAD checkpoint format.

Imports of Nerfstudio and CUDA extensions are deliberately deferred until load.
Public poses use the parser's centered world, camera CV coordinates, seconds,
and metres. This adapter is for inference, not continued training.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import threading
from types import FunctionType, MethodType

import torch

from ..geometry import (as_tensor, invert_transform, matrix_to_quaternion,
                        quaternion_multiply, validate_transform)
from ..lidar_renderer import expand_time_offsets


class UpstreamUnavailableError(RuntimeError):
    """The optional official backend has not been installed successfully."""


def cv_to_nerfstudio(T_world_camera):
    """CV right/down/forward -> Nerfstudio right/up/back, same world."""
    result = T_world_camera.clone()
    result[..., :3, 1:3] *= -1
    return result


def _with_clip_planes(bound_method, sensor, rasterizer_name, near, far):
    """Bind upstream code to private globals, leaving its module untouched.

    Upstream hardcodes clip distances. A fresh function object uses the exact
    upstream implementation with a private rasterizer closure overriding only
    those arguments; this avoids process-global monkey patches.
    """
    function = bound_method.__func__
    original = function.__globals__.get(rasterizer_name)
    if original is None:
        raise RuntimeError(f"Incompatible upstream: {rasterizer_name} not found in renderer globals")

    def clipped(*args, **kwargs):
        kwargs.update(near_plane=float(near), far_plane=float(far))
        return original(*args, **kwargs)

    private_globals = dict(function.__globals__, **{rasterizer_name: clipped})
    private_function = FunctionType(function.__code__, private_globals, function.__name__,
                                    function.__defaults__, function.__closure__)
    private_function.__kwdefaults__ = function.__kwdefaults__
    return MethodType(private_function, bound_method.__self__)(sensor)


@contextmanager
def _temporary_attributes(target, **attributes):
    """Restore instance attributes, including inherited methods, on any exit."""
    absent = object()
    saved = {key: target.__dict__.get(key, absent) for key in attributes}
    try:
        for key, value in attributes.items():
            setattr(target, key, value)
        yield
    finally:
        for key, value in saved.items():
            if value is absent:
                delattr(target, key)
            else:
                setattr(target, key, value)


def lidar_raster_grid(azimuth, elevation, scan_duration=0.0, time_offsets=None):
    """Build upstream [azimuth_deg,elevation_deg,range,time,intensity] tiles.

    Full, uniform azimuth coverage starts at -pi; elevations are increasing.
    The last tile is padded to 32 columns, then removed from returned images.
    Explicit [A] or [E,A] times are relative to the request timestamp and may
    have any capture order. Otherwise a synthetic scan increases with azimuth.
    """
    azimuth, elevation = torch.as_tensor(azimuth), torch.as_tensor(elevation)
    if azimuth.ndim != 1 or elevation.ndim != 1 or len(azimuth) < 2 or len(elevation) < 8:
        raise ValueError("Official LiDAR requires 1D azimuth and elevation with >=2 rays and >=8 beams")
    if len(elevation) % 8:
        raise ValueError("Official LiDAR requires elevation count divisible by the 8-beam tile height")
    if not torch.isfinite(azimuth).all() or not torch.isfinite(elevation).all():
        raise ValueError("LiDAR angles must be finite")
    step = azimuth[1] - azimuth[0]
    if (step <= 0 or not torch.allclose(azimuth.diff(), step.expand(len(azimuth)-1), atol=1e-5)
            or abs(float(azimuth[0]) + math.pi) > 1e-5
            or abs(float(step) * len(azimuth) - 2*math.pi) > 1e-4):
        raise ValueError("Official LiDAR requires uniform full 360-degree azimuth [-pi, pi), without duplicated endpoint")
    if not torch.all(elevation.diff() > 0) or torch.any(elevation.abs() >= math.pi/2):
        raise ValueError("Elevations must increase strictly within (-pi/2, pi/2)")
    if not math.isfinite(float(scan_duration)) or scan_duration < 0:
        raise ValueError("scan_duration must be finite and nonnegative")
    width = math.ceil(len(azimuth)/32)*32
    padded = azimuth[0] + torch.arange(width, device=azimuth.device, dtype=azimuth.dtype)*step
    elev_grid, azim_grid = torch.meshgrid(elevation, padded, indexing="ij")
    # Clamping padded columns keeps them inside the last tile's valid angles.
    azim_grid = azim_grid.clamp_max(azimuth[-1])
    if time_offsets is None:
        offsets = (azim_grid / (2*math.pi))*float(scan_duration)
        # Center finite sampled rays (not continuous sweep endpoints) at t.
        offsets -= (offsets[:, :len(azimuth)].max() + offsets[:, :len(azimuth)].min())/2
    else:
        offsets = expand_time_offsets(time_offsets, len(elevation), len(azimuth), like=azimuth)
        offsets = torch.cat((offsets, offsets[:, -1:].expand(-1, width-len(azimuth))), dim=1)
        # Preserve the supplied origin: upstream shifts pose/time together
        # before centering offsets internally. Centering only times here would
        # change each ray's physical capture time.
    raster_pts = torch.stack((torch.rad2deg(azim_grid), torch.rad2deg(elev_grid),
                              torch.ones_like(azim_grid), offsets, torch.zeros_like(azim_grid)), -1)[None]
    degrees = torch.rad2deg(elevation)
    boundaries = torch.cat((degrees[:1]-1, (degrees[8::8]+degrees[7:-1:8])/2, degrees[-1:]+1))
    return raster_pts, boundaries, float(torch.rad2deg(step))


class NeuradBackend:
    """Inference adapter; one instance owns its pipeline and serializes renders.

    Actor poses are anchored trajectory edits: set_actor_pose(k,t,T) applies
    T @ inverse(original_pose(k,t)) to the entire existing trajectory. This
    preserves motion and makes both sensors agree across rolling shutter.
    """

    def __init__(self, pipeline, *, config=None, checkpoint_path=None, step=None):
        self.pipeline, self.model = pipeline, pipeline.model
        self.config, self.checkpoint_path, self.step = config, checkpoint_path, step
        required = ("get_camera_outputs", "get_lidar_outputs", "gauss_params", "dynamic_actors")
        if any(not hasattr(self.model, name) for name in required):
            raise TypeError("Checkpoint must contain the official SplatADModel, not NeuRAD or generic Splatfacto")
        self.model.eval()
        self._lock = threading.RLock()
        self._actor_poses, self._visibility = {}, {}
        metadata = getattr(self.model, "kwargs", {}).get("metadata", {})
        self.sensor_idx_to_name = {int(k): v for k, v in metadata.get("sensor_idx_to_name", {}).items()}
        self._lidar_sensor_ids = sorted(k for k, name in self.sensor_idx_to_name.items()
                                       if any(x in name.lower() for x in ("pandar", "lidar", "velodyne")))

    @classmethod
    def from_config(cls, config_path, *, test_mode="inference", data=None, load_dir=None):
        """Load a trusted upstream config.yml and its trained checkpoint.

        Upstream config YAML contains Python objects and checkpoint loading
        uses pickle; only use files from a trusted experiment. The official
        datamanager can still require the original dataset in inference mode.
        """
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        try:
            from nerfstudio.utils.eval_utils import eval_setup
            from gsplat.rendering import rasterization, lidar_rasterization  # noqa: F401
        except (ImportError, OSError) as error:
            raise UpstreamUnavailableError("Install the pinned NeuRAD and custom SplatAD CUDA backend; see docs/UPSTREAM.md") from error
        if not torch.cuda.is_available():
            raise UpstreamUnavailableError("The official SplatAD backend requires a working CUDA GPU")

        def update(upstream_config):
            if data is not None:
                upstream_config.pipeline.datamanager.dataparser.data = Path(data).resolve()
            if load_dir is not None:
                upstream_config.load_dir = Path(load_dir).resolve()
            elif (config_path.parent / "nerfstudio_models").is_dir():
                upstream_config.load_dir = config_path.parent / "nerfstudio_models"
            return upstream_config

        from ..checkpoint import numpy_scalar_checkpoint_context
        with numpy_scalar_checkpoint_context():
            config, pipeline, checkpoint, step = eval_setup(config_path, test_mode=test_mode,
                                                            update_config_callback=update, strict_load=True)
        return cls(pipeline, config=config, checkpoint_path=checkpoint, step=step)

    @property
    def device(self):
        return self.model.gauss_params["means"].device

    def _actor_id(self, actor_id):
        if int(actor_id) != actor_id or not 0 <= int(actor_id) < self.model.dynamic_actors.n_actors:
            raise ValueError(f"Unknown dynamic actor id {actor_id}; static scene is not an editable actor")
        return int(actor_id)

    def set_actor_pose(self, actor_id, timestamp, T_world_actor):
        with self._lock:
            actor_id = self._actor_id(actor_id)
            transform = as_tensor(T_world_actor, like=self.model.gauss_params["means"])
            validate_transform(transform, "T_world_actor")
            if not math.isfinite(float(timestamp)):
                raise ValueError("timestamp must be finite")
            self._actor_poses[actor_id] = (float(timestamp), transform.detach().clone())

    def set_actor_visible(self, actor_id, visible):
        with self._lock:
            self._visibility[self._actor_id(actor_id)] = bool(visible)

    def clear_edits(self):
        with self._lock:
            self._actor_poses.clear()
            self._visibility.clear()

    @contextmanager
    def _edit_context(self, scene=None):
        with self._lock, torch.no_grad():
            actors = self.model.dynamic_actors
            edits, visibility = dict(self._actor_poses), dict(self._visibility)
            if scene is not None:
                edits.update({self._actor_id(k): (float(scene.timestamp), as_tensor(v, like=self.model.means))
                              for k, v in scene.actor_poses.items()})
                visibility.update({self._actor_id(k): bool(v) for k, v in scene.actor_visibility.items()})
            if not edits and all(visibility.values()):
                # Ordinary inference needs no masks, parameter copies or method
                # replacement. Avoid scanning every Gaussian for an empty edit.
                yield
                return
            get_boxes, get_velocities = actors.get_boxes2world, actors.get_velocities
            deltas = {}
            for actor_id, (timestamp, target) in edits.items():
                validate_transform(target, "T_world_actor")
                boxes, _ = get_boxes(torch.tensor([timestamp], device=self.device), flatten=False)
                deltas[actor_id] = target @ invert_transform(boxes.reshape(-1, actors.n_actors, 4, 4)[0, actor_id])

            def edited_boxes(query_times, flatten=True):
                if flatten:
                    raise RuntimeError("SplatAD-Drive edits require official unflattened actor rendering")
                boxes, *extra = get_boxes(query_times, flatten=False)
                boxes = boxes.clone()
                for actor_id, delta in deltas.items():
                    boxes[..., actor_id, :, :] = delta @ boxes[..., actor_id, :, :]
                return (boxes, *extra)

            def edited_velocities(query_times):
                velocities = get_velocities(query_times).clone()
                for actor_id, delta in deltas.items():
                    velocities[..., actor_id, :3] = velocities[..., actor_id, :3] @ delta[:3, :3].T
                return velocities

            params = self.model.gauss_params
            ids = params["id"].reshape(-1).long()
            touched = torch.zeros_like(ids, dtype=torch.bool)
            for actor_id in set(deltas) | {k for k, v in visibility.items() if not v}:
                touched |= ids == actor_id
            saved_opacities, saved_quats = params["opacities"][touched].clone(), params["quats"][touched].clone()
            try:
                for actor_id, visible in visibility.items():
                    if not visible:
                        params["opacities"][ids == actor_id] = -torch.inf
                for actor_id, delta in deltas.items():
                    mask = ids == actor_id
                    # Upstream supplies quats directly in world axes. Apply only
                    # the edit delta, preserving the checkpoint's unedited output.
                    params["quats"][mask] = quaternion_multiply(matrix_to_quaternion(delta[:3, :3]), params["quats"][mask])
                with _temporary_attributes(actors, get_boxes2world=edited_boxes, get_velocities=edited_velocities):
                    yield
            finally:
                params["opacities"][touched] = saved_opacities
                params["quats"][touched] = saved_quats

    def _metadata(self, sensor_idx, velocity, angular_velocity=None):
        if sensor_idx < 0 or sensor_idx >= self.model.appearance_embedding.num_embeddings:
            raise ValueError(f"Sensor embedding {sensor_idx} is absent from checkpoint")
        velocity = torch.zeros(3, device=self.device) if velocity is None else torch.as_tensor(velocity, device=self.device, dtype=torch.float32)
        if velocity.shape != (3,) or not torch.isfinite(velocity).all():
            raise ValueError("linear_velocity must be a finite sensor-local three-vector")
        angular = torch.zeros(3, device=self.device) if angular_velocity is None else torch.as_tensor(angular_velocity, device=self.device, dtype=torch.float32)
        if angular.shape != (3,) or not torch.isfinite(angular).all():
            raise ValueError("angular_velocity must be a finite sensor-local three-vector in radians per second")
        return {"sensor_idxs": torch.tensor([[sensor_idx]], device=self.device, dtype=torch.long),
                "linear_velocities_local": velocity[None],
                "angular_velocities_local": angular[None]}

    def _sensor_index(self, identifier, *, lidar=False):
        if isinstance(identifier, str) and not identifier.isdigit():
            matches = [index for index, name in self.sensor_idx_to_name.items()
                       if name.lower().removesuffix("_camera") == identifier.lower().removesuffix("_camera")]
            if len(matches) != 1:
                raise ValueError(f"Unknown sensor {identifier!r}; available: {self.sensor_idx_to_name}")
            sensor_idx = matches[0]
        elif lidar:
            if not 0 <= int(identifier) < len(self._lidar_sensor_ids):
                raise ValueError(f"Unknown LiDAR ordinal {identifier}; checkpoint LiDAR sensors: {self._lidar_sensor_ids}")
            sensor_idx = self._lidar_sensor_ids[int(identifier)]
        else:
            sensor_idx = int(identifier)
        if (sensor_idx in self._lidar_sensor_ids) != lidar:
            raise ValueError(f"Sensor {identifier!r} is not a {'LiDAR' if lidar else 'camera'}")
        return sensor_idx

    @contextmanager
    def _request_config(self, **values):
        # Inference uses caller poses/velocities, including explicit zero RS.
        velocity_config = self.model.camera_velocity_optimizer.config
        with _temporary_attributes(self.model.config, use_camopt_in_eval=False, **values), \
             _temporary_attributes(velocity_config, enabled=False, zero_initial_velocities=False):
            yield

    def render_camera(self, request, scene=None):
        from nerfstudio.cameras.cameras import Cameras, CameraType
        T_world_camera = torch.as_tensor(request.T_world_camera, dtype=torch.float32, device=self.device)
        validate_transform(T_world_camera, "T_world_camera")
        K = torch.as_tensor(request.K, dtype=torch.float32, device=self.device)
        if (K.shape != (3, 3) or not torch.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not torch.allclose(K[2], K.new_tensor([0, 0, 1]))
                or abs(float(K[0, 1])) > 1e-7 or abs(float(K[1, 0])) > 1e-7):
            raise ValueError("Official camera requires finite zero-skew pinhole K with positive focal lengths")
        if abs(float(getattr(request, "pixel_variance", .3))-.3) > 1e-8:
            raise ValueError("Official camera uses fixed pixel_variance=.3")
        metadata = self._metadata(self._sensor_index(request.camera_id), request.linear_velocity, request.angular_velocity)
        metadata["linear_velocities_local"] = metadata["linear_velocities_local"] * K.new_tensor([1, -1, -1])
        # Official Cameras metadata is GL-local; the model flips it back to CV
        # for its projection kernel. The proper 180-degree rotation applies to
        # angular vectors as well as linear vectors (no world-frame rotation).
        metadata["angular_velocities_local"] = metadata["angular_velocities_local"] * K.new_tensor([1, -1, -1])
        metadata.update(rolling_shutter_time=K.new_tensor([[request.rolling_shutter_duration]]),
                        time_to_center_pixel=K.new_zeros(1, 1))
        camera = Cameras(camera_to_worlds=cv_to_nerfstudio(T_world_camera)[None, :3],
                         fx=K[0, 0][None], fy=K[1, 1][None], cx=K[0, 2][None], cy=K[1, 2][None],
                         width=request.width, height=request.height, camera_type=CameraType.PERSPECTIVE,
                         times=K.new_tensor([request.timestamp]), metadata=metadata)
        with self._edit_context(scene), self._request_config(compensate_rs_camera=True, background_color="random"), \
             _temporary_attributes(self.model, background_color=K.new_tensor(request.background)):
            result = _with_clip_planes(self.model.get_camera_outputs, camera, "rasterization", request.near, request.far)
        alpha = result["accumulation"].reshape(request.height, request.width)
        depth = result["depth"].reshape(request.height, request.width)
        return {"rgb": result["rgb"], "depth": torch.where(alpha > 1e-8, depth, torch.zeros_like(depth)), "alpha": alpha}

    def render_lidar(self, request, scene=None):
        from nerfstudio.cameras.lidars import Lidars, LidarType
        if abs(request.beam_divergence - .003) > 1e-8:
            raise ValueError("Official trained SplatAD uses fixed .003-radian beam divergence; arbitrary divergence is unsupported")
        T_world_lidar = torch.as_tensor(request.T_world_lidar, dtype=torch.float32, device=self.device)
        validate_transform(T_world_lidar, "T_world_lidar")
        azimuth = torch.as_tensor(request.azimuth, device=self.device, dtype=torch.float32)
        elevation = torch.as_tensor(request.elevation, device=self.device, dtype=torch.float32)
        raster_pts, boundaries, resolution = lidar_raster_grid(
            azimuth, elevation, request.scan_duration, request.time_offsets)
        sensor_idx = self._sensor_index(request.lidar_id, lidar=True)
        metadata = self._metadata(sensor_idx, request.linear_velocity, request.angular_velocity)
        lidar = Lidars(lidar_to_worlds=T_world_lidar[None, :3], lidar_type=LidarType.PANDAR64,
                       times=T_world_lidar.new_tensor([request.timestamp]), metadata=metadata)
        # Add large raster tensors after TensorDataclass's sensor broadcasting.
        lidar.metadata.update(raster_pts=raster_pts, elevation_boundaries=boundaries, azimuth_resolution=resolution)
        with self._edit_context(scene), self._request_config(compensate_rs_lidar=True):
            result = _with_clip_planes(self.model.get_lidar_outputs, lidar, "lidar_rasterization", request.near, request.far)
        height, width = len(elevation), len(azimuth)
        plane = lambda name: result[name].reshape(height, -1)[:, :width]
        alpha, depth_sum, ray_drop = plane("accumulation"), plane("depth"), plane("ray_drop_prob")
        ranges = torch.where(alpha > 1e-8, depth_sum/alpha.clamp_min(1e-8), torch.zeros_like(depth_sum))
        elev_grid, azim_grid = torch.meshgrid(elevation, azimuth, indexing="ij")
        directions = torch.stack((elev_grid.cos()*azim_grid.cos(), elev_grid.cos()*azim_grid.sin(), elev_grid.sin()), -1)
        return {"range": ranges, "intensity": plane("intensity"), "hit_probability": alpha*(1-ray_drop),
                "ray_drop": ray_drop, "alpha": alpha, "points": directions*ranges[..., None],
                "depth_sum": depth_sum, "median_range": plane("median_depth")}
