"""A deterministic synthetic road scene and a real differentiable fit.

This validates the public API. It is deliberately labelled reference/synthetic
and must never be reported as a PandaSet or official SplatAD benchmark.
"""
from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import torch

from .scene import GaussianScene, ActorTrajectory, SceneState
from .camera_renderer import CameraRequest
from .lidar_renderer import LiDARRequest, expand_time_offsets
from .drive import SplatADDrive
from .geometry import invert_transform, transform_points
from .io import save_json, save_png, save_ply


def make_demo(device="cpu", width=128, height=80):
    # World is FLU; camera's columns are right=-Y, down=-Z, forward=+X.
    means, scales, colors, actor_ids = [], [], [], []
    def add(position, scale, color, actor=-1):
        means.append(position); scales.append(scale); colors.append(color); actor_ids.append(actor)
    for x in np.linspace(2, 22, 18):
        for y in np.linspace(-5, 5, 13):
            color = [.45,.46,.47] if abs(y) > .3 else [.9,.85,.55]
            add([x,y,-1.4],[.55,.5,.06],color)
    for x in np.linspace(5,22,12):
        for z in np.linspace(-.7,3,6):
            add([x,5,z],[.8,.12,.5],[.29,.45,.6])
            add([x,-5,z],[.8,.12,.5],[.5,.39,.28])
    for x in np.linspace(-1.4,1.4,7):
        for y in np.linspace(-.65,.65,4):
            for z in (-.4,.2,.55):
                add([x,y,z],[.34,.25,.22],[.82,.15,.055],0)
    kwargs = {"device": device, "dtype": torch.float32}
    first = torch.eye(4,**kwargs); first[:3,3] = torch.tensor([8.,0.,-.5],**kwargs)
    last = first.clone(); last[1,3] = 1.
    scene = GaussianScene(torch.tensor(means,**kwargs), torch.tensor([[1,0,0,0]]*len(means),**kwargs),
                          torch.tensor(scales,**kwargs),torch.full((len(means),),.8,**kwargs),
                          torch.tensor(colors,**kwargs),torch.tensor(actor_ids,device=device),
                          {0:ActorTrajectory(torch.tensor([0.,1.],**kwargs),torch.stack([first,last]))})
    T_world_camera = torch.eye(4,**kwargs)
    T_world_camera[:3,:3] = torch.tensor([[0,0,1],[-1,0,0],[0,-1,0]],**kwargs)
    camera = CameraRequest(0,.5,T_world_camera,
                           torch.tensor([[width*.8,0,width/2],[0,width*.8,height/2],[0,0,1]],**kwargs),
                           width,height,background=(.12,.18,.28))
    # Deliberately nonlinear beam spacing; not a claim of a real sensor calibration.
    beams = torch.linspace(-1,1,16,**kwargs).sign()*torch.linspace(-1,1,16,**kwargs).abs().pow(1.3)*.27
    lidar = LiDARRequest(0,.5,torch.eye(4,**kwargs),
                         torch.arange(128,**kwargs)*(2*math.pi/128)-math.pi, beams)
    return scene, camera, lidar


def depth_colors(depth, valid=None, maximum=25.):
    depth = torch.as_tensor(depth).detach().cpu()
    fraction = (depth/maximum).clamp(0,1)
    result = torch.stack((1-fraction, 1-(fraction-.5).abs()*2, fraction),-1)
    valid = depth > 0 if valid is None else torch.as_tensor(valid).cpu()
    return result * valid[...,None]


def lidar_points_to_world(points_lidar, lidar, *, compensate_translation=False, point_time_offsets=None):
    """Map instantaneous ray-local points to world, optionally with linear sensor motion.

    Returns flattened [N,3]. Already deskewed raw sweeps must leave compensation off.
    """
    original_points = torch.as_tensor(points_lidar)
    if original_points.ndim < 2 or original_points.shape[-1] != 3:
        raise ValueError("points_lidar must have shape [N,3] or [E,A,3]")
    if not original_points.is_floating_point():
        original_points = original_points.float()
    points = original_points.reshape(-1, 3)
    lidar_pose = torch.as_tensor(lidar.T_world_lidar, dtype=points.dtype, device=points.device)
    points_world = transform_points(lidar_pose, points)
    if point_time_offsets is not None and not compensate_translation:
        raise ValueError("point_time_offsets requires compensate_translation=True")
    if compensate_translation:
        if point_time_offsets is not None:
            offsets = torch.as_tensor(point_time_offsets, dtype=points.dtype, device=points.device)
            if offsets.shape not in (original_points.shape[:-1], (len(points),)):
                raise ValueError("point_time_offsets must match the input point grid or flattened point count")
        else:
            elevation_count, azimuth_count = len(lidar.elevation), len(lidar.azimuth)
            explicit = getattr(lidar, "time_offsets", None)
            duration = float(lidar.scan_duration)
            if explicit is None and duration == 0:
                offsets = points.new_zeros(len(points))
            else:
                if len(points) != elevation_count*azimuth_count:
                    raise ValueError("A filtered cloud needs point_time_offsets; request timing requires the full [E,A] grid")
                if explicit is not None:
                    offsets = expand_time_offsets(explicit, elevation_count, azimuth_count, like=points)
                else:
                    if not math.isfinite(duration) or duration < 0 or azimuth_count < 1:
                        raise ValueError("scan_duration must be finite nonnegative seconds on a nonempty grid")
                    columns = (torch.arange(azimuth_count, dtype=points.dtype, device=points.device)+.5)/azimuth_count-.5
                    offsets = (columns*duration)[None].expand(elevation_count, -1)
        if not torch.isfinite(offsets).all():
            raise ValueError("point_time_offsets must contain finite seconds relative to lidar.timestamp")
        velocity = points.new_zeros(3) if lidar.linear_velocity is None else torch.as_tensor(
            lidar.linear_velocity, dtype=points.dtype, device=points.device)
        if velocity.shape != (3,) or not torch.isfinite(velocity).all():
            raise ValueError("linear_velocity must be a finite sensor-local [3] vector")
        world_velocity = lidar_pose[:3, :3] @ velocity
        points_world = points_world+offsets.reshape(-1, 1)*world_velocity
    return points_world


def overlay(rgb, points_lidar, camera, lidar, valid=None, *,
            compensate_lidar_translation=False, point_time_offsets=None):
    """Project LiDAR points onto the nominal camera-center exposure pose.

    This is a static-geometry diagnostic. Camera/LiDAR timestamps may differ:
    each request pose is used at its own timestamp. Moving actors are NOT
    propagated between sensor times; camera row timing and sensor angular
    velocity are NOT compensated.

    By default all points use T_world_lidar, preserving the legacy behavior.
    Set compensate_lidar_translation=True only for points expressed in each
    ray's instantaneous sensor-local frame, as returned by our renderers:

        p_world = R_lidar @ p_ray + t_lidar + R_lidar @ v_local * ray_dt

    ray_dt is relative to lidar.timestamp. point_time_offsets optionally
    supplies one offset per input point (including a filtered/unstructured
    cloud). Otherwise the full [E,A] grid uses lidar.time_offsets, or the
    reference schedule ((column+.5)/A-.5)*scan_duration. Explicit offsets are
    never recentered. A missing linear_velocity means zero translation.

    Do NOT enable this for already ego-compensated sweep-reference points,
    such as the official parser's exported NPZ cloud: that would double
    compensate sensor motion. This function does not add camera.timestamp -
    lidar.timestamp to ray_dt; the supplied camera pose already represents
    its own center exposure. No full rolling-shutter validation is implied.
    """
    points_world = lidar_points_to_world(points_lidar, lidar,
        compensate_translation=compensate_lidar_translation, point_time_offsets=point_time_offsets)
    points = torch.as_tensor(points_lidar).reshape(-1, 3).to(points_world)
    camera_pose = torch.as_tensor(camera.T_world_camera, dtype=points.dtype, device=points.device)
    intrinsic = torch.as_tensor(camera.K, dtype=points.dtype, device=points.device)
    points_camera = transform_points(invert_transform(camera_pose), points_world)
    pixels = points_camera @ intrinsic.T
    uv = pixels[:,:2] / pixels[:,2:].clamp_min(1e-8)
    keep = torch.isfinite(points_camera).all(-1) & (points_camera[:,2] > camera.near) & (points_camera[:,2] < camera.far)
    keep &= (uv[:,0] >= 0) & (uv[:,0] < camera.width) & (uv[:,1] >= 0) & (uv[:,1] < camera.height)
    if valid is not None:
        keep &= torch.as_tensor(valid,device=points.device).reshape(-1)
    # Diagnostic drawing happens on CPU to avoid one GPU synchronization per
    # projected point. Preserve the input image's device in the returned value.
    result = rgb.detach().cpu().clone()
    selected = torch.where(keep)[0]
    # Farthest first, so the nearest projected point owns colliding pixels.
    selected = selected[torch.argsort(points_camera[selected,2],descending=True)]
    point_colors = depth_colors(points_camera[:,2]).to(result)
    pixel_indices = uv.long().detach().cpu()
    for index in selected.tolist():
        u,v = pixel_indices[index].tolist()
        result[v,u] = point_colors[index]
    return result.to(rgb.device)


def run_demo(output_dir, device="cpu", fit_steps=20):
    if fit_steps < 0:
        raise ValueError("fit_steps must be nonnegative")
    torch.manual_seed(42)
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    scene,camera,lidar = make_demo(device)
    drive = SplatADDrive.from_scene(scene)
    with torch.no_grad():
        before_rgb = drive.render_camera(camera)
        before_lidar = drive.render_lidar(lidar)
        save_png(output/"camera_before.png",before_rgb["rgb"])
        save_png(output/"lidar_range_before.png",depth_colors(before_lidar["range"],before_lidar["hit_probability"]>.1))
        save_png(output/"rgb_lidar_overlay.png",overlay(before_rgb["rgb"],before_lidar["points"],camera,lidar,before_lidar["hit_probability"]>.1))
        save_ply(output/"lidar_before.ply",before_lidar["points"][before_lidar["hit_probability"]>.1])
        drive.remove_actor(0)
        removed_rgb = drive.render_camera(camera); removed_lidar = drive.render_lidar(lidar)
        save_png(output/"camera_removed.png",removed_rgb["rgb"])
        save_png(output/"lidar_range_removed.png",depth_colors(removed_lidar["range"],removed_lidar["hit_probability"]>.1))
        drive.set_actor_visible(0,True)
        moved = scene.trajectories[0].at(.5).clone(); moved[1,3] += 1.5
        drive.set_actor_pose(0,.5,moved)
        moved_rgb = drive.render_camera(camera); moved_lidar = drive.render_lidar(lidar)
        save_png(output/"camera_moved.png",moved_rgb["rgb"])
        save_png(output/"lidar_range_moved.png",depth_colors(moved_lidar["range"],moved_lidar["hit_probability"]>.1))
        save_ply(output/"lidar_moved.ply",moved_lidar["points"][moved_lidar["hit_probability"]>.1])
        rs_camera = replace(camera,rolling_shutter_duration=.08,linear_velocity=torch.tensor([3.,0,0],device=device))
        save_png(output/"camera_rolling_shutter.png",drive.render_camera(rs_camera)["rgb"])
        rs_lidar = replace(lidar,scan_duration=.1,linear_velocity=torch.tensor([4.,0,0],device=device))
        rs_lidar_result = drive.render_lidar(rs_lidar)
        save_png(output/"lidar_rolling_shutter.png",depth_colors(rs_lidar_result["range"],rs_lidar_result["hit_probability"]>.1))
        # Rotate the edited actor about its own center, not the world origin.
        angle=math.radians(10)
        yaw=moved.new_tensor([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
        rotated=moved.clone(); rotated[:3,:3]=yaw@moved[:3,:3]
        drive.set_actor_pose(0,.5,rotated)
        save_png(output/"camera_rotated.png",drive.render_camera(camera)["rgb"])
        rotated_lidar=drive.render_lidar(lidar)
        save_png(output/"lidar_range_rotated.png",depth_colors(rotated_lidar["range"],rotated_lidar["hit_probability"]>.1))
        from .rig import SensorRig
        shifted_ego=torch.eye(4,device=device); shifted_ego[1,3]=.5
        novel_camera,novel_lidar=SensorRig(camera.T_world_camera,lidar.T_world_lidar).requests(shifted_ego,camera,lidar,timestamp=.5)
        save_png(output/"camera_ego_shift.png",drive.render_camera(novel_camera)["rgb"])
        novel_scan=drive.render_lidar(novel_lidar)
        save_png(output/"lidar_ego_shift.png",depth_colors(novel_scan["range"],novel_scan["hit_probability"]>.1))
    # Real optimization of appearance against jointly rendered synthetic targets.
    # Fixed geometry avoids pretending this small fixture can replace a full model.
    fit_scene,fit_camera,fit_lidar = make_demo(device,width=48,height=32)
    fit_drive = SplatADDrive.from_scene(fit_scene)
    with torch.no_grad():
        target_rgb = fit_drive.render_camera(fit_camera)["rgb"].detach()
        target_lidar = fit_drive.render_lidar(fit_lidar)
        target_intensity = target_lidar["intensity"].detach()
        hit = target_lidar["hit_probability"].detach()>.1
        fit_scene.color_logits.add_(.8)
        fit_scene.intensity_logits.sub_(.6)
    for name, parameter in fit_scene.named_parameters():
        parameter.requires_grad_(name in ("color_logits","intensity_logits"))
    optimizer = torch.optim.Adam([fit_scene.color_logits,fit_scene.intensity_logits],lr=.08)
    history = []
    for step in range(fit_steps+1):
        rendered_rgb = fit_drive.render_camera(fit_camera)["rgb"]
        rendered_intensity = fit_drive.render_lidar(fit_lidar)["intensity"]
        rgb_loss = (rendered_rgb-target_rgb).square().mean()
        intensity_loss = (rendered_intensity[hit]-target_intensity[hit]).square().mean()
        loss = rgb_loss+intensity_loss
        if not torch.isfinite(loss):
            raise RuntimeError("synthetic optimization produced a nonfinite loss")
        history.append({"step":step,"loss":float(loss.detach()),"rgb_mse":float(rgb_loss.detach()),"intensity_mse":float(intensity_loss.detach())})
        if step < fit_steps:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    save_json(output/"reference_scene.json",scene.export())
    save_json(output/"fit_history.json",history)
    report = {"backend":"pytorch_reference", "dataset":"synthetic_road", "seed":42,"device":device,
              "gaussians":len(scene.means),"camera_resolution":[camera.width,camera.height],"lidar_shape":list(before_lidar["range"].shape),
              "fit_parameters":["color_logits","intensity_logits"],"fit_steps":fit_steps,
              "initial_loss":history[0]["loss"],"final_loss":history[-1]["loss"],
              "removal_rgb_mean_change":float((before_rgb["rgb"]-removed_rgb["rgb"]).abs().mean()),
              "removal_range_mean_change":float((before_lidar["range"]-removed_lidar["range"]).abs().mean()),
              "official_baseline":"NOT_RUN", "reference_limitations":"No learned CNN/MLP; ray_drop is 1-alpha; row/column RS with linear sensor translation."}
    save_json(output/"report.json",report)
    return report
