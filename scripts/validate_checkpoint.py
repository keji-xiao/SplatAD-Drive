"""Validate a real official checkpoint through the SplatAD-Drive public facade.

This executes inference and scene edits only. It does not train, substitute a
synthetic scene, or claim paper metrics from smoke-checkpoint images.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs" / "torch_cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "outputs" / "torch_extensions"))

import numpy as np
from PIL import Image
import torch

from splatad_drive import CameraRequest, LiDARRequest, SplatADDrive
from splatad_drive.backends.neurad import NeuradBackend, cv_to_nerfstudio
from splatad_drive.geometry import invert_transform
from splatad_drive.io import save_json, save_ply, save_png, to_numpy
from splatad_drive.demo import overlay, lidar_points_to_world


def _homogeneous(transform):
    result = torch.eye(4, dtype=transform.dtype, device=transform.device)
    result[:3] = transform[:3]
    return result


def _metadata_vector(sensor, name):
    value = sensor.metadata.get(name) if sensor.metadata else None
    return torch.zeros(3, device=sensor.device) if value is None else value.reshape(-1, 3)[0]


def pandaset_native_time_offsets(azimuth, scan_period):
    """Pandar64 phase estimate: native -Y forward at t=0, +X at -T/4.

    Preserve the parser's native calibration. This is a calibrated angular
    phase estimate for a novel raster, not the measured time of each return.
    """
    if not math.isfinite(scan_period) or scan_period <= 0:
        raise ValueError("PandaSet scan period must be finite and positive")
    return torch.atan2(-azimuth.cos(), -azimuth.sin()) * (scan_period/(2*math.pi))


def _pandaset_scan_period(backend):
    from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSet

    dataparser = backend.pipeline.datamanager.dataparser
    if not isinstance(dataparser, PandaSet):
        raise ValueError("--rolling-shutter LiDAR phase is calibrated only for the PandaSet parser")
    # Split sensor times can be every other frame. Use all raw sweep times,
    # retaining float64 precision for the absolute Unix timestamps.
    raw_times = np.asarray(dataparser.sequence.lidar.timestamps, dtype=np.float64)
    differences = np.diff(raw_times)
    if raw_times.ndim != 1 or len(differences) == 0 or not np.isfinite(raw_times).all() or np.any(differences <= 0):
        raise ValueError("PandaSet raw LiDAR timestamps must be finite and strictly increasing")
    return float(differences.mean())


def make_requests(backend, image_dataset, lidar_dataset, camera_index, args):
    """Use the loaded checkpoint's own parser world, indices and calibration."""
    from nerfstudio.cameras.lidars import LidarType, get_lidar_elevation_mapping

    cameras = image_dataset.cameras
    camera = cameras[camera_index:camera_index+1].to(backend.device)
    camera_id = int(camera.metadata["sensor_idxs"].reshape(-1)[0])
    native_camera_time = float(camera.times.reshape(-1)[0])
    center_offset = float(camera.metadata.get("time_to_center_pixel", torch.zeros(1)).reshape(-1)[0])
    velocity_gl = _metadata_vector(camera, "linear_velocities_local")
    angular_gl = _metadata_vector(camera, "angular_velocities_local")
    T_world_camera_gl = _homogeneous(camera.camera_to_worlds[0]).clone()
    # Exactly the official model's translation to the center-pixel exposure.
    T_world_camera_gl[:3, 3] += T_world_camera_gl[:3, :3] @ velocity_gl * center_offset
    T_world_camera_cv = cv_to_nerfstudio(T_world_camera_gl)
    K = camera.get_intrinsics_matrices()[0].clone()
    native_width, native_height = int(camera.width.item()), int(camera.height.item())
    K[0, :] *= args.width/native_width
    K[1, :] *= args.height/native_height
    shutter_duration = float(camera.metadata.get("rolling_shutter_time", torch.zeros(1)).reshape(-1)[0])
    camera_request = CameraRequest(camera_id=camera_id, timestamp=native_camera_time+center_offset,
        T_world_camera=T_world_camera_cv, K=K, width=args.width, height=args.height,
        rolling_shutter_duration=shutter_duration if args.rolling_shutter else 0.,
        linear_velocity=velocity_gl*velocity_gl.new_tensor([1, -1, -1]),
        angular_velocity=angular_gl*angular_gl.new_tensor([1, -1, -1]))

    lidar_times = lidar_dataset.lidars.times.reshape(-1)
    lidar_index = int(torch.argmin(torch.abs(lidar_times-native_camera_time)))
    lidar = lidar_dataset.lidars[lidar_index:lidar_index+1].to(backend.device)
    lidar_sensor_id = int(lidar.metadata["sensor_idxs"].reshape(-1)[0])
    lidar_name = backend.sensor_idx_to_name[lidar_sensor_id]
    lidar_type = LidarType(int(lidar.lidar_type.item()))
    elevations = torch.tensor(sorted(get_lidar_elevation_mapping(lidar_type).values()),
                              dtype=torch.float32, device=backend.device).deg2rad()
    azimuths = torch.arange(args.azimuth_count, device=backend.device)*2*math.pi/args.azimuth_count-math.pi
    original_points = lidar_dataset.point_clouds[lidar_index]
    point_times = original_points[:, 4]
    measured_duration = float(point_times.max()-point_times.min()) if len(point_times) else 0.
    scan_period = _pandaset_scan_period(backend) if args.rolling_shutter else None
    lidar_request = LiDARRequest(lidar_id=lidar_name, timestamp=float(lidar.times.item()),
        T_world_lidar=_homogeneous(lidar.lidar_to_worlds[0]), azimuth=azimuths, elevation=elevations,
        scan_duration=scan_period if args.rolling_shutter else 0.,
        linear_velocity=_metadata_vector(lidar, "linear_velocities_local"),
        angular_velocity=_metadata_vector(lidar, "angular_velocities_local"),
        time_offsets=pandaset_native_time_offsets(azimuths, scan_period) if args.rolling_shutter else None)

    manifest = {"camera_index": camera_index, "lidar_index": lidar_index,
        "camera_sensor_id": camera_id, "lidar_sensor_id": lidar_sensor_id,
        "camera_name": backend.sensor_idx_to_name[camera_id], "lidar_name": lidar_name,
        "source_image": image_dataset.image_filenames[camera_index],
        "camera_parser_timestamp": native_camera_time, "camera_center_offset_s": center_offset,
        "camera_request_timestamp": camera_request.timestamp, "lidar_timestamp": lidar_request.timestamp,
        "sensor_time_difference_s": camera_request.timestamp-lidar_request.timestamp,
        "native_camera_size": [native_width, native_height], "render_camera_size": [args.width, args.height],
        "lidar_grid": [len(elevations), len(azimuths)], "elevation_degrees": elevations.rad2deg(),
        "K": K, "T_world_camera_cv": T_world_camera_cv, "T_world_lidar": lidar_request.T_world_lidar,
        "camera_angular_velocity_local_cv_rad_s": camera_request.angular_velocity,
        "lidar_angular_velocity_local_rad_s": lidar_request.angular_velocity,
        "rolling_shutter_enabled": bool(args.rolling_shutter),
        "scan_schedule": "PandaSet native -Y-forward calibrated phase estimate" if args.rolling_shutter else "instantaneous full sweep",
        "scan_period_s": scan_period,
        "scan_period_source": "mean difference of raw unsplit LiDAR timestamps" if args.rolling_shutter else None,
        "measured_sweep_duration_s": measured_duration,
        "notes": ["Camera/LiDAR are sampled at their respective native times; they are not assumed simultaneous.",
                  "Resized GT is context only; no photometric metrics are reported.",
                  "Public requests carry sensor-local linear and angular velocities for the upstream first-order RS correction.",
                  "Novel LiDAR grid times use the calibrated angular phase; they are not measured per-return times."]}
    return camera_request, lidar_request, manifest


def _validate_tensors(result, expected_shapes):
    checks = {}
    for name, shape in expected_shapes.items():
        if name not in result:
            raise AssertionError(f"Renderer omitted required output {name}")
        tensor = result[name]
        finite = bool(torch.isfinite(tensor).all())
        if tuple(tensor.shape) != shape or not finite:
            raise AssertionError(f"{name} shape={tuple(tensor.shape)}, expected={shape}, finite={finite}")
        checks[name] = {"shape": list(tensor.shape), "finite": finite,
                        "min": float(tensor.min()), "max": float(tensor.max())}
    for name, value in result.items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise AssertionError(f"Additional output {name} contains NaN/Inf")
    return checks


def _delta(first, second):
    return {key: {"mean_absolute_change": float((first[key]-second[key]).abs().mean()),
                  "max_absolute_change": float((first[key]-second[key]).abs().max())}
            for key in first if key in second and torch.is_tensor(first[key])}


def render_pair(drive, camera, lidar, output_dir, hit_threshold):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {output_dir.name}: camera then LiDAR", flush=True)
    with torch.no_grad():
        camera_outputs = drive.render_camera(camera)
        lidar_outputs = drive.render_lidar(lidar)
    camera_checks = _validate_tensors(camera_outputs,
        {"rgb": (camera.height, camera.width, 3), "depth": (camera.height, camera.width), "alpha": (camera.height, camera.width)})
    height, width = len(lidar.elevation), len(lidar.azimuth)
    lidar_checks = _validate_tensors(lidar_outputs, {"range": (height, width), "intensity": (height, width),
        "hit_probability": (height, width), "ray_drop": (height, width), "points": (height, width, 3)})
    save_png(output_dir / "rgb.png", camera_outputs["rgb"])
    np.savez_compressed(output_dir / "camera.npz", **{k: to_numpy(v) for k, v in camera_outputs.items()},
                        K=to_numpy(camera.K), T_world_camera=to_numpy(camera.T_world_camera), timestamp=camera.timestamp)
    np.savez_compressed(output_dir / "lidar.npz", **{k: to_numpy(v) for k, v in lidar_outputs.items()},
        T_world_lidar=to_numpy(lidar.T_world_lidar), timestamp=lidar.timestamp,
        azimuth=to_numpy(lidar.azimuth), elevation=to_numpy(lidar.elevation))
    returned = ((lidar_outputs["hit_probability"] >= hit_threshold) & (lidar_outputs["range"] > lidar.near)
                & (lidar_outputs["range"] < lidar.far))
    points = lidar_outputs["points"][returned]
    intensity_colors = lidar_outputs["intensity"][returned][:, None].expand(-1, 3).clamp(0, 1)
    save_ply(output_dir / "lidar_local.ply", points, intensity_colors)
    points_world = lidar_points_to_world(lidar_outputs["points"], lidar, compensate_translation=True).reshape(height, width, 3)[returned]
    save_ply(output_dir / "lidar_world.ply", points_world, intensity_colors)
    save_png(output_dir / "overlay.png", overlay(camera_outputs["rgb"], lidar_outputs["points"], camera, lidar, returned,
                                                compensate_lidar_translation=True))
    summary = {"camera": camera_checks, "lidar": lidar_checks,
               "point_return_threshold": hit_threshold, "returned_point_count": int(returned.sum()),
               "overlay_scope":"LiDAR sensor translation compensated per ray; camera nominal center pose; no angular or actor cross-time compensation"}
    save_json(output_dir / "numeric_checks.json", summary)
    return camera_outputs, lidar_outputs, summary


def choose_actor(backend, camera, lidar, requested_actor=None):
    actors = backend.model.dynamic_actors
    if actors.n_actors == 0:
        return None, None
    with torch.no_grad():
        boxes, present = actors.get_boxes2world(torch.tensor([lidar.timestamp], device=backend.device), flatten=False)
    boxes, present = boxes.reshape(-1, actors.n_actors, 4, 4)[0], present.reshape(-1, actors.n_actors)[0]
    ids = backend.model.gauss_params["id"].reshape(-1).long()
    camera_centers = boxes[:, :3, 3] @ invert_transform(camera.T_world_camera)[:3, :3].T + invert_transform(camera.T_world_camera)[:3, 3]
    scores = []
    for actor_id in range(actors.n_actors):
        if requested_actor is not None and actor_id != requested_actor:
            continue
        count = int((ids == actor_id).sum())
        if not bool(present[actor_id]) or count == 0:
            continue
        projected = camera.K @ camera_centers[actor_id]
        x, y = (projected[:2]/projected[2].clamp_min(1e-6)).tolist()
        on_screen = bool(camera_centers[actor_id, 2] > camera.near) and 0 <= x < camera.width and 0 <= y < camera.height
        scores.append((on_screen, count, actor_id))
    if not scores:
        return None, None
    actor_id = max(scores)[2]
    return actor_id, boxes[actor_id].detach().clone()


def _parameters_unchanged(backend, snapshots):
    return all(torch.equal(backend.model.gauss_params[key], value) for key, value in snapshots.items())


def validate_sample(drive, image_dataset, lidar_dataset, camera_index, args, sample_dir):
    backend = drive.backend
    camera, lidar, sensor_manifest = make_requests(backend, image_dataset, lidar_dataset, camera_index, args)
    save_json(sample_dir / "sensors.json", sensor_manifest)
    save_json(sample_dir / "requests.json", {"camera": camera, "lidar": lidar})
    gt_image = image_dataset.get_image_float32(camera_index)[..., :3]
    gt = torch.nn.functional.interpolate(gt_image.permute(2, 0, 1)[None], size=(args.height, args.width),
                                        mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    save_png(sample_dir / "gt_rgb_resized.png", gt.clamp(0, 1))
    lidar_index = sensor_manifest["lidar_index"]
    gt_points = lidar_dataset.point_clouds[lidar_index]
    gt_return = gt_points[:, :3].norm(dim=-1) < lidar_dataset.lidars.valid_lidar_distance_threshold
    np.savez_compressed(sample_dir / "gt_lidar.npz", points=to_numpy(gt_points),
        columns=np.array(["x", "y", "z", "intensity", "point_relative_time"]),
        T_world_lidar=to_numpy(lidar.T_world_lidar), timestamp=lidar.timestamp)
    save_ply(sample_dir / "gt_lidar_local.ply", gt_points[gt_return, :3])
    snapshots = {key: backend.model.gauss_params[key].detach().clone() for key in ("opacities", "quats")}
    base_camera, base_lidar, numeric = render_pair(drive, camera, lidar, sample_dir / "baseline", args.hit_threshold)
    if not _parameters_unchanged(backend, snapshots):
        raise AssertionError("Unedited rendering changed Gaussian opacity or covariance parameters")
    record = {"camera_index": camera_index, "baseline": numeric, "actor_edit": {"status": "SKIPPED_BY_OPTION"}}
    contact_names = ["baseline"]
    if not args.skip_edits:
        actor_id, original_actor = choose_actor(backend, camera, lidar, args.actor_id)
        if actor_id is None:
            record["actor_edit"] = {"status": "SKIPPED", "reason": "No present actor with Gaussian points at selected LiDAR time"}
        else:
            try:
                drive.remove_actor(actor_id)
                removed_camera, removed_lidar, removed_checks = render_pair(drive, camera, lidar, sample_dir / "actor_removed", args.hit_threshold)
                if not _parameters_unchanged(backend, snapshots):
                    raise AssertionError("Actor removal leaked Gaussian parameter changes")
                backend.clear_edits()
                target_actor = original_actor.clone()
                world_lateral = camera.T_world_camera[:3, 0]
                target_actor[:3, 3] += world_lateral*args.actor_translation
                drive.set_actor_pose(actor_id, lidar.timestamp, target_actor)
                moved_camera, moved_lidar, moved_checks = render_pair(drive, camera, lidar, sample_dir / "actor_translated", args.hit_threshold)
                if not _parameters_unchanged(backend, snapshots):
                    raise AssertionError("Actor translation leaked Gaussian parameter changes")
                backend.clear_edits()
                angle = math.radians(args.actor_rotation)
                yaw = original_actor.new_tensor([[math.cos(angle), -math.sin(angle), 0.],
                                                [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
                rotated_actor = original_actor.clone()
                rotated_actor[:3, :3] = yaw @ original_actor[:3, :3]
                drive.set_actor_pose(actor_id, lidar.timestamp, rotated_actor)
                rotated_camera, rotated_lidar, rotated_checks = render_pair(drive, camera, lidar, sample_dir / "actor_rotated", args.hit_threshold)
                if not _parameters_unchanged(backend, snapshots):
                    raise AssertionError("Actor rotation leaked Gaussian parameter changes")
            finally:
                backend.clear_edits()
            restored_camera, restored_lidar, restored_checks = render_pair(drive, camera, lidar, sample_dir / "restored", args.hit_threshold)
            restored_deltas = {"camera": _delta(base_camera, restored_camera), "lidar": _delta(base_lidar, restored_lidar)}
            restore_error = max(row["max_absolute_change"] for group in restored_deltas.values() for row in group.values())
            if restore_error > args.restore_tolerance or not _parameters_unchanged(backend, snapshots):
                raise AssertionError(f"Rendering did not restore to baseline: max error {restore_error}")
            record["actor_edit"] = {"status": "NUMERIC_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED", "actor_id": actor_id,
                "original_pose": original_actor, "target_pose": target_actor, "translation_metres": args.actor_translation,
                "removed": removed_checks, "translated": moved_checks, "restored": restored_checks,
                "remove_change": {"camera": _delta(base_camera, removed_camera), "lidar": _delta(base_lidar, removed_lidar)},
                "translate_change": {"camera": _delta(base_camera, moved_camera), "lidar": _delta(base_lidar, moved_lidar)},
                "rotation_degrees": args.actor_rotation, "rotated_pose": rotated_actor,
                "rotated": rotated_checks,
                "rotate_change": {"camera": _delta(base_camera, rotated_camera), "lidar": _delta(base_lidar, rotated_lidar)},
                "restore_change": restored_deltas, "restore_max_error": restore_error,
                "parameter_restoration": True,
                "note": "Zero image change can mean this actor was outside visible/returned support; it is not a visual editing PASS."}
            contact_names.extend(["actor_removed", "actor_translated", "actor_rotated", "restored"])
        T_shifted_world_world = torch.eye(4, device=backend.device)
        T_shifted_world_world[:3, 3] = camera.T_world_camera[:3, 0]*args.ego_translation
        ego_camera = replace(camera, T_world_camera=T_shifted_world_world@camera.T_world_camera)
        ego_lidar = replace(lidar, T_world_lidar=T_shifted_world_world@lidar.T_world_lidar)
        _, _, ego_checks = render_pair(drive, ego_camera, ego_lidar, sample_dir / "ego_shifted", args.hit_threshold)
        before_extrinsic = invert_transform(camera.T_world_camera)@lidar.T_world_lidar
        after_extrinsic = invert_transform(ego_camera.T_world_camera)@ego_lidar.T_world_lidar
        extrinsic_error = float((before_extrinsic-after_extrinsic).abs().max())
        if extrinsic_error > 1e-5:
            raise AssertionError("Joint ego shift changed camera-to-LiDAR relative calibration")
        record["ego_shift"] = {"translation_metres": args.ego_translation, "T_shifted_world_world": T_shifted_world_world,
            "relative_extrinsic_max_error": extrinsic_error, "numeric_checks": ego_checks,
            "status": "NUMERIC_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED"}
        contact_names.append("ego_shifted")
    previews = [Image.open(sample_dir / "gt_rgb_resized.png").convert("RGB")]
    previews.extend(Image.open(sample_dir / name / "rgb.png").convert("RGB") for name in contact_names)
    contact = Image.new("RGB", (args.width*len(previews), args.height))
    for index, preview in enumerate(previews):
        contact.paste(preview, (index*args.width, 0))
        preview.close()
    contact.save(sample_dir / "comparison.png")
    record["comparison_order"] = ["gt_rgb_resized"]+contact_names
    save_json(sample_dir / "validation.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Trusted official config.yml with a saved checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "checkpoint_validation")
    parser.add_argument("--data", type=Path, help="Optional dataset path override")
    parser.add_argument("--load-dir", type=Path, help="Optional checkpoint directory override")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--camera", default="front")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0, help="Offset within selected camera's split")
    parser.add_argument("--sample-stride", type=int, default=1, help="Stride through the selected camera's split")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--azimuth-count", type=int, default=1800)
    parser.add_argument("--hit-threshold", type=float, default=.5)
    parser.add_argument("--actor-translation", type=float, default=1.)
    parser.add_argument("--actor-rotation", type=float, default=15., help="Yaw edit in degrees about parser-world +Z")
    parser.add_argument("--actor-id", type=int, help="Choose one actor instead of automatic selection")
    parser.add_argument("--all-actors", action="store_true", help="Repeat selected frames for every actor")
    parser.add_argument("--ego-translation", type=float, default=.5)
    parser.add_argument("--restore-tolerance", type=float, default=1e-5)
    parser.add_argument("--rolling-shutter", action="store_true", help="Use native camera readout duration and PandaSet -Y-forward calibrated LiDAR phase")
    parser.add_argument("--skip-edits", action="store_true")
    args = parser.parse_args(argv)
    if args.all_actors and args.actor_id is not None:
        parser.error("Choose --all-actors or --actor-id")
    if min(args.width, args.height, args.samples, args.azimuth_count, args.sample_stride) < 1 or args.sample_index < 0 or not 0 <= args.hit_threshold <= 1:
        parser.error("Sizes/sample count must be positive, sample-index nonnegative, and hit-threshold in [0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"config": args.config.resolve(), "kind": "real_official_checkpoint_inference", "split": args.split,
              "status": "RUNNING", "paper_metrics_claimed": False, "samples": []}
    save_json(args.output / "validation.json", report)
    try:
        backend = NeuradBackend.from_config(args.config, test_mode="test", data=args.data, load_dir=args.load_dir)
        drive = SplatADDrive(backend)
        datamanager = backend.pipeline.datamanager
        image_dataset = getattr(datamanager, args.split+"_dataset")
        lidar_dataset = getattr(datamanager, args.split+"_lidar_dataset")
        if not len(image_dataset) or not len(lidar_dataset):
            raise ValueError(f"Checkpoint's {args.split} split contains no camera images or LiDAR sweeps")
        matches = [index for index, name in backend.sensor_idx_to_name.items()
                   if name.lower().removesuffix("_camera") == args.camera.lower().removesuffix("_camera")]
        if not matches:
            raise ValueError(f"Camera {args.camera!r} absent; checkpoint sensors: {backend.sensor_idx_to_name}")
        ids = image_dataset.cameras.metadata["sensor_idxs"].reshape(-1)
        camera_indices = (ids == matches[0]).nonzero().reshape(-1).tolist()
        selected = camera_indices[args.sample_index:args.sample_index+args.samples*args.sample_stride:args.sample_stride]
        if len(selected) != args.samples:
            raise ValueError(f"Only {len(camera_indices)} frames for camera {args.camera!r}; requested slice unavailable")
        report.update(checkpoint_path=backend.checkpoint_path, checkpoint_step=backend.step,
            gaussian_count=int(backend.model.gauss_params["means"].shape[0]),
            actor_count=int(backend.model.dynamic_actors.n_actors), sensor_idx_to_name=backend.sensor_idx_to_name)
        if args.actor_id is not None and not 0 <= args.actor_id < backend.model.dynamic_actors.n_actors:
            raise ValueError("actor-id is outside the loaded scene")
        actor_ids = (list(range(backend.model.dynamic_actors.n_actors)) or [None]) if args.all_actors else [args.actor_id]
        tasks = [(camera_index, actor_id) for camera_index in selected for actor_id in actor_ids]
        for sample_number, (camera_index, actor_id) in enumerate(tasks):
            args.actor_id = actor_id
            sample_dir = args.output / f"sample_{sample_number:03d}"
            sample_dir.mkdir(exist_ok=True)
            report["samples"].append(validate_sample(drive, image_dataset, lidar_dataset, camera_index, args, sample_dir))
            save_json(args.output / "validation.json", report)
        report.update(status="NUMERIC_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED", numeric_checks_passed=True,
            visual_quality="UNREVIEWED", note="Numeric checks validate execution and editing mechanics; reconstruction quality requires separate held-out metrics and visual review.")
        save_json(args.output / "validation.json", report)
        print(json.dumps({"status": report["status"], "report": str((args.output / "validation.json").resolve())}), flush=True)
        return 0
    except Exception as error:
        report.update(status="FAIL", numeric_checks_passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save_json(args.output / "validation.json", report)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
