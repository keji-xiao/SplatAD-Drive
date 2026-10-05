"""Reproducible entry points for reference validation and official experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .io import save_json, save_png, save_ply, to_jsonable


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reference_scene(path, device="cpu"):
    from .scene import GaussianScene, ActorTrajectory
    contents = _load_json(path)
    contents["trajectories"] = {int(key):ActorTrajectory(torch.tensor(item["timestamps"],dtype=torch.float64),torch.tensor(item["poses"],dtype=torch.float32))
                               for key,item in contents.get("trajectories",{}).items()}
    return GaussianScene(**contents).to(device)


def _load_drive(args):
    from .drive import SplatADDrive
    if args.load_config:
        return SplatADDrive.from_checkpoint(args.load_config)
    return SplatADDrive.from_scene(load_reference_scene(args.scene,args.device))


def _load_requests(path, device):
    from .camera_renderer import CameraRequest
    from .lidar_renderer import LiDARRequest
    values = _load_json(path)
    requests = {}
    for sensor, factory in (("camera",CameraRequest),("lidar",LiDARRequest)):
        if sensor not in values:
            continue
        fields = dict(values[sensor])
        for name in ("T_world_camera","K","T_world_lidar","azimuth","elevation","linear_velocity","angular_velocity","time_offsets"):
            if fields.get(name) is not None:
                fields[name] = torch.tensor(fields[name],dtype=torch.float32,device=device)
        requests[sensor] = factory(**fields)
    if not requests:
        raise ValueError("request JSON must contain camera and/or lidar")
    return requests


def _apply_edits(drive, path):
    if path is None:
        return
    for edit in _load_json(path):
        if "visible" in edit:
            if not isinstance(edit["visible"],bool):
                raise ValueError("visible must be boolean")
            drive.set_actor_visible(int(edit["actor_id"]),edit["visible"])
        if "T_world_actor" in edit:
            drive.set_actor_pose(int(edit["actor_id"]),float(edit["timestamp"]),torch.tensor(edit["T_world_actor"],dtype=torch.float32))


def _render(args):
    from .demo import depth_colors, overlay
    drive = _load_drive(args); requests = _load_requests(args.requests,args.device)
    _apply_edits(drive,args.edits)
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    results = {}
    with torch.inference_mode():
        for name,request in requests.items():
            results[name] = getattr(drive,"render_"+name)(request)
        if "camera" in results:
            save_png(output/"rgb.png",results["camera"]["rgb"])
            save_png(output/"depth.png",depth_colors(results["camera"]["depth"]))
        if "lidar" in results:
            lidar = results["lidar"]
            valid = (lidar["hit_probability"]>.1) & (lidar["ray_drop"]<.5) & (lidar["range"]>0)
            save_ply(output/"lidar.ply",lidar["points"][valid])
            save_png(output/"range.png",depth_colors(lidar["range"],valid))
        if len(results)==2:
            save_png(output/"overlay.png",overlay(results["camera"]["rgb"],lidar["points"],requests["camera"],requests["lidar"],valid,
                                                  compensate_lidar_translation=True))
    import numpy as np
    for name,result in results.items():
        np.savez_compressed(output/f"{name}.npz",**{key:value.detach().cpu().numpy() for key,value in result.items() if isinstance(value,torch.Tensor)})
    save_json(output/"render_manifest.json",{"backend":"official_neurad" if args.load_config else "reference",
        "load_config":args.load_config,"scene":args.scene,"requests":_load_json(args.requests),
        "edits":_load_json(args.edits) if args.edits else [],
        "overlay":{"mode":"nominal camera pose, per-ray LiDAR linear-translation compensation",
                   "sensor_time_difference_s":requests["camera"].timestamp-requests["lidar"].timestamp if len(requests)==2 else None,
                   "limits":"static geometry diagnostic; no angular velocity, camera-row motion or cross-sensor actor-motion compensation"}})
    return {"output":str(output.resolve())}


def _profile(args):
    from .benchmark import benchmark_renderer
    if args.requests:
        if not (args.scene or args.load_config):
            raise ValueError("profiling requests requires --scene or --load-config")
        drive = _load_drive(args); requests = _load_requests(args.requests,args.device)
        gaussian_count = len(drive.backend.model.gauss_params["means"]) if args.load_config else len(drive.backend.scene.means)
    else:
        if args.load_config or args.scene:
            raise ValueError("custom scene/checkpoint profiling requires --requests")
        from .demo import make_demo
        from .drive import SplatADDrive
        scene,camera,lidar = make_demo(args.device)
        drive = SplatADDrive.from_scene(scene); requests = {"camera":camera,"lidar":lidar}
        gaussian_count = len(scene.means)
    timing_device = str(drive.backend.device) if args.load_config else args.device
    results = {"backend":"official_neurad" if args.load_config else "reference", "gaussian_count":gaussian_count,
               "load_config":args.load_config,"scene":args.scene,"requests":args.requests}
    if args.requests:
        results["request_values"] = _load_json(args.requests)
    for name,request in requests.items():
        counts = {"pixels_per_frame":request.width*request.height} if name=="camera" else {"rays_per_frame":len(request.azimuth)*len(request.elevation)}
        results[name] = benchmark_renderer(lambda: getattr(drive,"render_"+name)(request),device=timing_device,warmup=args.warmup,iterations=args.iterations,**counts)
    save_json(args.output,results)
    return results


def _evaluate(args):
    """GT evaluation_mask marks observed LiDAR rays, independently of hit/drop.

    Its shape matches GT range, or the leading dimensions of GT points when
    range is absent. Organized/flattened point arrays use the same row-major
    ray order. The target mask applies to both clouds in paired Chamfer.
    Without it, the existing all-rays evaluation policy is preserved.
    """
    import numpy as np
    from .evaluator import camera_metrics,lidar_metrics,chamfer_distance
    with np.load(args.prediction,allow_pickle=False) as pred, np.load(args.target,allow_pickle=False) as gt:
        results = {}
        evaluation_mask = None
        if "evaluation_mask" in gt:
            mask = np.asarray(gt["evaluation_mask"])
            if "range" in gt:
                expected_shape = gt["range"].shape
            elif "points" in gt and gt["points"].ndim >= 2 and gt["points"].shape[-1] == 3:
                expected_shape = gt["points"].shape[:-1]
            else:
                raise ValueError("GT evaluation_mask requires target LiDAR range or XYZ points")
            if mask.shape != expected_shape:
                raise ValueError("GT evaluation_mask shape must match target LiDAR ray dimensions")
            numeric = np.issubdtype(mask.dtype,np.integer) or np.issubdtype(mask.dtype,np.floating)
            if mask.dtype != np.bool_ and not (numeric and np.isfinite(mask).all() and np.isin(mask,[0,1]).all()):
                raise ValueError("GT evaluation_mask must contain boolean or finite binary values")
            evaluation_mask = mask.astype(bool,copy=False)
        if "rgb" in pred and "rgb" in gt:
            results["camera"] = camera_metrics(pred["rgb"],gt["rgb"])
        if "range" in pred and "range" in gt:
            results["lidar"] = lidar_metrics(pred["range"],gt["range"],target_hit=gt["hit"] if "hit" in gt else None,
                pred_drop_probability=pred["ray_drop"] if "ray_drop" in pred else None,
                pred_intensity=pred["intensity"] if "intensity" in pred else None,target_intensity=gt["intensity"] if "intensity" in gt else None,
                evaluation_mask=evaluation_mask)
        if "points" in pred and "points" in gt:
            pred_points,gt_points=pred["points"].reshape(-1,3),gt["points"].reshape(-1,3)
            pred_valid=np.isfinite(pred_points).all(1)&(np.linalg.norm(pred_points,axis=1)>0)
            gt_valid=np.isfinite(gt_points).all(1)&(np.linalg.norm(gt_points,axis=1)>0)
            if evaluation_mask is not None:
                if len(pred_points) != evaluation_mask.size or len(gt_points) != evaluation_mask.size:
                    raise ValueError("GT evaluation_mask requires one corresponding predicted and target point per ray")
                pred_valid &= evaluation_mask.reshape(-1)
                gt_valid &= evaluation_mask.reshape(-1)
            if "ray_drop" in pred: pred_valid &= pred["ray_drop"].reshape(-1)<.5
            if "hit" in gt: gt_valid &= gt["hit"].reshape(-1).astype(bool)
            results["chamfer_bidirectional_sum_m2"] = chamfer_distance(pred_points[pred_valid],gt_points[gt_valid])
        if not results:
            raise ValueError("NPZ files have no matching rgb, range or points arrays")
    save_json(args.output,results)
    return results


def _inspect(args):
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    drive=_load_drive(args)
    if args.scene:
        scene=drive.backend.scene
        means,colors,ids=scene.means,scene.colors,scene.actor_ids
        static_id=-1
    else:
        model=drive.backend.model
        means=model.gauss_params["means"]
        ids=model.gauss_params["id"].reshape(-1).long()
        colors=model.gauss_params["features_dc"][:,:3].clamp(0,1)
        static_id=model.dynamic_actors.n_actors
    counts={}
    for actor_id in torch.unique(ids).tolist():
        name="static_gaussians" if actor_id==static_id else f"actor_{actor_id:03d}"
        selected=ids==actor_id
        save_ply(output/f"{name}.ply",means[selected],colors[selected])
        counts[name]=int(selected.sum())
    save_json(output/"manifest.json",{"counts":counts,"static_id":static_id,"dynamic_frame":"actor-local","static_frame":"parser-world",
        "colors":"reference RGB" if args.scene else "first three learned feature channels clipped to [0,1]; not decoded RGB"})
    return counts


def main(argv=None):
    parser=argparse.ArgumentParser(prog="splatad-drive",description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    doctor=commands.add_parser("doctor"); doctor.add_argument("--output",default="outputs/environment.json"); doctor.add_argument("--check-upstream",action="store_true")
    demo=commands.add_parser("demo"); demo.add_argument("--output",default="outputs/demo"); demo.add_argument("--device",default="cpu"); demo.add_argument("--fit-steps",type=int,default=20)
    audit=commands.add_parser("audit"); audit.add_argument("--data",default="data/pandaset"); audit.add_argument("--sequence",default="028"); audit.add_argument("--output",default="outputs/audit"); audit.add_argument("--frames",type=int,default=20)
    audit.add_argument("--all-camera-overlays",action="store_true")
    train=commands.add_parser("train"); train.add_argument("--config",default="configs/smoke.yaml"); train.add_argument("--execute",action="store_true"); train.add_argument("--audit"); train.add_argument("--alignment-review")
    for name in ("render","inspect","profile"):
        item=commands.add_parser(name)
        source=item.add_mutually_exclusive_group(required=name!="profile")
        source.add_argument("--scene"); source.add_argument("--load-config")
        item.add_argument("--device",default="cpu"); item.add_argument("--output",default=f"outputs/{name}" if name!="profile" else "outputs/profile.json")
        if name!="inspect": item.add_argument("--requests",required=name=="render")
        if name=="render": item.add_argument("--edits")
        if name=="profile":
            item.add_argument("--warmup",type=int,default=10); item.add_argument("--iterations",type=int,default=50)
    evaluate=commands.add_parser("evaluate"); evaluate.add_argument("--prediction",required=True); evaluate.add_argument("--target",required=True); evaluate.add_argument("--output",default="outputs/evaluation.json")
    official=commands.add_parser("eval-official"); official.add_argument("--load-config",required=True); official.add_argument("--output",default="outputs/eval_official.json"); official.add_argument("--include-fid",action="store_true")
    official.add_argument("--seed",type=int,default=42)
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            from .environment import environment_report
            result=environment_report(check_upstream=args.check_upstream); save_json(args.output,result)
        elif args.command=="demo":
            from .demo import run_demo
            result=run_demo(args.output,args.device,args.fit_steps)
        elif args.command=="audit":
            from .data.pandaset import audit_pandaset
            result=audit_pandaset(args.data,args.output,sequence=args.sequence,max_frames=args.frames,all_camera_overlays=args.all_camera_overlays)
        elif args.command=="train":
            from .training import load_training_config,build_train_command,run_training
            config=load_training_config(args.config)
            if args.execute: return run_training(config,audit_path=args.audit,alignment_review=args.alignment_review)
            result={"execute":False,"argv":build_train_command(config),"note":"Pass --execute with matching audited dataset and reviewed overlays."}
        elif args.command=="render": result=_render(args)
        elif args.command=="inspect": result=_inspect(args)
        elif args.command=="profile": result=_profile(args)
        elif args.command=="evaluate": result=_evaluate(args)
        elif args.command=="eval-official":
            from .backends.neurad import NeuradBackend
            backend = NeuradBackend.from_config(args.load_config, test_mode="test")
            torch.manual_seed(args.seed)
            import numpy as np
            np.random.seed(args.seed)
            with torch.no_grad():
                metrics = backend.pipeline.get_average_eval_image_metrics(step=None if args.include_fid else backend.step, get_std=True)
            result = {"experiment_name":backend.config.experiment_name, "method_name":"splatad",
                      "checkpoint":str(backend.checkpoint_path), "checkpoint_step":backend.step,
                      "source":"official pipeline get_average_eval_image_metrics", "results":metrics}
            result["fid_requested"] = args.include_fid
            result["evaluation_seed"] = args.seed
            result["timing_note"] = "Pipeline timing fields are informational; use the synchronized profile command for performance comparisons."
            result["metric_notes"] = {"depth_median_l2":"median squared range residual (m^2), not metres",
                "chamfer_distance":"official sum of both directed squared-distance sums divided by GT point count; distinct from the NPZ evaluator's sum of directed means"}
            manager = backend.pipeline.datamanager
            result["evaluation_scope"] = {"camera_images":len(manager.eval_dataset), "lidar_sweeps":len(manager.eval_lidar_dataset),
                "sequence":str(manager.config.dataparser.sequence), "dataset_end_fraction":manager.config.dataparser.dataset_end_fraction,
                "train_split_fraction":manager.config.dataparser.train_split_fraction}
            save_json(args.output,result)
        print(json.dumps(to_jsonable(result),ensure_ascii=False,indent=2))
        return 0
    except (ValueError,RuntimeError,FileNotFoundError,ImportError) as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr)
        return 2
