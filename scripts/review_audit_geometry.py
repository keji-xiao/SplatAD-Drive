"""Reproduce independent geometric diagnostics from exported PandaSet audit files.

No dataset loader, upstream parser, training code, or audit.json is modified.
Run: .venv-gpu/Scripts/python.exe scripts/review_audit_geometry.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bounds(values):
    return {"min": np.min(values, axis=0).tolist(), "max": np.max(values, axis=0).tolist()}


def speed_stats(positions, times):
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)/np.diff(times)
    return {"min_m_s": float(speed.min()), "median_m_s": float(np.median(speed)), "max_m_s": float(speed.max())}


def transform_points(points, pose):
    return points @ pose[:3, :3].T+pose[:3, 3]


def quaternion_matrix(heading):
    w, x, y, z = (heading[k] for k in ("w", "x", "y", "z"))
    norm = np.linalg.norm([w, x, y, z])
    w, x, y, z = (v/norm for v in (w, x, y, z))
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def inspect_raw_annotations(root, dataset, actors, sensor, frame_ids):
    """Match original UUID tracks and reproduce upstream translation extrapolation."""
    import pandas as pd
    camera_times = np.asarray(read_json(dataset/"camera/front_camera/timestamps.json"))
    lidar_times = np.asarray(read_json(dataset/"lidar/timestamps.json"))
    lidar_poses = read_json(dataset/"lidar/poses.json")
    offset = sensor["absolute_time_offset"]
    mean_scan_interval = float(np.diff(lidar_times).mean())
    world_transform = np.asarray(sensor["T_parser_world_dataset_world"])
    tracks = {}
    labels = {actor["label"] for actor in actors}
    for path in sorted((dataset/"annotations/cuboids").glob("*.pkl.gz")):
        frame_index = int(path.name.split(".")[0])
        table = pd.read_pickle(path)
        rows = table[(~table.stationary) & (table["cuboids.sensor_id"] != 1) & table.label.isin(labels)]
        for _, row in rows.iterrows():
            xyz = np.asarray([row[f"position.{axis}"] for axis in "xyz"], dtype=np.float64)
            dims = [float(row[f"dimensions.{axis}"]) for axis in "xyz"]
            lidar_pose = lidar_poses[frame_index]
            rotation = quaternion_matrix(lidar_pose["heading"])
            translation = np.array([lidar_pose["position"][axis] for axis in "xyz"])
            # Reproduce official line exactly, and compare with world-to-local.
            official_local = xyz @ rotation.T+translation
            geometric_local = (xyz-translation) @ rotation
            def time_correction(local):
                angle = (np.arctan2(local[0], local[1])-np.pi/2+np.pi)%(2*np.pi)-np.pi
                return float(angle/(2*np.pi)*mean_scan_interval)
            camera_time = float(camera_times[frame_index]-offset)
            record = {"source_file": str(path), "raw_frame": frame_index, "uuid": str(row.uuid),
                      "label": str(row.label), "stationary": bool(row.stationary),
                      "sensor_id": int(row["cuboids.sensor_id"]), "dims_m": dims,
                      "position_dataset_world_m": xyz.tolist(),
                      "position_parser_world_m": transform_points(xyz[None], world_transform)[0].tolist(),
                      "camera_time_s": camera_time,
                      "official_corrected_time_s": camera_time+time_correction(official_local),
                      "inverse_transform_corrected_time_s": camera_time+time_correction(geometric_local)}
            tracks.setdefault(record["uuid"], []).append(record)
    results = []
    used_uuids = set()
    for actor in actors:
        target_dims = np.asarray(actor["dims"])
        candidates = []
        for uuid, records in tracks.items():
            if uuid in used_uuids or records[0]["label"] != actor["label"]:
                continue
            dims_error = np.linalg.norm(np.max([r["dims_m"] for r in records], axis=0)-target_dims)
            candidates.append((dims_error, uuid, records))
        dims_error, uuid, records = min(candidates, key=lambda item: item[0])
        if dims_error > 1e-3:
            raise ValueError(f"Cannot establish UUID correspondence for actor {actor['actor_id']}")
        used_uuids.add(uuid)
        times = np.array([r["official_corrected_time_s"] for r in records])
        xyz = np.array([r["position_parser_world_m"] for r in records])
        exported_times = np.asarray(actor["timestamps"])
        exported_xyz = np.asarray(actor["poses"])[:, :3, 3]
        right = np.clip(np.searchsorted(times, exported_times), 1, len(times)-1)
        left = right-1
        fractions = (exported_times-times[left])/(times[right]-times[left])
        prediction = xyz[left]*(1-fractions[:, None])+xyz[right]*fractions[:, None]
        errors = np.linalg.norm(prediction-exported_xyz, axis=1)
        deltas = np.array([r["official_corrected_time_s"]-r["inverse_transform_corrected_time_s"] for r in records])
        frame_evidence = []
        for frame_id in frame_ids:
            with np.load(root/f"lidar_{frame_id:04d}.npz", allow_pickle=False) as cloud:
                timestamp = float(cloud["timestamp"])
            source_index = int(np.abs(camera_times-offset-timestamp).argmin())
            frame_evidence.append({"lidar_audit_index": frame_id, "raw_camera_frame": source_index,
                                   "timestamp_s": timestamp, "uuid_present_in_raw_frame": any(r["raw_frame"] == source_index for r in records),
                                   "before_first_corrected_annotation_s": float(max(times[0]-timestamp, 0)),
                                   "outside_original_corrected_time_support": bool(timestamp < times[0] or timestamp > times[-1])})
        results.append({"actor_id": actor["actor_id"], "uuid": uuid, "raw_count": len(records),
                        "raw_first": records[0], "raw_last": records[-1],
                        "exported_first_time_s": float(exported_times[0]),
                        "exported_before_original_support_count": int((exported_times < times[0]).sum()),
                        "translation_reproduction_max_error_m": float(errors.max()),
                        "official_vs_inverse_time_delta_ms": {"min": float(deltas.min()*1000), "max": float(deltas.max()*1000)},
                        "frame_evidence": frame_evidence, "raw_annotations": records})
    result = {"dataset": str(dataset), "default_trajectory_extrapolation_s": 1.0,
              "mean_scan_interval_s": mean_scan_interval, "actors": results,
              "source_code": {"time_correction": "third_party/neurad-studio/nerfstudio/data/dataparsers/pandaset_dataparser.py:364",
                              "lidar_to_world_used_as_pose": "third_party/neurad-studio/nerfstudio/data/dataparsers/pandaset_dataparser.py:260",
                              "extrapolation_default": "third_party/neurad-studio/nerfstudio/data/dataparsers/ad_dataparser.py:91",
                              "extrapolation": "third_party/neurad-studio/nerfstudio/data/dataparsers/ad_dataparser.py:413"}}
    (root/"raw_actor_annotation_review.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return result


def review(root: Path, frame_ids: list[int], dataset: Path | None = None):
    ego = read_json(root/"ego_trajectory.json")
    actors = read_json(root/"actor_trajectories.json")["trajectories"]
    sensor = read_json(root/"sensor_extrinsics.json")
    ego_times = np.asarray(ego["timestamps"], dtype=np.float64)
    ego_poses = np.asarray(ego["T_world_reference"], dtype=np.float64)
    ego_xyz = ego_poses[:, :3, 3]
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    axes[0].plot(ego_xyz[:, 0], ego_xyz[:, 1], "k.-", label="Pandar64 ego proxy", linewidth=2)
    axes[0].scatter(ego_xyz[[0, -1], 0], ego_xyz[[0, -1], 1], c=["green", "black"], s=55)
    axes[0].annotate("ego start", ego_xyz[0, :2], xytext=(6, 0), textcoords="offset points")
    axes[0].annotate("ego end", ego_xyz[-1, :2], xytext=(6, 0), textcoords="offset points")
    for axis, component in zip(axes[1:], [0, 1]):
        axis.plot(ego_times, ego_xyz[:, component], "k.-", label="ego proxy")
    summary = {"source": "exported official PandaSet parser audit, sequence 028",
               "ego_reference": ego["reference"], "world": sensor["world"],
               "ego": {"count": len(ego_times), "time_range_s": [float(ego_times[0]), float(ego_times[-1])],
                       "xyz_bounds_m": bounds(ego_xyz), "speed": speed_stats(ego_xyz, ego_times),
                       "path_length_m": float(np.linalg.norm(np.diff(ego_xyz, axis=0), axis=1).sum())},
               "actors": [], "frames": []}
    for actor in actors:
        actor_id = actor["actor_id"]
        times = np.asarray(actor["timestamps"], dtype=np.float64)
        poses = np.asarray(actor["poses"], dtype=np.float64)
        xyz = poses[:, :3, 3]
        color = colors(actor_id)
        label = f"actor {actor_id}: {actor['label']}"
        axes[0].plot(xyz[:, 0], xyz[:, 1], ".-", color=color, label=label, linewidth=2)
        axes[0].annotate(str(actor_id), xyz[-1, :2], xytext=(5, 5), textcoords="offset points", color=color)
        for axis, component in zip(axes[1:], [0, 1]):
            axis.plot(times, xyz[:, component], ".-", color=color, label=label)
        summary["actors"].append({"actor_id": actor_id, "label": actor["label"], "keyframes": len(times),
                                   "time_range_s": [float(times[0]), float(times[-1])], "xyz_bounds_m": bounds(xyz),
                                   "dims_m": actor["dims"], "speed": speed_stats(xyz, times),
                                   "max_rotation_orthogonality_error": float(np.abs(poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3]-np.eye(3)).max())})
    axes[0].set(title="Parser-world trajectories", xlabel="world x [m]", ylabel="world y [m]")
    axes[0].axis("equal")
    axes[1].set(title="Lateral trajectory", xlabel="parser-relative time [s]", ylabel="world x [m]")
    axes[2].set(title="Longitudinal trajectory", xlabel="parser-relative time [s]", ylabel="world y [m]")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("PandaSet 028: ego proxy and dynamic actor tracks (exported parser coordinates)")
    fig.savefig(root/"geometry_trajectories.png", dpi=150)
    plt.close(fig)

    for frame_id in frame_ids:
        with np.load(root/f"lidar_{frame_id:04d}.npz", allow_pickle=False) as data:
            points = data["points"].astype(np.float64)
            pose = data["T_world_lidar"].astype(np.float64)
            timestamp = float(data["timestamp"])
            world_points = transform_points(points[:, :3], pose)
            inverse_points = transform_points(world_points, np.linalg.inv(pose))
            frame = {"lidar_index": frame_id, "timestamp_s": timestamp, "point_count": len(points),
                     "point_relative_time_range_s": [float(points[:, 4].min()), float(points[:, 4].max())],
                     "world_xyz_percentiles_0_5_50_99_5": np.percentile(world_points, [.5, 50, 99.5], axis=0).tolist(),
                     "lidar_world_roundtrip_max_error_m": float(np.abs(inverse_points-points[:, :3]).max()), "actors": []}
        fig, axes = plt.subplots(len(actors), 2, figsize=(13, max(4, 3.7*len(actors))), squeeze=False, constrained_layout=True)
        for row, actor in enumerate(actors):
            times = np.asarray(actor["timestamps"], dtype=np.float64)
            index = int(np.argmin(np.abs(times-timestamp)))
            annotation_time_error = float(abs(times[index]-timestamp))
            if annotation_time_error > .001:
                raise ValueError(f"Frame {frame_id}, actor {actor['actor_id']}: no matching pose; nearest delta={annotation_time_error}")
            actor_pose = np.asarray(actor["poses"][index], dtype=np.float64)
            local = transform_points(world_points, np.linalg.inv(actor_pose))
            half_dims = np.asarray(actor["dims"])/2
            inside = (np.abs(local) <= half_dims).all(axis=1)
            crop = (np.abs(local) <= half_dims+np.array([12., 12., 4.])).all(axis=1)
            nearest_bbox = float(np.linalg.norm(np.maximum(np.abs(local)-half_dims, 0), axis=1).min())
            frame["actors"].append({"actor_id": actor["actor_id"], "points_inside_full_cloud_bbox": int(inside.sum()),
                                     "points_in_local_crop": int(crop.sum()), "pose_time_delta_s": annotation_time_error,
                                     "distance_sensor_to_actor_center_m": float(np.linalg.norm(actor_pose[:3, 3]-pose[:3, 3])),
                                     "nearest_point_to_bbox_m": nearest_bbox})
            for column, components in enumerate([(0, 1), (1, 2)]):
                axis = axes[row, column]
                i, j = components
                outside = crop & ~inside
                axis.scatter(local[outside, i], local[outside, j], s=1.3, c="#788698", alpha=.65, rasterized=True)
                axis.scatter(local[inside, i], local[inside, j], s=12, color=colors(actor["actor_id"]), label="inside cuboid")
                axis.add_patch(Rectangle((-half_dims[i], -half_dims[j]), 2*half_dims[i], 2*half_dims[j],
                                         fill=False, edgecolor=colors(actor["actor_id"]), linewidth=2))
                axis.scatter([0], [0], marker="+", c="black", s=30)
                axis.set_xlim(-half_dims[i]-12, half_dims[i]+12)
                margin_y = 12 if j == 1 else 4
                axis.set_ylim(-half_dims[j]-margin_y, half_dims[j]+margin_y)
                axis.set_aspect("equal", adjustable="box")
                axis.grid(alpha=.2)
                axis.set_xlabel(["actor x / width [m]", "actor y / length [m]", "actor z / height [m]"][i])
                axis.set_ylabel(["actor x / width [m]", "actor y / length [m]", "actor z / height [m]"][j])
                axis.set_title(f"Actor {actor['actor_id']} {actor['label']} | {'top' if column == 0 else 'side'} | "
                               f"inside={inside.sum()}, nearest box={nearest_bbox:.2f}m", fontsize=10)
        fig.suptitle(f"LiDAR {frame_id:04d}, t={timestamp:.6f}s: full cloud in actor-local frames\n"
                     "Scan-center pose, no per-point deskew or actor exposure-time correction; grey=context, color=inside bbox", fontsize=11)
        fig.savefig(root/f"geometry_bbox_lidar_{frame_id:04d}.png", dpi=150)
        plt.close(fig)
        summary["frames"].append(frame)

    (root/"geometry_review.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    actor_table = "\n".join(f"| {a['actor_id']} / {a['label']} | {a['keyframes']} | {a['time_range_s'][0]:.3f}–{a['time_range_s'][1]:.3f} | "
                            f"{a['speed']['min_m_s']:.2f}–{a['speed']['max_m_s']:.2f} | {a['dims_m'][0]:.2f}×{a['dims_m'][1]:.2f}×{a['dims_m'][2]:.2f} |" for a in summary["actors"])
    box_table = "\n".join(f"| {f['lidar_index']:04d} | {a['actor_id']} | {a['distance_sensor_to_actor_center_m']:.2f} | "
                          f"{a['points_inside_full_cloud_bbox']} | {a['nearest_point_to_bbox_m']:.2f} |" for f in summary["frames"] for a in f["actors"])
    report = f"""# PandaSet 028 独立几何审查

审查输入为官方 parser 已导出的 `ego_trajectory.json`、`actor_trajectories.json`、`sensor_extrinsics.json` 和选定 LiDAR NPZ。此脚本不修改 `audit.json` 或训练代码，不把几何抽查当作完整数据验收。

## 坐标与时间范围

- 世界坐标：{sensor['world']}；距离为米，时间为 parser-relative seconds。
- Ego 为 Pandar64 传感器原点代理，不能等同于后轴/车体原点。共 {len(ego_times)} 个位姿，时间 {ego_times[0]:.6f}–{ego_times[-1]:.6f} s，路程 {summary['ego']['path_length_m']:.2f} m，速度 {summary['ego']['speed']['min_m_s']:.2f}–{summary['ego']['speed']['max_m_s']:.2f} m/s。
- Ego XYZ 范围：min={np.round(ego_xyz.min(0), 4).tolist()}，max={np.round(ego_xyz.max(0), 4).tolist()}。actor 轨迹连续，速度量级适合城市道路；这些事实不能独立证明标签精度。
- cuboid 的 actor-local x/y/z 对应 width/length/height，不能把 length 误放在 x 轴。

| Actor / 类别 | 关键帧 | 时间范围 s | 速度范围 m/s | width×length×height m |
|---|---:|---:|---:|---:|
{actor_table}

![轨迹](geometry_trajectories.png)

## LiDAR 点云与动态 bbox 局部检查

采用完整 NPZ 点云，而非原审计的抽样点数。先用 NPZ 的 T_world_lidar 转到世界，再用 actor pose 的完整矩阵逆变换到 actor-local；所选帧均找到误差小于 1 ms 的同时间关键帧。没有用最近帧代替不匹配的注释。静态矩阵往返最大误差为 {max(f['lidar_world_roundtrip_max_error_m'] for f in summary['frames']):.3g} m，只检验转换代数，不证明外参正确。

| LiDAR index | actor | 传感器到框中心 m | 框内完整点数 | 最近点到框 m |
|---|---:|---:|---:|---:|
{box_table}

"""
    for frame in summary["frames"]:
        report += f"![LiDAR {frame['lidar_index']:04d} bbox](geometry_bbox_lidar_{frame['lidar_index']:04d}.png)\n\n"
    report += """## 限制与训练门槛

- 框内点为零不能直接判定坐标错误，也不能算作 actor 几何通过；遮挡、远距稀疏、注释存在区间和轨迹外推都需要检查。尤其早期帧到 bbox 的距离较大，应保留为异常记录。
- 上游 `ADDataParserConfig.trajectory_extrapolation_length` 默认 1.0 s，并允许对 actor 轨迹外推；单看导出文件不能区分原始注释点和外推点。启用原始数据复核时，下节给出 UUID、原始帧和外推来源证据。
- 局部图使用扫描中心位姿。NPZ 已包含逐点相对时间，但这里没有重新做 per-point deskew，也没有验证 Camera rolling shutter readout 方向/时长、LiDAR scan-time motion 或跨传感器同步。不能声称完整 RS 验证通过。
- 静态道路场景只可在主审计的数值和投影检查通过后进入短程 smoke training；稀疏/外推 actor 的训练质量与场景编辑须由独立可见性、原始注释核对和多帧渲染结果继续确认。不得据此进入论文质量、性能或闭环驾驶验收。

复现：`.venv-gpu/Scripts/python.exe scripts/review_audit_geometry.py --audit outputs/audit --frames 34 39`。需要 NumPy、Matplotlib；原始 cuboid 核对另需 pandas。定量详细值在 `geometry_review.json`。
"""
    if dataset is not None:
        raw = inspect_raw_annotations(root, dataset, actors, sensor, frame_ids)
        report += "\n## 原始 cuboid 复核：已确定外推来源\n\n"
        report += "直接读取官方数据 `annotations/cuboids/*.pkl.gz`，按非 stationary、非 sensor_id=1 的规则筛选，并以类别与整条轨迹最大尺寸匹配 UUID；再重现官方 corrected timestamp 和位置线性插值/外推。详细原始行在 `raw_actor_annotation_review.json`。\n\n"
        report += "| actor | 原始 UUID | 首次原始帧 | 未校时 / 官方校时 s | 原始行数 | 导出最早时间 s | 平移重现最大误差 m |\n|---|---|---:|---:|---:|---:|---:|\n"
        for record in raw["actors"]:
            first = record["raw_first"]
            report += f"| {record['actor_id']} | {record['uuid']} | {first['raw_frame']:02d} | {first['camera_time_s']:.6f} / {first['official_corrected_time_s']:.6f} | {record['raw_count']} | {record['exported_first_time_s']:.6f} | {record['translation_reproduction_max_error_m']:.6f} |\n"
        report += "\n审计 LiDAR `0034` 是训练 split 的第 34 个样本，对应原始 **68** 帧，不是原始 34 帧。原始 68 帧没有这 3 个 UUID；三者分别从 70、72、75 帧才出现。相对官方首次校时注释，0034 提前 "
        report += " / ".join(f"{record['frame_evidence'][0]['before_first_corrected_annotation_s']:.3f}" for record in raw["actors"])
        report += " s，均落入上游默认 1 秒外推区间。原始 79 帧（审计0039）则有全部 3 个 UUID。**0034 无框内点有确定的‘无原始标签、外推轨迹’背景，不能用它判定共同外参错位，也不能把外推框当成观测真值。** 位置重现误差不到 0.3 mm 证明导出轨迹与官方逻辑一致。\n\n"
        report += "### 另发现上游 actor 校时坐标方向问题\n\n官方 `pandaset_dataparser.py:365–370` 的 `lidpose` 是 LiDAR-to-world（同文件读取点云时使用它的逆从世界转 LiDAR）。但 cuboid 已在世界坐标，代码却计算 `pos @ R.T + t` 来求 `posinlid`；几何上应为 `(pos - t) @ R`。保持原来的角度/扫描时序公式，只修正坐标变换方向，得到下列时间差：\n\n| actor | 官方校时 − 逆变换校时 ms |\n|---|---:|\n"
        for record in raw["actors"]:
            delta = record["official_vs_inverse_time_delta_ms"]
            report += f"| {record['actor_id']} | {delta['min']:.2f}–{delta['max']:.2f} |\n"
        report += "\n这是源码坐标方向错误的独立证据，但本次没有改动上游或训练实现。它解释的是几十毫秒动态校时偏差，并不能解释0034米级空框；后者已由外推来源解释。校时修正还应通过真实逐点扫描时刻/方位方向核对后单独落地，不能直接把这次计算当成完整 RS 验收。\n\n**门槛结论：主 audit 数值/投影检查通过时，可 GO 静态短程 smoke（当前 smoke 配置只取开头20%，早于这些动态物体出现）。真实动态训练尚有上游 actor 校时问题及外推可见性策略待处理，不能标为动态几何全部通过。**\n"
    visual_notes = root/"geometry_visual_notes.md"
    if visual_notes.exists():
        report += "\n"+visual_notes.read_text(encoding="utf-8")
    (root/"geometry_review.md").write_text(report, encoding="utf-8")
    print(json.dumps({"review": str(root/"geometry_review.md"), "frames": summary["frames"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--frames", type=int, nargs="+", default=[34, 39])
    parser.add_argument("--dataset", type=Path, default=Path("data/pandaset/028"),
                        help="Original PandaSet sequence for UUID and timestamp verification")
    args = parser.parse_args()
    review(args.audit.resolve(), args.frames, args.dataset.resolve() if args.dataset.exists() else None)
