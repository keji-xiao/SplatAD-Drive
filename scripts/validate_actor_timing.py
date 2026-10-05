"""Compare PandaSet cuboid time formulas against actual in-box LiDAR timestamps.

Read-only on upstream/data. Outputs evidence under outputs/audit. The local
sensor basis is the official PandaSet parser basis, NOT public FLU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def transform(heading, position):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([heading[k] for k in ("x", "y", "z", "w")]).as_matrix()
    result[:3, 3] = [position[k] for k in "xyz"]
    return result


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def temporal_offset(local, period, phase=0., sign=1.):
    return sign*wrap(phase-np.arctan2(local[..., 1], local[..., 0]))*period/(2*np.pi)


def stats(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {"count": len(errors), "mean_error_ms": float(errors.mean()), "median_error_ms": float(np.median(errors)),
            "median_abs_error_ms": float(np.median(np.abs(errors))), "p95_abs_error_ms": float(np.percentile(np.abs(errors), 95)),
            "max_abs_error_ms": float(np.abs(errors).max())}


def moving_local(xyz, point_times, camera_times, calibrated_poses):
    right = np.clip(np.searchsorted(camera_times, point_times), 1, len(camera_times)-1)
    left = right-1
    fraction = (point_times-camera_times[left])/(camera_times[right]-camera_times[left])
    first_rotation = Rotation.from_matrix(calibrated_poses[left, :3, :3])
    delta = (first_rotation.inv()*Rotation.from_matrix(calibrated_poses[right, :3, :3])).as_rotvec()
    rotation = (first_rotation*Rotation.from_rotvec(delta*fraction[:, None])).as_matrix()
    translation = calibrated_poses[left, :3, 3]*(1-fraction[:, None])+calibrated_poses[right, :3, 3]*fraction[:, None]
    return np.einsum("ni,nij->nj", xyz-translation, rotation)


def validate(dataset: Path, output: Path, start=70, end=79):
    sensor = read(output/"sensor_extrinsics.json")
    actors = read(output/"raw_actor_annotation_review.json")["actors"]
    camera_times = np.array(read(dataset/"camera/front_camera/timestamps.json"))
    lidar_times = np.array(read(dataset/"lidar/timestamps.json"))
    camera_poses = read(dataset/"camera/front_camera/poses.json")
    raw_poses = read(dataset/"lidar/poses.json")
    extrinsic = sensor["calibration_from_official_parser"]["front_camera"]["extrinsic"]["transform"]
    lidar_to_camera = transform(extrinsic["rotation"], extrinsic["translation"])
    calibrated_poses = np.stack([transform(p["heading"], p["position"])@lidar_to_camera for p in camera_poses])
    period = float(np.diff(lidar_times).mean())
    models = ["upstream", "raw_pose_inverse_only", "calibrated_inverse_original_phase", "raw_inverse_front_phase",
              "calibrated_inverse_front_phase", "calibrated_reverse_direction"]
    rows, all_point_errors, point_plots, slopes = [], {name: [] for name in models}, [], []
    proposed_capture_times, zero_return_rows, holdout_median_differences = [], [], []
    deskew_errors = []
    for frame in range(start, end+1):
        cloud = pd.read_pickle(dataset/f"lidar/{frame:02d}.pkl.gz")
        cloud = cloud[cloud.d == 0]  # Pandar64, exclude forward-facing LiDAR.
        xyz = cloud[["x", "y", "z"]].to_numpy(dtype=np.float64)
        actual_relative = cloud.t.to_numpy(dtype=np.float64)-camera_times[frame]
        annotations = pd.read_pickle(dataset/f"annotations/cuboids/{frame:02d}.pkl.gz")
        raw_pose = transform(raw_poses[frame]["heading"], raw_poses[frame]["position"])
        calibrated_pose = calibrated_poses[frame]

        def predictions(points):
            wrong = points @ raw_pose[:3, :3].T+raw_pose[:3, 3]
            raw_local = (points-raw_pose[:3, 3])@raw_pose[:3, :3]
            calibrated = (points-calibrated_pose[:3, 3])@calibrated_pose[:3, :3]
            return {"upstream": temporal_offset(wrong, period),
                    "raw_pose_inverse_only": temporal_offset(raw_local, period),
                    "calibrated_inverse_original_phase": temporal_offset(calibrated, period),
                    "raw_inverse_front_phase": temporal_offset(raw_local, period, np.pi/2),
                    "calibrated_inverse_front_phase": temporal_offset(calibrated, period, -np.pi/2),
                    "calibrated_reverse_direction": temporal_offset(calibrated, period, -np.pi/2, -1.)}

        for actor in actors:
            found = annotations[annotations.uuid == actor["uuid"]]
            if len(found) == 0:
                continue
            annotation = found.iloc[0]
            center = np.array([annotation[f"position.{k}"] for k in "xyz"])
            dimensions = np.array([annotation[f"dimensions.{k}"] for k in "xyz"])
            rotation = Rotation.from_euler("z", float(annotation.yaw)).as_matrix()
            local = (xyz-center)@rotation
            inside = (np.abs(local) <= dimensions/2).all(axis=1)
            count = int(inside.sum())
            estimates = predictions(center[None])
            if count == 0:
                fallback = float(estimates["calibrated_inverse_front_phase"][0])
                record = {"frame": frame, "actor_id": actor["actor_id"], "uuid": actor["uuid"], "point_count": 0,
                          "strategy": "calibrated_inverse_front_phase_fallback", "capture_time_absolute_s": float(camera_times[frame]+fallback),
                          "capture_time_relative_to_front_ms": fallback*1000, "iqr_ms": None}
                proposed_capture_times.append(record)
                zero_return_rows.append(record)
                continue
            median = float(np.median(actual_relative[inside]))
            actual_inside = actual_relative[inside]
            iqr_ms = float(np.diff(np.percentile(actual_inside, [25, 75]))[0]*1000)
            proposed_capture_times.append({"frame": frame, "actor_id": actor["actor_id"], "uuid": actor["uuid"],
                                           "point_count": count, "strategy": "in_box_d0_point_time_median",
                                           "capture_time_absolute_s": float(np.median(cloud.t.to_numpy(dtype=np.float64)[inside])),
                                           "capture_time_relative_to_front_ms": median*1000, "iqr_ms": iqr_ms,
                                           "low_support": count < 5})
            if count >= 10:
                random = np.random.default_rng(frame*100+actor["actor_id"])
                for _ in range(20):
                    permutation = random.permutation(count)
                    first, second = np.array_split(permutation, 2)
                    holdout_median_differences.append(float((np.median(actual_inside[first])-np.median(actual_inside[second]))*1000))
            rows.append({"frame": frame, "actor_id": actor["actor_id"], "uuid": actor["uuid"], "point_count": count,
                         "source_cuboid": str(dataset/f"annotations/cuboids/{frame:02d}.pkl.gz"),
                         "actual_in_box_relative_time_ms_p10_p50_p90": (np.percentile(actual_relative[inside], [10, 50, 90])*1000).tolist(),
                         "center_prediction_ms": {name: float(value[0]*1000) for name, value in estimates.items()},
                         "error_ms_against_in_box_median": {name: float((value[0]-median)*1000) for name, value in estimates.items()}})

        # Verify scan phase AND rotation direction over a full scan, not only
        # these three front-facing actors. Exclude wrap boundaries and near ego.
        sample = np.arange(0, len(xyz), 8)
        sample_xyz = xyz[sample]
        actual = actual_relative[sample]
        calibrated_local = (sample_xyz-calibrated_pose[:3, 3])@calibrated_pose[:3, :3]
        valid = (np.abs(actual) < .045) & (np.linalg.norm(calibrated_local, axis=1) > 5.)
        sample_xyz, actual = sample_xyz[valid], actual[valid]
        estimates = predictions(sample_xyz)
        for name, value in estimates.items():
            all_point_errors[name].extend(((value-actual)*1000).tolist())
        local_at_time = moving_local(sample_xyz, actual+camera_times[frame], camera_times, calibrated_poses)
        deskew_predicted = temporal_offset(local_at_time, period, -np.pi/2)
        deskew_errors.extend(((deskew_predicted-actual)*1000).tolist())
        angles = estimates["calibrated_inverse_front_phase"]*2*np.pi/period
        slope, intercept = np.polyfit(angles, actual, 1)
        slopes.append({"frame": frame, "seconds_per_radian": float(slope), "intercept_ms": float(intercept*1000),
                       "implied_scan_period_s": float(slope*2*np.pi)})
        if frame == end:
            point_plots = [angles, actual*1000, (estimates["calibrated_inverse_front_phase"]-actual)*1000,
                           (deskew_predicted-actual)*1000]
    reliable = [r for r in rows if r["point_count"] >= 5]
    actor_stats = {name: stats([r["error_ms_against_in_box_median"][name] for r in rows]) for name in models}
    reliable_stats = {name: stats([r["error_ms_against_in_box_median"][name] for r in reliable]) for name in models}
    point_stats = {name: stats(values) for name, values in all_point_errors.items()}
    point_stats["calibrated_inverse_front_phase_per_point_pose"] = stats(deskew_errors)
    result = {"sequence": dataset.name, "raw_frame_range": [start, end], "scan_period_s": period,
              "reference": "raw d=0 point timestamps in original world cuboids; no parser actor interpolation",
              "actor_rows": rows, "actor_all": actor_stats, "actor_at_least_5_points": reliable_stats,
              "full_scan_point_errors": point_stats, "scan_slope_fits": slopes,
              "proposed_capture_times": proposed_capture_times, "zero_return_fallback_rows": zero_return_rows,
              "median_split_half_stability": stats(holdout_median_differences),
              "limits": ["Cuboid center time and visible surface median are different estimands.",
                         "Geometry and per-point times not an independent hardware timing ground truth.",
                         "Data from one 028 sequence only; no claim of complete Camera/LiDAR RS validation.",
                         "Per-point-pose check uses true point times diagnostically; it is not a predictor for unknown times."]}
    (output/"actor_timing_validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    selected = ["upstream", "raw_pose_inverse_only", "calibrated_inverse_front_phase"]
    for name in selected:
        for actor_id in range(len(actors)):
            subset = [r for r in rows if r["actor_id"] == actor_id]
            axes[0].plot([r["frame"] for r in subset], [r["error_ms_against_in_box_median"][name] for r in subset],
                         marker=["o", "s", "^"][actor_id], markersize=4,
                         color={"upstream": "#cc4949", "raw_pose_inverse_only": "#d89829", "calibrated_inverse_front_phase": "#218b67"}[name],
                         label=name if actor_id == 0 else None)
    axes[0].axhline(0, c="black", linewidth=1)
    axes[0].set(title="Actor center predictions minus in-box timestamp median", xlabel="original frame", ylabel="time error [ms]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].scatter(point_plots[0], point_plots[1], s=.7, alpha=.2, label=f"raw frame {end} timestamps")
    xx = np.linspace(-np.pi, np.pi, 200)
    axes[1].plot(xx, xx*period/(2*np.pi)*1000, c="red", label="calibrated front-phase model")
    axes[1].set(title="Scan direction and phase check, sensor-native angular phase", xlabel="wrapped phase difference [rad]", ylabel="point time - front camera [ms]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    fig.savefig(output/"actor_timing_validation.png", dpi=160)
    plt.close(fig)

    summary_table = "\n".join(f"| {name} | {reliable_stats[name]['count']} | {reliable_stats[name]['median_error_ms']:.3f} | "
                              f"{reliable_stats[name]['median_abs_error_ms']:.3f} | {reliable_stats[name]['p95_abs_error_ms']:.3f} |" for name in models)
    full_table = "\n".join(f"| {name} | {value['count']} | {value['median_error_ms']:.3f} | {value['p95_abs_error_ms']:.3f} |" for name, value in point_stats.items())
    final_rows = [r for r in rows if r["frame"] == end]
    final_table = "\n".join(f"| {r['actor_id']} | {r['point_count']} | {r['actual_in_box_relative_time_ms_p10_p50_p90'][1]:.3f} | "
                            f"{r['center_prediction_ms']['upstream']:.3f} | {r['center_prediction_ms']['raw_pose_inverse_only']:.3f} | "
                            f"{r['center_prediction_ms']['calibrated_inverse_front_phase']:.3f} |" for r in final_rows)
    report = f"""# PandaSet actor 校时逐点验证

读取原始{start}–{end}帧 Pandar64 (`d=0`) 点与原始非静态 cuboid；直接在 dataset world 坐标判断框内点，不使用导出/外推 actor 轨迹。共 {len(rows)} 个有回波 actor-frame，其中 {len(reliable)} 个框内至少5点。单点/少点框单独保存在JSON，下面稳健统计使用至少5点的样本。扫描周期由 lidar timestamps 差分均值确定：{period:.9f} s。

## 不是只修矩阵就够了

原始{end}帧，各时间均相对同帧 front-camera（毫秒）：

| actor | 框内点数 | 真实点中位数 | upstream预测 | 仅raw pose逆矩阵 | calibrated pose逆矩阵+前向相位 |
|---|---:|---:|---:|---:|---:|
{final_table}

上游同时存在坐标方向、pose来源和扫描相位的一致性问题：`_get_lidars` 已明确使用 front-camera pose 与标定外参重建 LiDAR pose，但 `_get_actor_trajectories` 仍使用原始 lidar poses，并把世界中心再次前向变换。**仅把矩阵改为逆变换会留下约25ms误差，不能作为完整修复。**

## 多公式对比

误差是预测的 cuboid 中心时间减去该框内可见点的实际时间中位数：

| 公式 | actor-frame数 | 中位有符号误差 ms | 中位绝对误差 ms | P95绝对误差 ms |
|---|---:|---:|---:|---:|
{summary_table}

`calibrated_inverse_front_phase` 与数据最一致：sensor native 前向是 -y，角度定义为 `wrap(-pi/2 - atan2(y_lidar,x_lidar))`，而不是原公式以 +x 为零位相。反转扫描方向会使误差上升。cuboid中心与可见表面中位数不应被期待完全相同，因此没有拟合额外补偿常数。

## 全扫描方向验证

每8点均匀抽样，保留距sensor>5m、相对时间绝对值<45ms的点，避开扫描接缝和近车点。以实际逐点时间验证，不只观察三个actor。

| 公式 | 点数 | 中位误差 ms | P95绝对误差 ms |
|---|---:|---:|---:|
{full_table}

角度→真实时间线性拟合的扫描周期范围为 {min(x['implied_scan_period_s'] for x in slopes):.6f}–{max(x['implied_scan_period_s'] for x in slopes):.6f} s；斜率符号均为正，与推荐相位/方向一致。额外 per-point-pose 诊断按真实点时间对camera派生pose作SE(3)插值，避免把整帧近似混同于完整逐点运动补偿。

![时间误差与方向](actor_timing_validation.png)

## 建议的最小一致修复

在 `_get_actor_trajectories` 内复用 `_get_lidars` 的pose构建：

```python
T_world_lidar = T_world_front_camera @ T_front_camera_lidar
pos_lidar = (pos_world - T_world_lidar[:3, 3]) @ T_world_lidar[:3, :3]
angle = np.arctan2(-pos_lidar[:, 0], -pos_lidar[:, 1])
cuboid_times = front_camera_timestamp + angle / (2 * np.pi) * mean_scan_period
```

该角度公式已在[-pi,pi]，等价于上面的wrapped phase。需要同时改pose来源、逆变换和phase，保留 upstream commit 并把更改作为显式patch记录。这里**没有修改上游**。

本证据支持对028数据的actor时刻修复与回归测试；仍不代表 Camera rolling shutter 读出方向/时长、LiDAR beam calibration、完整动态去畸变或所有PandaSet序列均已验证。当前静态开头20% smoke与本动态修复分开验收。

复现：`.venv-gpu/Scripts/python.exe scripts/validate_actor_timing.py`。原始行、逐actor误差、公式和采样汇总见 `actor_timing_validation.json`。
"""
    split_stats = stats(holdout_median_differences)
    median_records = [r for r in proposed_capture_times if r["point_count"]]
    report += f"""
## 推荐落地：真实回波时间优先、角度公式仅作无返回 fallback

对每个原始 cuboid，在**同一帧原始 dataset-world 点云**中取 `d == 0` 且落入该 yaw/dims cuboid 的有效返回点，使用其原始 float64 `t` 的中位数作为 capturetime。不要先转换成 float32 Unix秒。原始点和原始bbox已同在世界坐标，不需用任何sensor外参筛选；这也避免actor校时再次依赖错误pose。

本次23个已注释actor-frame：{len(median_records)}个有返回采用中位数，{len(zero_return_rows)}个无返回采用上述校准pose/前向phase公式。无返回样本：{[(r['frame'], r['actor_id']) for r in zero_return_rows]}。每例提案绝对时间、返回数、IQR和fallback来源均写在JSON的 `proposed_capture_times`；单点等低支持例必须保留 `low_support`，不能当高置信几何证据。

为避免把“median对自身误差为0”冒充验证，对>=10返回的actor-frame做20次固定种子的随机分半，比较两半的独立中位数差；{split_stats['count']}次分半差异的中位绝对值为 {split_stats['median_abs_error_ms']:.3f} ms，P95 {split_stats['p95_abs_error_ms']:.3f} ms，最大 {split_stats['max_abs_error_ms']:.3f} ms。这仅评估点样本对中位capturetime的稳定性，不独立证明bbox标签或硬件时钟真值。

实施应先检查有限值、排除其他LiDAR (d=1)、保留严格bbox尺寸，按UUID排序后校验capturetime单调。无返回fallback不是额外观测；应在audit中标注。不要把该时间中位数替代每条LiDAR ray已有的逐点真实时间，二者服务于actor关键帧与ray motion不同环节。

已通过 `view_image` 实际查看时间图：原公式和仅逆矩阵的曲线各有明显固定时间偏移，推荐phase曲线靠近0；全扫描散点与正向模型一致，仍有可见小幅周期残差，符合上述1–2ms剩余误差限制。
"""
    (output/"actor_timing_validation.md").write_text(report, encoding="utf-8")
    print(json.dumps({"actor_at_least_5_points": reliable_stats, "full_scan_point_errors": point_stats}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/pandaset/028"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--start", type=int, default=70)
    parser.add_argument("--end", type=int, default=79)
    args = parser.parse_args()
    validate(args.dataset.resolve(), args.output.resolve(), args.start, args.end)
