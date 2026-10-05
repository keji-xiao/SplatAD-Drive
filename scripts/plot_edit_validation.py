"""Plot native-resolution actor edit regions from saved inference arrays."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def region(mask, padding):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("No measurable actor edit; cannot manufacture a visual validation region")
    return (max(0, int(y.min())-padding), min(mask.shape[0], int(y.max())+padding+1),
            max(0, int(x.min())-padding), min(mask.shape[1], int(x.max())+padding+1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    args = parser.parse_args()
    names = [name for name in ("baseline", "actor_removed", "actor_translated", "actor_rotated")
             if (args.sample / name / "camera.npz").exists()]
    cameras, lidars = [], []
    for name in names:
        with np.load(args.sample / name / "camera.npz", allow_pickle=False) as data:
            cameras.append(data["rgb"])
        with np.load(args.sample / name / "lidar.npz", allow_pickle=False) as data:
            lidars.append(data["range"])
    if len(names) < 2:
        raise ValueError("At least one completed actor edit is required")
    camera_change = np.max([np.max(np.abs(value-cameras[0]), axis=-1) for value in cameras[1:]], axis=0)
    lidar_change = np.max([np.abs(value-lidars[0]) for value in lidars[1:]], axis=0)
    camera_roi = region(camera_change > 1/255, 16)
    lidar_roi = region(lidar_change > .05, 3)
    cy0, cy1, cx0, cx1 = camera_roi
    ly0, ly1, lx0, lx1 = lidar_roi
    fig, axes = plt.subplots(2, len(names), figsize=(4*len(names), 6.6), constrained_layout=True)
    maximum = float(np.quantile(lidars[0][ly0:ly1, lx0:lx1], .98))
    for column, name in enumerate(names):
        axes[0, column].imshow(cameras[column][cy0:cy1, cx0:cx1], interpolation="nearest")
        axes[0, column].set_title(name.replace("_", " "))
        axes[0, column].set_axis_off()
        mappable = axes[1, column].imshow(lidars[column][ly0:ly1, lx0:lx1], aspect="auto",
            interpolation="nearest", cmap="viridis", vmin=0, vmax=maximum,
            extent=(lx0, lx1, ly1, ly0))
        axes[1, column].set_xlabel("Azimuth column")
        axes[1, column].set_ylabel("Elevation row")
    fig.colorbar(mappable, ax=axes[1, :].tolist(), label="Expected range (m)", shrink=.8)
    fig.suptitle("Actor edit diagnostics: RGB detail and LiDAR range on the same edited scene\n"
                 f"RGB crop x=[{cx0},{cx1}), y=[{cy0},{cy1}); regions selected from measured differences")
    path = args.sample / "actor_edit_diagnostics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    (args.sample / "actor_edit_diagnostics.json").write_text(json.dumps({
        "camera_roi_y0_y1_x0_x1": camera_roi, "lidar_roi_y0_y1_x0_x1": lidar_roi,
        "region_thresholds": {"rgb": 1/255, "range_metres": .05}, "views": names,
        "note": "Single sample mechanics diagnostic; no claim of converged actor reconstruction or complete motion compensation",
    }, indent=2), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
