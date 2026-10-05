"""Plot recorded joint losses and validation PSNR from one real training run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/training_curves"))
    args = parser.parse_args()
    series = {}
    for path in sorted(args.run.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            series.setdefault(tag, []).extend({"step": event.step, "value": event.value} for event in accumulator.Scalars(tag))
    panels = [("Train Loss Dict/main_loss", "Camera reconstruction loss"),
              ("Train Loss Dict/depth_loss", "LiDAR depth loss"),
              ("Train Loss Dict/intensity_loss", "LiDAR intensity loss"),
              ("Train Loss Dict/ray_drop_loss", "LiDAR ray-drop loss"),
              ("Train Metrics Dict/gaussian_count", "Gaussian count"),
              ("Eval Images Metrics Dict (all images)/psnr", "All held-out images: PSNR (dB)")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (tag, title) in zip(axes.ravel(), panels):
        values = sorted(series.get(tag, []), key=lambda value: value["step"])
        if values:
            x, y = np.array([(item["step"], item["value"]) for item in values]).T
            if not np.isfinite(y).all():
                raise ValueError(f"Nonfinite recorded scalar in {tag}")
            axis.plot(x, y, alpha=.45, linewidth=.8, marker="o" if len(x)<15 else None, markersize=3)
            if len(x) >= 20 and "Loss" in tag:
                window = min(30, max(5, len(x)//20))
                axis.plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode="valid"), linewidth=1.5,
                          label=f"{window}-record mean")
                axis.legend(loc="upper right")
        else:
            axis.text(.5, .5, "No recorded samples yet", ha="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.grid(alpha=.2)
    fig.suptitle(args.run.parent.parent.name + " / recorded training evidence\n"
                 "Training batches vary; only held-out PSNR uses the evaluation split.")
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "curves.png", dpi=150)
    plt.close(fig)
    (args.output / "scalars.json").write_text(json.dumps({"run": str(args.run.resolve()), "series": series}), encoding="utf-8")
    print((args.output / "curves.png").resolve())


if __name__ == "__main__":
    main()
