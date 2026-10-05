"""Collect a short CUDA operator trace from a real checkpoint and fixed requests.

Trace timings include instrumentation overhead. Use the separate profile command
for throughput measurements, with no training job concurrently using the GPU.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs/torch_cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "outputs/torch_extensions"))

import torch
from splatad_drive import SplatADDrive
from splatad_drive.cli import _load_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/renderer_trace"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations positive")
    drive = SplatADDrive.from_checkpoint(args.config)
    requests = _load_requests(args.requests, "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for _ in range(args.warmup):
            for name, request in requests.items():
                getattr(drive, "render_"+name)(request)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profile:
            for _ in range(args.iterations):
                for name, request in requests.items():
                    with torch.profiler.record_function(name + "_complete_render"):
                        getattr(drive, "render_"+name)(request)
                torch.cuda.synchronize()
                profile.step()
    profile.export_chrome_trace(str(args.output / "trace.json"))
    averages = profile.key_averages()
    (args.output / "operators.txt").write_text(averages.table(sort_by="self_cuda_time_total", row_limit=50), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({
        "checkpoint": drive.backend.checkpoint_path, "step": drive.backend.step,
        "requests": str(args.requests.resolve()), "iterations": args.iterations,
        "warmup": args.warmup, "gpu": torch.cuda.get_device_name(),
        "scope": "Instrumented real CUDA operator trace; not the throughput benchmark",
    }, indent=2, default=str), encoding="utf-8")
    print((args.output / "operators.txt").resolve())


if __name__ == "__main__":
    main()
