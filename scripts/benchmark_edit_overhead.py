"""Compare the unedited inference fast path with its saved previous source.

Both variants render the same loaded model and fixed requests. The legacy source
must be a trusted project source snapshot; it is imported as Python code.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs/torch_cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "outputs/torch_extensions"))

import torch
from splatad_drive.backends.neurad import NeuradBackend
from splatad_drive.benchmark import benchmark_renderer
from splatad_drive.cli import _load_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/edit_overhead_benchmark.json"))
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("splatad_drive.backends.neurad_legacy", args.legacy_source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    backend = NeuradBackend.from_config(args.config)
    requests = _load_requests(args.requests, "cuda")
    optimized_context = backend._edit_context
    legacy_context = legacy.NeuradBackend._edit_context.__get__(backend)
    report = {"checkpoint": str(backend.checkpoint_path), "step": backend.step,
              "gaussian_count": len(backend.model.gauss_params["means"]),
              "requests": str(args.requests.resolve()), "legacy_source": str(args.legacy_source.resolve()),
              "legacy_sha256": hashlib.sha256(args.legacy_source.read_bytes()).hexdigest(),
              "optimized_sha256": hashlib.sha256(Path(ROOT / "src/splatad_drive/backends/neurad.py").read_bytes()).hexdigest(),
              "order": ["legacy", "optimized", "optimized", "legacy"], "sensors": {},
              "scope": "Same checkpoint, no scene edits, one fixed request per sensor; ABBA GPU timing. No training or pruning change."}
    try:
        for name, request in requests.items():
            render = lambda: getattr(backend, "render_"+name)(request)
            with torch.inference_mode():
                backend._edit_context = legacy_context
                reference = {key: value.detach().cpu() for key, value in render().items()}
                backend._edit_context = optimized_context
                current = render()
                differences = {key: float((value.detach().cpu()-reference[key]).abs().max()) for key, value in current.items()}
                if max(differences.values()) > 1e-6:
                    raise AssertionError(f"Fast path changed {name} outputs: {differences}")
                del reference, current
            rounds = []
            counts = {"pixels_per_frame": request.width*request.height} if name=="camera" else {"rays_per_frame": len(request.elevation)*len(request.azimuth)}
            for variant in report["order"]:
                backend._edit_context = legacy_context if variant=="legacy" else optimized_context
                rounds.append({"variant": variant, **benchmark_renderer(render, device="cuda", warmup=25,
                                                                        iterations=args.iterations, **counts)})
            times = {variant: statistics.mean(row["mean_ms"] for row in rounds if row["variant"]==variant)
                     for variant in ("legacy", "optimized")}
            report["sensors"][name] = {"output_max_absolute_differences": differences, "rounds": rounds,
                                       "mean_ms": times, "speedup": times["legacy"]/times["optimized"]}
    finally:
        backend._edit_context = optimized_context
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
