"""Synchronized inference benchmarks with explicit timing/memory semantics."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def _integer(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def benchmark_renderer(
    render: Callable[[], Any],
    *,
    device: str | torch.device = "cpu",
    warmup: int = 10,
    iterations: int = 50,
    pixels_per_frame: int | None = None,
    rays_per_frame: int | None = None,
) -> dict[str, Any]:
    """Measure a caller-specified complete render in torch.inference_mode.

    CPU samples use perf_counter; CUDA samples use events on the current stream
    and synchronize after each call. The callback must enqueue work on that
    stream (or explicitly join any worker streams). CUDA event time excludes
    host-only postprocessing; synchronized wall time is separately reported.
    Output tensors are released between iterations. CUDA memory is process-wide
    PyTorch allocator memory, not total board usage or isolated model memory.
    No GPU performance is inferred from a CPU run.
    """
    _integer(warmup, "warmup", allow_zero=True)
    _integer(iterations, "iterations")
    for name, value in (("pixels_per_frame", pixels_per_frame), ("rays_per_frame", rays_per_frame)):
        if value is not None:
            _integer(value, name)
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("only cpu and cuda timing are supported")
    is_cuda = selected_device.type == "cuda"
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but CUDA is unavailable")
    durations: list[float] = []
    wall_durations: list[float] = []
    memory_before: int | None = None
    memory_peak: int | None = None
    memory_reserved: int | None = None

    def run() -> None:
        nonlocal memory_before, memory_peak, memory_reserved
        with torch.inference_mode():
            for _ in range(warmup):
                output = render()
                if is_cuda:
                    torch.cuda.synchronize(selected_device)
                del output
            if is_cuda:
                torch.cuda.synchronize(selected_device)
                memory_before = torch.cuda.memory_allocated(selected_device)
                torch.cuda.reset_peak_memory_stats(selected_device)
            for _ in range(iterations):
                if is_cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    start.record()
                    output = render()
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end) / 1000
                else:
                    wall_start = time.perf_counter()
                    output = render()
                    elapsed = time.perf_counter() - wall_start
                wall_durations.append(time.perf_counter() - wall_start)
                durations.append(elapsed)
                del output
            if is_cuda:
                memory_peak = torch.cuda.max_memory_allocated(selected_device)
                memory_reserved = torch.cuda.max_memory_reserved(selected_device)

    if is_cuda:
        with torch.cuda.device(selected_device):
            run()
    else:
        run()
    mean_seconds = statistics.mean(durations)
    if mean_seconds <= 0:
        raise RuntimeError("timing resolution was insufficient; increase renderer workload")
    sorted_ms = sorted(value * 1000 for value in durations)
    # Linear interpolation matches the common percentile definition.
    percentile_position = 0.95 * (len(sorted_ms) - 1)
    lower_index = int(percentile_position)
    upper_index = min(lower_index + 1, len(sorted_ms) - 1)
    p95 = sorted_ms[lower_index] + (sorted_ms[upper_index] - sorted_ms[lower_index]) * (percentile_position - lower_index)
    return {
        "device": str(selected_device),
        "device_name": torch.cuda.get_device_name(selected_device) if is_cuda else "CPU",
        "timing_method": "cuda_events_current_stream" if is_cuda else "cpu_perf_counter",
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": mean_seconds * 1000,
        "median_ms": statistics.median(sorted_ms),
        "p95_ms": p95,
        "min_ms": sorted_ms[0],
        "max_ms": sorted_ms[-1],
        "synchronized_wall_mean_ms": statistics.mean(wall_durations) * 1000,
        "fps": 1 / mean_seconds,
        "pixels_per_frame": pixels_per_frame,
        "rays_per_frame": rays_per_frame,
        "megapixels_per_second": pixels_per_frame / mean_seconds / 1e6 if pixels_per_frame is not None else None,
        "rays_per_second": rays_per_frame / mean_seconds if rays_per_frame is not None else None,
        "megarays_per_second": rays_per_frame / mean_seconds / 1e6 if rays_per_frame is not None else None,
        "cuda_memory_allocated_before_bytes": memory_before,
        "cuda_peak_memory_allocated_bytes": memory_peak,
        "cuda_peak_memory_reserved_bytes": memory_reserved,
        "cuda_incremental_peak_allocated_bytes": memory_peak - memory_before if is_cuda else None,
        "memory_scope": "process-wide PyTorch CUDA allocator; excludes non-PyTorch allocations" if is_cuda else "CUDA memory not applicable; CPU memory not measured",
    }
