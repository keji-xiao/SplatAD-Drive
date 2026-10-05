import pytest
import torch

from splatad_drive.benchmark import benchmark_renderer


def test_cpu_benchmark_counts_units_and_no_fabricated_gpu_memory():
    calls = []

    def render():
        calls.append(torch.is_inference_mode_enabled())
        return torch.arange(100, dtype=torch.float32).square()

    report = benchmark_renderer(render, warmup=2, iterations=5, pixels_per_frame=2000, rays_per_frame=400)
    assert calls == [True] * 7
    assert report["timing_method"] == "cpu_perf_counter"
    assert report["mean_ms"] > 0
    assert report["fps"] == pytest.approx(1000 / report["mean_ms"])
    assert report["megapixels_per_second"] == pytest.approx(2000 * report["fps"] / 1e6)
    assert report["megarays_per_second"] == pytest.approx(400 * report["fps"] / 1e6)
    assert report["min_ms"] <= report["median_ms"] <= report["p95_ms"] <= report["max_ms"]
    assert report["cuda_peak_memory_allocated_bytes"] is None
    assert not torch.is_inference_mode_enabled()


@pytest.mark.parametrize("kwargs", [{"iterations": 0}, {"iterations": 1.5}, {"warmup": -1}, {"pixels_per_frame": 0}, {"rays_per_frame": True}, {"device": "mps"}])
def test_invalid_benchmark_parameters(kwargs):
    with pytest.raises(ValueError):
        benchmark_renderer(lambda: None, **kwargs)


def test_callback_error_propagates_and_inference_context_restores():
    def fail():
        raise RuntimeError("render failure")

    with pytest.raises(RuntimeError, match="render failure"):
        benchmark_renderer(fail, warmup=0, iterations=1)
    assert not torch.is_inference_mode_enabled()


@pytest.mark.skipif(torch.cuda.is_available(), reason="only tests an unavailable CUDA installation")
def test_unavailable_cuda_is_explicit():
    with pytest.raises(RuntimeError, match="unavailable"):
        benchmark_renderer(lambda: None, device="cuda", iterations=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA hardware")
def test_cuda_events_and_memory():
    values = torch.randn(512, 512, device="cuda")
    report = benchmark_renderer(lambda: values @ values, device="cuda", warmup=1, iterations=3)
    assert report["timing_method"] == "cuda_events_current_stream"
    assert report["mean_ms"] > 0
    assert report["cuda_peak_memory_allocated_bytes"] >= values.numel() * values.element_size()
