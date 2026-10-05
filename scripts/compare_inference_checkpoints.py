"""Compare public camera/LiDAR tensors from two trusted official checkpoints.

Only one model is loaded at a time. Reference outputs stay on CPU while the
reference model is released before candidate loading. All sample*/requests.json
files are tested unchanged; the first/last also get explicitly zero RS. This
is output-equivalence validation, not a timing benchmark or a proof for unseen
requests. Camera median metadata and rasterizer internals are not public outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs/torch_cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "outputs/torch_extensions"))

import torch

from splatad_drive.backends.neurad import NeuradBackend
from splatad_drive.cli import _load_requests
from splatad_drive.io import save_json


ABSOLUTE_BANDS = (1e-6, 1e-5, 1e-4, 1e-3)
DISTANCE_KEYS = {"camera": {"depth"}, "lidar": {"range", "median_range", "points", "depth_sum"}}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(value):
    finite = torch.isfinite(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel(),
            "finite_count": int(finite.sum()), "nonfinite_count": int((~finite).sum()),
            "all_finite": bool(finite.all())}


def compare_tensor(reference, candidate, *, atol=1e-5, rtol=1e-5, distance=False, points=False):
    """CPU comparison; reference magnitude defines the relative tolerance.

    Every nonfinite value fails, including matching NaN/Inf. Absolute-error
    metrics exclude nonfinite pairs (counts are explicit), never hide them as
    zero error. Additional absolute-only bands do not override the pass rule.
    """
    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and nonnegative")
    reference, candidate = reference.detach().cpu(), candidate.detach().cpu()
    result = {"reference": tensor_summary(reference), "candidate": tensor_summary(candidate),
              "shape_equal": reference.shape == candidate.shape,
              "dtype_equal": reference.dtype == candidate.dtype, "status": "FAIL"}
    if not result["shape_equal"]:
        result["reason"] = "shape mismatch; elementwise metrics are undefined"
        return result
    if reference.is_complex() or candidate.is_complex():
        result["reason"] = "complex tensors are outside this public output contract"
        return result
    left, right = reference.to(torch.float64), candidate.to(torch.float64)
    finite = torch.isfinite(left) & torch.isfinite(right)
    error = (right-left).abs()
    finite_error = error[finite]
    allowed = atol+rtol*left.abs()
    close = finite & (error <= allowed)
    exact = reference == candidate
    result.update(
        finite_pair_count=int(finite.sum()), nonfinite_pair_count=int((~finite).sum()),
        unequal_count=int((~exact).sum()), exact_equal=bool(exact.all()),
        within_tolerance_count=int(close.sum()), outside_tolerance_count=int((~close).sum()),
        max_absolute_error=float(finite_error.max()) if finite_error.numel() else None,
        mean_absolute_error=float(finite_error.mean()) if finite_error.numel() else None,
        absolute_only_bands={str(bound): {"within_count": int((finite & (error <= bound)).sum()),
                                        "outside_count": int((~(finite & (error <= bound))).sum())}
                             for bound in ABSOLUTE_BANDS})
    if bool(close.all()) and result["dtype_equal"]:
        result["status"] = "PASS"
    if distance:
        result["distance_error_metres"] = {
            "max": result["max_absolute_error"], "mean": result["mean_absolute_error"],
            "finite_pairs_over_1mm": int((finite_error > .001).sum()),
            "finite_pairs_over_1cm": int((finite_error > .01).sum()),
            "finite_pairs_over_1m": int((finite_error > 1.).sum()),
            "note": "points uses component errors here; depth_sum is alpha-weighted range"}
    if points and reference.ndim and reference.shape[-1] == 3:
        point_finite = finite.all(dim=-1)
        l2 = (right-left).norm(dim=-1)[point_finite]
        result["point_l2_error_metres"] = {
            "finite_point_count": int(point_finite.sum()),
            "nonfinite_point_count": int((~point_finite).sum()),
            "max": float(l2.max()) if l2.numel() else None,
            "mean": float(l2.mean()) if l2.numel() else None,
            "finite_points_over_1cm": int((l2 > .01).sum()),
            "finite_points_over_1m": int((l2 > 1.).sum())}
    return result


def compare_outputs(reference, candidate, *, sensor, atol, rtol):
    missing, extra = sorted(reference.keys()-candidate.keys()), sorted(candidate.keys()-reference.keys())
    rows = {}
    for key in sorted(reference.keys() & candidate.keys()):
        rows[key] = compare_tensor(reference[key], candidate[key], atol=atol, rtol=rtol,
                                  distance=key in DISTANCE_KEYS.get(sensor, set()), points=key == "points")
    passed = bool(rows) and not missing and not extra and all(row["status"] == "PASS" for row in rows.values())
    return {"status": "PASS" if passed else "FAIL", "missing_candidate_keys": missing,
            "extra_candidate_keys": extra, "outputs": rows}


def discover_cases(root, expected_native_count=20):
    paths = sorted(Path(root).glob("sample*/requests.json"))
    if not paths or len(paths) != expected_native_count:
        raise ValueError(f"Expected {expected_native_count} native requests, found {len(paths)} in {root}")
    cases = [{"id": path.parent.name+"__native_rs", "path": path, "zero_rs": False} for path in paths]
    for path in dict.fromkeys((paths[0], paths[-1])):
        cases.append({"id": path.parent.name+"__zero_rs", "path": path, "zero_rs": True})
    return cases


def case_requests(case, device):
    requests = _load_requests(case["path"], device)
    if requests.keys() != {"camera", "lidar"}:
        raise ValueError("Each case must contain both camera and lidar requests")
    if case["zero_rs"]:
        requests["camera"] = replace(requests["camera"], rolling_shutter_duration=0.)
        requests["lidar"] = replace(requests["lidar"], scan_duration=0., time_offsets=None)
    return requests


def request_summary(requests):
    camera, lidar = requests["camera"], requests["lidar"]
    times = lidar.time_offsets
    return {"camera_timestamp": camera.timestamp, "lidar_timestamp": lidar.timestamp,
            "camera_minus_lidar_seconds": camera.timestamp-lidar.timestamp,
            "camera_rs_seconds": camera.rolling_shutter_duration,
            "lidar_scan_seconds": lidar.scan_duration,
            "lidar_explicit_time_range_seconds": None if times is None else [float(times.min()), float(times.max())],
            "camera_width_height": [camera.width, camera.height],
            "lidar_elevation_azimuth_count": [len(lidar.elevation), len(lidar.azimuth)]}


def render_to_cpu(backend, request, sensor):
    # Function scope ensures the GPU result dies before the next render/model.
    with torch.inference_mode():
        outputs = getattr(backend, "render_"+sensor)(request)
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError("Renderer did not return a nonempty public output mapping")
        if any(not isinstance(value, torch.Tensor) for value in outputs.values()):
            raise TypeError("Every public output must be a tensor; refusing to silently omit an output")
        return {key: value.detach().cpu().clone() for key, value in outputs.items()}


def save_report(path, report):
    # Flush after every sensor, including errors, so a later failure keeps evidence.
    save_json(path, report)


def _error():
    # Store text, never traceback objects retaining model/GPU references.
    return traceback.format_exc()


def run_comparison(args, backend_factory=None):
    """Run both phases; injectable factory supports CPU-only harness tests."""
    protected = {getattr(args, phase+suffix).resolve()
                 for phase in ("reference", "candidate") for suffix in ("_config", "_checkpoint")}
    protected.update(path.resolve() for path in args.requests_root.glob("sample*/requests.json"))
    if args.output.resolve() in protected:
        # This must precede the report-writing finally block.
        raise ValueError("Report output must not overwrite any config, checkpoint, or request input")
    factory = backend_factory or NeuradBackend.from_config
    report = {"schema_version": 1, "status": "RUNNING", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Public inference tensors, unedited scene; same requests; one model loaded at a time. Not a speed benchmark or a guarantee for untested views.",
              "tolerance": {"atol": args.atol, "rtol": args.rtol,
                            "rule": "abs(candidate-reference) <= atol + rtol*abs(reference); same shape/dtype; all finite",
                            "near_identical_bands": list(ABSOLUTE_BANDS), "bands_override_pass": False},
              "requests_root": str(args.requests_root.resolve()), "models": {}, "cases": [], "errors": []}
    references = {}
    try:
        if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
            raise ValueError("atol/rtol must be finite and nonnegative")
        cases = discover_cases(args.requests_root, args.expected_native_count)
        report["native_cases"] = sum(not c["zero_rs"] for c in cases)
        report["zero_rs_cases"] = sum(c["zero_rs"] for c in cases)
        report["expected_sensor_comparisons"] = len(cases)*2
        for case in cases:
            # Validate both requests on CPU before any model is loaded.
            requests = case_requests(case, "cpu")
            report["cases"].append({"id": case["id"], "requests": str(case["path"].resolve()),
                                    "requests_sha256": sha256_file(case["path"]), "zero_rs": case["zero_rs"],
                                    "request_summary": request_summary(requests), "reference": {}, "candidate": {}})
        del requests
        for phase in ("reference", "candidate"):
            config = getattr(args, phase+"_config").resolve()
            checkpoint = getattr(args, phase+"_checkpoint").resolve()
            if not config.is_file() or not checkpoint.is_file():
                raise FileNotFoundError(f"Missing {phase} config/checkpoint: {config}, {checkpoint}")
            report["models"][phase] = {"config": str(config), "config_sha256": sha256_file(config),
                                         "expected_checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                                         "checkpoint_bytes": checkpoint.stat().st_size, "status": "PENDING"}
        save_report(args.output, report)
        for phase in ("reference", "candidate"):
            model_info = report["models"][phase]
            backend = None
            try:
                print(f"Loading {phase}: {model_info['expected_checkpoint']}", flush=True)
                backend = factory(model_info["config"], load_dir=Path(model_info["expected_checkpoint"]).parent)
                actual = Path(backend.checkpoint_path).resolve()
                model_info.update(actual_checkpoint=str(actual), step=backend.step,
                                  gaussian_count=len(backend.model.gauss_params["means"]), device=str(backend.device))
                if actual != Path(model_info["expected_checkpoint"]):
                    raise ValueError(f"Loaded {actual}, expected {model_info['expected_checkpoint']}; from_config selects the configured/latest step")
                model_info["checkpoint_path_verified"] = True
                model_info["status"] = "RENDERING"
                for case, record in zip(cases, report["cases"]):
                    requests = None
                    try:
                        if sha256_file(case["path"]) != record["requests_sha256"]:
                            raise ValueError("Request file changed during comparison")
                        requests = case_requests(case, backend.device)
                        for sensor in ("camera", "lidar"):
                            values = None
                            try:
                                values = render_to_cpu(backend, requests[sensor], sensor)
                                summaries = {key: tensor_summary(value) for key, value in values.items()}
                                if phase == "reference":
                                    references[case["id"], sensor] = values
                                    record[phase][sensor] = {"status": "PASS" if all(v["all_finite"] for v in summaries.values()) else "FAIL",
                                                             "outputs": summaries}
                                else:
                                    before = references.get((case["id"], sensor))
                                    if before is None:
                                        record[phase][sensor] = {"status": "ERROR", "reason": "Reference render unavailable", "outputs": summaries}
                                    else:
                                        record[phase][sensor] = compare_outputs(before, values, sensor=sensor, atol=args.atol, rtol=args.rtol)
                            except (Exception, SystemExit):
                                record[phase][sensor] = {"status": "ERROR", "traceback": _error()}
                            finally:
                                values = None
                                save_report(args.output, report)
                            print(f"{phase} {case['id']} {sensor}: {record[phase][sensor]['status']}", flush=True)
                    except (Exception, SystemExit):
                        record[phase]["request_error"] = _error()
                        save_report(args.output, report)
                    finally:
                        requests = None
                model_info["status"] = "RENDERED"
            except (Exception, SystemExit):
                model_info["status"] = "ERROR"
                model_info["traceback"] = _error()
            finally:
                # No renderer closures or GPU outputs are retained in references.
                backend = None
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    model_info["cuda_allocated_bytes_after_release"] = torch.cuda.memory_allocated()
                    model_info["cuda_reserved_bytes_after_release"] = torch.cuda.memory_reserved()
                model_info["released_before_next_model"] = True
                save_report(args.output, report)
        rows = [record["candidate"].get(sensor, {"status": "ERROR"})
                for record in report["cases"] for sensor in ("camera", "lidar")]
        report["summary"] = {"sensor_comparisons": len(rows),
                              "passed": sum(row["status"] == "PASS" for row in rows),
                              "failed": sum(row["status"] == "FAIL" for row in rows),
                              "errors": sum(row["status"] == "ERROR" for row in rows),
                              "all_compared_tensors_exact_equal": bool(rows) and all(
                                  row["status"] == "PASS" and all(v.get("exact_equal", False) for v in row["outputs"].values()) for row in rows)}
        passed = bool(rows) and all(row["status"] == "PASS" for row in rows)
        passed = passed and all(info["status"] == "RENDERED" for info in report["models"].values())
        report["status"] = "PASS" if passed else "FAIL"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["errors"].append(_error())
    except (Exception, SystemExit):
        report["status"] = "ERROR"
        report["errors"].append(_error())
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_report(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in ("reference", "candidate"):
        parser.add_argument("--"+phase+"-config", type=Path, required=True)
        parser.add_argument("--"+phase+"-checkpoint", type=Path, required=True,
                            help="Exact expected checkpoint. Loading a different/latest step is a reported error.")
    parser.add_argument("--requests-root", type=Path, default=ROOT / "outputs/validation_baseline_sequence")
    parser.add_argument("--expected-native-count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pruning_comparison/public_outputs.json")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_comparison(args)
    print(f"{report['status']}: {args.output.resolve()}", flush=True)
    return 0 if report["status"] == "PASS" else (130 if report["status"] == "INTERRUPTED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
