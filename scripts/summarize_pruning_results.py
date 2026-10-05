"""Recompute a compact acceptance summary from this experiment's saved evidence."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = {}
    def read(relative):
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8"))

    baseline = read("outputs/eval_baseline_official.json")
    candidate = read("outputs/eval_pruned_official.json")
    if any(baseline[k] != candidate[k] for k in ("evaluation_scope", "evaluation_seed", "checkpoint_step", "fid_requested")):
        raise ValueError("Official evaluation scopes differ")
    keys = ("psnr", "ssim", "lpips", "intensity_rmse", "ray_drop_accuracy",
            "depth_median_l2", "depth_mean_rel_l2", "chamfer_distance")
    quality = {}
    for key in keys:
        a, b = baseline["results"][key], candidate["results"][key]
        if not all(math.isfinite(v) for v in (a, b)):
            raise ValueError(f"Nonfinite metric: {key}")
        quality[key] = {"baseline": a, "pruned": b, "delta": b-a}
    bp = read("outputs/profile_baseline_official.json")
    cp = read("outputs/profile_pruned_official.json")
    if bp["request_values"] != cp["request_values"]:
        raise ValueError("Performance requests differ")
    performance = {}
    for sensor in ("camera", "lidar"):
        a, b = bp[sensor], cp[sensor]
        for field in ("device_name", "warmup", "iterations", "timing_method", "pixels_per_frame", "rays_per_frame"):
            if a[field] != b[field]:
                raise ValueError(f"Performance scope differs: {sensor}/{field}")
        performance[sensor] = {"baseline_ms": a["mean_ms"], "pruned_ms": b["mean_ms"],
            "speedup": a["mean_ms"]/b["mean_ms"], "baseline_fps": a["fps"], "pruned_fps": b["fps"],
            "baseline_peak_bytes": a["cuda_peak_memory_allocated_bytes"],
            "pruned_peak_bytes": b["cuda_peak_memory_allocated_bytes"], "scope": a["memory_scope"]}
    outputs = {}
    for name in ("public_outputs", "public_outputs_motion"):
        record = read(f"outputs/pruning_comparison/{name}.json")
        if record["status"] != "PASS":
            raise ValueError(f"Output comparison failed: {name}")
        outputs[name] = record["summary"]
    edits = read("outputs/validation_pruned_motion/validation.json")
    if not edits["numeric_checks_passed"]:
        raise ValueError("Editing numeric validation failed")
    actor_checks = [{"actor_id": r["actor_edit"]["actor_id"],
                     "restore_max_error": r["actor_edit"]["restore_max_error"],
                     "parameters_restored": r["actor_edit"]["parameter_restoration"],
                     "ego_relative_extrinsic_max_error": r["ego_shift"]["relative_extrinsic_max_error"]}
                    for r in edits["samples"]]
    xml = ROOT / "reports/tests_release.xml"
    sources["reports/tests_release.xml"] = hashlib.sha256(xml.read_bytes()).hexdigest()
    cases = ET.parse(xml).getroot().findall(".//testcase")
    test_counts = {"total": len(cases), "skipped": sum(c.find("skipped") is not None for c in cases),
                   "failed": sum(c.find("failure") is not None or c.find("error") is not None for c in cases)}
    test_counts["passed"] = test_counts["total"]-test_counts["skipped"]-test_counts["failed"]
    if test_counts["failed"] or not test_counts["passed"]:
        raise ValueError("Release tests failed or absent")
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Verified saved evidence for one local PandaSet 028 experiment; not deployment or multi-sequence paper reproduction",
        "evaluation_scope": baseline["evaluation_scope"], "quality": quality,
        "all_listed_quality_metrics_exact_equal": all(v["delta"] == 0 for v in quality.values()),
        "performance": performance, "public_outputs": outputs, "actor_edits": actor_checks,
        "tests": test_counts, "source_sha256": sources}
    output = ROOT / "outputs/pruning_comparison/summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
