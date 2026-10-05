"""Derive angular-motion requests from archived request/sensor JSON, without torch.

Only camera.angular_velocity and lidar.angular_velocity are added. Source files
are immutable; existing destinations are refused. Metadata uses rad/s in sensor
local axes: archived camera GL -> public CV via diag(1,-1,-1), LiDAR unchanged.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def velocity(value, name):
    if (not isinstance(value, list) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError(f"{name} must contain three finite numeric rad/s components")
    return value


def derive(requests, sensors):
    for name in ("camera", "lidar"):
        if "angular_velocity" in requests[name]:
            raise ValueError(f"Source {name} already has angular_velocity; refusing to replace it")
    # A wrong sample's finite velocities could silently produce plausible output.
    for actual, expected, name in (
        (requests["camera"]["timestamp"], sensors["camera_request_timestamp"], "camera timestamp"),
        (requests["lidar"]["timestamp"], sensors["lidar_timestamp"], "lidar timestamp"),
        (requests["camera"]["T_world_camera"], sensors["T_world_camera_cv"], "camera pose"),
        (requests["lidar"]["T_world_lidar"], sensors["T_world_lidar"], "lidar pose"),
        (requests["camera"]["K"], sensors["K"], "camera intrinsics"),
        ([requests["camera"]["width"], requests["camera"]["height"]], sensors["render_camera_size"], "camera size"),
    ):
        if actual != expected:
            raise ValueError(f"Request and sensor metadata do not match: {name}")
    camera_gl = velocity(sensors["camera_angular_velocity_local_ignored"], "camera GL angular velocity")
    lidar_native = velocity(sensors["lidar_angular_velocity_local_ignored"], "LiDAR angular velocity")
    result = copy.deepcopy(requests)
    result["camera"]["angular_velocity"] = [camera_gl[0], -camera_gl[1], -camera_gl[2]]
    result["lidar"]["angular_velocity"] = list(lidar_native)
    unchanged = copy.deepcopy(result)
    for sensor in ("camera", "lidar"):
        del unchanged[sensor]["angular_velocity"]
    if unchanged != requests:
        raise AssertionError("Derivation altered fields other than the two additions")
    return result


def generate(args):
    sequence = sorted(args.source_root.glob("sample*/requests.json"))
    if len(sequence) != 20:
        raise ValueError(f"Expected 20 sequence requests, found {len(sequence)}")
    pairs = [("sequence", p, p.with_name("sensors.json"), args.output_root / p.parent.name / "requests.json") for p in sequence]
    pairs.append(("native_1920", args.native_requests, args.native_sensors, args.native_output))
    manifest_path = args.output_root / "manifest.json"
    destinations = [p[3].resolve() for p in pairs]+[manifest_path.resolve()]
    sources = {p.resolve() for pair in pairs for p in pair[1:3]}
    if len(set(destinations)) != len(destinations) or sources.intersection(destinations):
        raise ValueError("Output paths must be distinct from each other and from every source")
    if any(path.exists() for path in destinations):
        raise FileExistsError("A destination already exists; old evidence will not be overwritten")
    prepared, rows = [], []
    for kind, request_path, sensor_path, destination in pairs:
        requests = json.loads(request_path.read_text(encoding="utf-8"))
        sensors = json.loads(sensor_path.read_text(encoding="utf-8"))
        derived = derive(requests, sensors)
        payload = json.dumps(derived, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
        if json.loads(payload) != derived:
            raise AssertionError("JSON roundtrip changed request fields")
        rows.append({"kind": kind, "sample": request_path.parent.name,
                     "source_requests": str(request_path.resolve()), "source_requests_sha256": sha256(request_path),
                     "source_sensors": str(sensor_path.resolve()), "source_sensors_sha256": sha256(sensor_path),
                     "output_requests": str(destination.resolve()),
                     "camera_angular_velocity_source_gl_rad_s": sensors["camera_angular_velocity_local_ignored"],
                     "camera_angular_velocity_cv_rad_s": derived["camera"]["angular_velocity"],
                     "lidar_angular_velocity_native_rad_s": derived["lidar"]["angular_velocity"],
                     "camera_timestamp": derived["camera"]["timestamp"], "lidar_timestamp": derived["lidar"]["timestamp"],
                     "other_fields_exactly_preserved": True})
        prepared.append((destination, payload))
    # All 21 sample pairings and destination collisions have been checked before writing.
    for (destination, payload), row in zip(prepared, rows):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        row["output_requests_sha256"] = sha256(destination)
        if json.loads(destination.read_text(encoding="utf-8")) != json.loads(payload):
            raise AssertionError("Saved output does not match the prepared request")
    for row in rows:
        for field in ("source_requests", "source_sensors"):
            if sha256(row[field]) != row[field+"_sha256"]:
                raise RuntimeError(f"Source changed during generation: {row[field]}")
    summaries = {}
    for group, members in (("sequence", rows[:-1]), ("native_1920", rows[-1:]), ("all", rows)):
        summaries[group] = {}
        for sensor, field in (("camera", "camera_angular_velocity_cv_rad_s"), ("lidar", "lidar_angular_velocity_native_rad_s")):
            summaries[group][sensor] = {
                "max_absolute_component_rad_s": max(abs(x) for row in members for x in row[field]),
                "max_norm_rad_s": max(math.sqrt(sum(x*x for x in row[field])) for row in members)}
    manifest = {"schema_version": 1, "status": "GENERATED_AND_VERIFIED", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "sequence_count": 20, "native_1920_count": 1, "total_request_files": 21,
                "generator": str(Path(__file__).resolve()), "generator_sha256": sha256(__file__),
                "units": "radians per second", "velocity_frame": "sensor-local",
                "camera_conversion": "omega_CV = diag(1,-1,-1) @ omega_GL; GL right/up/back to CV right/down/forward",
                "lidar_conversion": "identity; retain the native calibrated LiDAR local axes",
                "changes": ["camera.angular_velocity added", "lidar.angular_velocity added"],
                "source_files_sha256_unchanged": True, "other_original_request_fields_preserved": True,
                "limits": "Request derivation only; no rendering, angular-motion accuracy validation, or measured-per-ray timing claim.",
                "angular_velocity_summary": summaries, "requests": rows}
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs/validation_baseline_sequence")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/validation_motion_requests")
    parser.add_argument("--native-requests", type=Path, default=ROOT / "outputs/baseline_requests_native.json")
    parser.add_argument("--native-sensors", type=Path, default=ROOT / "outputs/validation_baseline_full_range/sample_000/sensors.json")
    parser.add_argument("--native-output", type=Path, default=ROOT / "outputs/baseline_requests_native_motion.json")
    path, manifest = generate(parser.parse_args())
    print(path.resolve())
    print(json.dumps({"count": manifest["total_request_files"], "angular_velocity_summary": manifest["angular_velocity_summary"]}, indent=2))


if __name__ == "__main__":
    main()
