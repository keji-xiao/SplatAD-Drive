"""Construct the pinned official trainer command and enforce audit prerequisites."""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_training_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_training_config(config)
    return config


def validate_training_config(config):
    """Reject YAML type ambiguities before launching an expensive real run."""
    required = {"name", "sequence", "iterations", "seed", "cameras", "dataset_end_fraction", "data", "output_dir", "vis"}
    optional = {"checkpoint_interval", "resume_checkpoint"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError(f"training config requires {sorted(required)}; optional keys are {sorted(optional)}")
    if isinstance(config["iterations"], bool) or not isinstance(config["iterations"], int) or config["iterations"] < 1:
        raise ValueError("iterations must be a positive integer")
    fraction = config["dataset_end_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("dataset_end_fraction must be in (0,1]")
    cameras = config["cameras"]
    if not isinstance(cameras, (list, tuple)) or not cameras or any(not isinstance(x, str) or x not in {"front", "front_left", "front_right", "back", "left", "right"} for x in cameras) or len(set(cameras)) != len(cameras):
        raise ValueError("invalid PandaSet cameras")
    if not isinstance(config["sequence"], str) or not re.fullmatch(r"[0-9]{3}", config["sequence"]):
        raise ValueError("sequence must be a quoted three-digit string, e.g. '028'")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(config["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config["name"]):
        raise ValueError("name must be a safe experiment directory name")
    for field in ("data", "output_dir"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a nonempty path string")
    if config["vis"] != "tensorboard":
        raise ValueError("vis must be tensorboard so training evidence is retained")
    interval = config.get("checkpoint_interval", min(config["iterations"], 2000))
    if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 2500:
        raise ValueError("checkpoint_interval must be an integer in [1, 2500]")
    checkpoint = config.get("resume_checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("resume_checkpoint must be null or a nonempty path string")


def build_train_command(config, root=None, python=None, timestamp=None):
    validate_training_config(config)
    root = Path(root or Path.cwd()).resolve()
    # Tyro uses hyphens; trainer options must precede the dataparser subcommand.
    command = [str(python or sys.executable), str(root / "scripts/train_official.py"),
            "splatad", "--max-num-iterations", str(config["iterations"]),
            "--experiment-name", config["name"], "--output-dir", str((root / config["output_dir"]).resolve()),
            "--machine.seed", str(config["seed"]), "--vis", config["vis"],
            "--steps-per-save", str(config.get("checkpoint_interval", min(config["iterations"], 2000)))]
    if config.get("resume_checkpoint") is not None:
        command.extend(["--load-checkpoint", str((root / config["resume_checkpoint"]).resolve())])
    if timestamp is not None:
        command.extend(["--timestamp", timestamp])
    command.extend(["pandaset-data", "--data", str((root / config["data"]).resolve()),
            "--sequence", str(config["sequence"]), "--dataset-end-fraction", str(config["dataset_end_fraction"]),
            "--cameras", *config["cameras"]])
    return command


def validate_review_scope(config, review):
    """A limited smoke review cannot implicitly approve a larger experiment."""
    scope = review.get("training_scope")
    if scope is None:
        return  # Preserve existing review manifests that did not declare scope.
    if not isinstance(scope, dict):
        raise ValueError("alignment review training_scope must be a mapping")
    if "iterations_max" in scope:
        maximum = scope["iterations_max"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("training_scope iterations_max must be a positive integer")
        if config["iterations"] > maximum:
            raise ValueError("training iterations exceed the alignment review training_scope")
    if "cameras" in scope:
        cameras = scope["cameras"]
        if not isinstance(cameras, list) or not cameras or any(not isinstance(camera, str) for camera in cameras):
            raise ValueError("training_scope cameras must be a nonempty list of names")
        if not set(config["cameras"]).issubset(cameras):
            raise ValueError("training cameras exceed the alignment review training_scope")
    if "dataset_end_fraction_max" in scope:
        maximum = scope["dataset_end_fraction_max"]
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or not 0 < maximum <= 1:
            raise ValueError("training_scope dataset_end_fraction_max must be in (0, 1]")
        if config["dataset_end_fraction"] > maximum:
            raise ValueError("training dataset fraction exceeds the alignment review training_scope")


def run_training(config, root=None, audit_path=None, alignment_review=None):
    from .environment import environment_report
    validate_training_config(config)
    root = Path(root or Path.cwd()).resolve()
    evidence = environment_report(root, check_upstream=True)
    if not evidence["official_training_ready"]:
        raise RuntimeError("official environment is not ready: " + "; ".join(evidence["blockers"]))
    if audit_path is None or alignment_review is None:
        raise ValueError("training requires --audit and --alignment-review; inspect generated overlays first")
    audit_path = (root / audit_path).resolve()
    alignment_review = (root / alignment_review).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    review = json.loads(alignment_review.read_text(encoding="utf-8"))
    digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if review.get("audit_sha256") != digest or review.get("approved") is not True or not str(review.get("reviewer", "")).strip():
        raise ValueError("alignment review must identify reviewer, approve=true, and match audit_sha256")
    if str(audit.get("sequence")) != str(config["sequence"]):
        raise ValueError("audit sequence differs from training config")
    if (root / audit.get("data", "")).resolve() != (root / config["data"]).resolve():
        raise ValueError("audit dataset differs from training config")
    if audit.get("numeric_checks_passed") is not True:
        raise ValueError("audit numerical checks have not passed")
    source_hashes = evidence.get("source_file_sha256", {})
    if source_hashes.get("pandaset_dataparser") and audit.get("parser_source_sha256") != source_hashes["pandaset_dataparser"]:
        raise ValueError("Data parser source changed or audit lacks its hash; rerun the real data audit")
    if audit.get("actor_timing_helper_sha256") and audit["actor_timing_helper_sha256"] != source_hashes.get("actor_timing_helper"):
        raise ValueError("Actor timing helper changed; rerun the real data audit")
    validate_review_scope(config, review)
    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint is not None and not (root / resume_checkpoint).is_file():
        raise ValueError("resume_checkpoint does not exist")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    command = build_train_command(config, root, timestamp=timestamp)
    # Matches pinned ExperimentConfig.get_base_dir(); repeated runs stay separate.
    output = root / config["output_dir"] / config["name"] / "splatad" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    launch = {"command": command, "environment": evidence, "config": config, "audit_sha256": digest,
              "audit_path": str(audit_path), "alignment_review": review, "run_directory": str(output),
              "expected_iterations": config["iterations"], "expected_final_step": config["iterations"] - 1,
              "checkpoint_interval": config.get("checkpoint_interval", min(config["iterations"], 2000)),
              "resume_checkpoint": str((root / resume_checkpoint).resolve()) if resume_checkpoint else None,
              "iteration_semantics": "total target: resumed runs execute target minus completed iterations",
              "started_utc": datetime.now(timezone.utc).isoformat()}
    (output / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(f"Training evidence directory: {output}", flush=True)
    completion = {"returncode": None, "status": "interrupted_or_failed"}
    try:
        with (output / "training.log").open("w", encoding="utf-8") as log:
            environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                    print(line.encode(encoding, errors="backslashreplace").decode(encoding), end="", flush=True)
                returncode = process.wait()
        completion.update(returncode=returncode, status="exited_zero" if returncode == 0 else "failed")
        return returncode
    finally:
        completion["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
