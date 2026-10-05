"""Environment evidence; importing gsplat is not proof its CUDA kernels work."""
import importlib
import importlib.util
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def environment_report(root=None, check_upstream=False):
    import torch
    root = Path(root or Path.cwd()).resolve()
    result = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "modules": {}, "sources": {}, "source_file_sha256": {},
        "kernel_execution": "NOT_TESTED: run pytest -m cuda",
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name()
        result["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    for name in ("numpy", "PIL", "yaml", "pytest", "gsplat", "nerfstudio", "tinycudann"):
        spec = importlib.util.find_spec(name)
        result["modules"][name] = None if spec is None else spec.origin
    blockers = []
    tracked_files = {
        "pandaset_dataparser": "third_party/neurad-studio/nerfstudio/data/dataparsers/pandaset_dataparser.py",
        "splatad_model": "third_party/neurad-studio/nerfstudio/models/splatad.py",
        "gsplat_rendering": "third_party/splatad/gsplat/rendering.py",
        "actor_timing_helper": "src/splatad_drive/data/actor_timing.py",
        "official_trainer": "src/splatad_drive/official_trainer.py",
        "trainer_checkpoint": "src/splatad_drive/trainer_checkpoint.py",
        "training_wrapper": "src/splatad_drive/training.py",
    }
    for key, relative in tracked_files.items():
        path = root / relative
        if path.is_file():
            result["source_file_sha256"][key] = hashlib.sha256(path.read_bytes()).hexdigest()
    patch_path = root / "patches/pandaset-actor-time.patch"
    if patch_path.is_file():
        result["actor_timing_patch_sha256"] = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    patch_manifest = root / "patches/pandaset-actor-time.json"
    if patch_manifest.is_file():
        manifest = json.loads(patch_manifest.read_text(encoding="utf-8"))
        source = root / tracked_files["pandaset_dataparser"]
        actual = hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest() if source.is_file() else None
        result["actor_timing_patch_applied"] = actual == manifest["patched_sha256"]
        if not result["actor_timing_patch_applied"]:
            blockers.append("actor timing patch is absent or modified; run scripts/apply_upstream_patches.py")
    pins_path = root / "third_party" / "versions.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8")) if pins_path.is_file() else {}
    for name in ("neurad-studio", "splatad"):
        folder = root / "third_party" / name
        try:
            commit = subprocess.check_output(["git", "-c", f"safe.directory={folder.as_posix()}", "-C", str(folder), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
            result["sources"][name] = commit
            expected = pins.get(name, {}).get("commit")
            if expected and commit != expected:
                blockers.append(f"source revision mismatch: {name}: expected {expected}, got {commit}")
        except (OSError, subprocess.CalledProcessError):
            result["sources"][name] = None
            blockers.append(f"missing source: {name}")
    if not torch.cuda.is_available():
        blockers.append("this Python has no usable CUDA PyTorch")
    if check_upstream:
        for name in ("tinycudann", "nerfstudio.models.splatad", "gsplat.rendering"):
            try:
                module = importlib.import_module(name)
                if name == "gsplat.rendering":
                    if not callable(getattr(module, "lidar_rasterization", None)):
                        raise ImportError("installed gsplat is not the SplatAD fork")
                result["modules"][name] = getattr(module, "__file__", None)
            except Exception as exc:
                blockers.append(f"{name}: {type(exc).__name__}: {exc}")
    else:
        for name in ("tinycudann", "nerfstudio", "gsplat"):
            if result["modules"][name] is None:
                blockers.append(f"missing module: {name}")
    result["blockers"] = blockers
    result["official_training_ready"] = not blockers and check_upstream
    return result


def freeze_environment(destination):
    """Capture actual installed versions, never call a speculative list a lock."""
    Path(destination).write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
