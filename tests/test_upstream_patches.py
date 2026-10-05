import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"scripts/apply_upstream_patches.py"
spec = importlib.util.spec_from_file_location("splatad_patch_application", SCRIPT)
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


@pytest.fixture
def fixture_patch(tmp_path):
    upstream = tmp_path/"upstream"
    upstream.mkdir()
    target = upstream/"parser.py"
    original = "def capture():\n    return 'original'\n"
    updated = "def capture():\n    # independently reviewed correction\n    return 'corrected'\n"
    target.write_bytes(original.replace("\n", "\r\n").encode())
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True),
                                         fromfile="a/parser.py", tofile="b/parser.py"))
    (tmp_path/"correction.patch").write_bytes(patch.encode())
    manifest = {"name": "fixture", "base_commit": "explicit-reference", "relative_path": "parser.py",
                "patch_file": "correction.patch", "original_sha256": patcher.normalized_hash(original),
                "patched_sha256": patcher.normalized_hash(updated), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return upstream, target, manifest_path, original, updated


def test_check_apply_and_idempotent_repeat_preserve_newlines(fixture_patch):
    upstream, target, manifest, original, updated = fixture_patch
    old_bytes = target.read_bytes()
    assert patcher.apply_patch(upstream, manifest, check_only=True)["status"] == "ready"
    assert target.read_bytes() == old_bytes
    assert patcher.apply_patch(upstream, manifest)["status"] == "applied"
    assert target.read_bytes() == updated.replace("\n", "\r\n").encode()
    new_bytes = target.read_bytes()
    assert patcher.apply_patch(upstream, manifest)["status"] == "already_applied"
    assert target.read_bytes() == new_bytes


def test_unknown_upstream_version_is_refused_without_changes(fixture_patch):
    upstream, target, manifest, _, _ = fixture_patch
    target.write_text("def capture():\n    return 'unreviewed revision'\n")
    old_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="unrecognized upstream file version"):
        patcher.apply_patch(upstream, manifest)
    assert target.read_bytes() == old_bytes


def test_tampered_patch_is_refused_without_changes(fixture_patch):
    upstream, target, manifest, _, _ = fixture_patch
    patch_file = manifest.parent/"correction.patch"
    patch_file.write_bytes(patch_file.read_bytes()+b"unexpected payload\n")
    old_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="Patch file SHA256"):
        patcher.apply_patch(upstream, manifest)
    assert target.read_bytes() == old_bytes


def test_expected_output_hash_is_verified_before_write(fixture_patch):
    upstream, target, manifest, _, _ = fixture_patch
    data = json.loads(manifest.read_text())
    data["patched_sha256"] = "not-the-reviewed-output"
    manifest.write_text(json.dumps(data))
    old_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="Patch output SHA256"):
        patcher.apply_patch(upstream, manifest)
    assert target.read_bytes() == old_bytes


def test_target_cannot_escape_upstream_directory(fixture_patch):
    upstream, target, manifest, _, _ = fixture_patch
    data = json.loads(manifest.read_text())
    data["relative_path"] = "../outside.py"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes upstream root"):
        patcher.apply_patch(upstream, manifest)
