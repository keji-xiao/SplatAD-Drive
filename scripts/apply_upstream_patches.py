"""Apply audited upstream patches, refusing unknown source versions.

The file SHA256 is verified before/after, including on repeat invocations.
No Git configuration or other upstream files are changed. This also works
for an archive checkout without .git; the manifest records its base commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def normalized_hash(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def apply_unified_diff(source: str, patch: str, relative_path: str) -> str:
    """Strict single-file unified-diff application; no fuzzy matching."""
    lines = patch.splitlines(keepends=True)
    if len(lines) < 3 or lines[0].rstrip() != "--- a/"+relative_path or lines[1].rstrip() != "+++ b/"+relative_path:
        raise ValueError("Unexpected patch target/header")
    original = source.splitlines(keepends=True)
    output, cursor, index = [], 0, 2
    while index < len(lines):
        header = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n?", lines[index])
        if header is None:
            raise ValueError(f"Invalid hunk header: {lines[index]!r}")
        old_start, old_count, new_start, new_count = (int(header[1]), int(header[2] or 1), int(header[3]), int(header[4] or 1))
        hunk_start = old_start-1 if old_count else old_start
        if hunk_start < cursor:
            raise ValueError("Overlapping or out-of-order patch hunk")
        output.extend(original[cursor:hunk_start])
        cursor = hunk_start
        if len(output) != (new_start-1 if new_count else new_start):
            raise ValueError("Patch new-line offset mismatch")
        consumed = emitted = 0
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if line.startswith((" ", "-")):
                if cursor >= len(original) or original[cursor] != line[1:]:
                    raise ValueError(f"Original/context code mismatch at source line {cursor+1}")
                cursor += 1
                consumed += 1
            if line.startswith((" ", "+")):
                output.append(line[1:])
                emitted += 1
            if not line.startswith((" ", "+", "-")):
                raise ValueError("Unsupported patch content")
            index += 1
        if consumed != old_count or emitted != new_count:
            raise ValueError("Patch hunk line-count mismatch")
    output.extend(original[cursor:])
    return "".join(output)


def apply_patch(upstream_root: Path, manifest_path: Path, *, check_only=False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = upstream_root.resolve()
    target = (root/manifest["relative_path"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Patch target escapes upstream root")
    patch_path = (manifest_path.parent/manifest["patch_file"]).resolve()
    if not patch_path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("Patch filename escapes manifest directory")
    patch_bytes = patch_path.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Patch file SHA256 does not match the reviewed manifest")
    source_bytes = target.read_bytes()
    source = source_bytes.decode("utf-8").replace("\r\n", "\n")
    source_hash = normalized_hash(source)
    result = {"patch": manifest["name"], "base_commit": manifest["base_commit"], "target": str(target),
              "patch_sha256": manifest["patch_sha256"], "source_sha256": source_hash,
              "patched_sha256": manifest["patched_sha256"]}
    if source_hash == manifest["patched_sha256"]:
        return {**result, "status": "already_applied"}
    if source_hash != manifest["original_sha256"]:
        raise ValueError(f"Refusing unrecognized upstream file version: {source_hash}; expected original or patched SHA256")
    updated = apply_unified_diff(source, patch_bytes.decode("utf-8"), manifest["relative_path"])
    if normalized_hash(updated) != manifest["patched_sha256"]:
        raise ValueError("Patch output SHA256 does not match the reviewed manifest")
    if not check_only:
        newline = "\r\n" if b"\r\n" in source_bytes else "\n"
        target.write_bytes(updated.replace("\n", newline).encode("utf-8"))
    return {**result, "status": "ready" if check_only else "applied"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=ROOT/"third_party/neurad-studio")
    parser.add_argument("--manifest", type=Path, default=ROOT/"patches/pandaset-actor-time.json")
    parser.add_argument("--check", action="store_true", help="Validate without changing files")
    args = parser.parse_args()
    try:
        result = apply_patch(args.upstream_root, args.manifest, check_only=args.check)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Patch refused: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
