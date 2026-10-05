"""Selectively download one PandaSet sequence from a remote ZIP64 archive.

Uses curl HTTP Range requests, concurrent resumable disk chunks, zipfile CRC32
verification and per-file SHA256. The 44 GB parent ZIP is never downloaded.
Example:
  python scripts/download_pandaset.py --sequence 028 --mirror
  python scripts/download_pandaset.py --sequence 028 --verify-only
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import time
import zipfile
import zlib

REVISION = "e2e123aea3b3132c67f4b395ec6120f63e190271"
ORIGIN = f"https://huggingface.co/datasets/georghess/pandaset/resolve/{REVISION}/pandaset.zip"
ARCHIVE_SHA256 = "6e2f978fe8e98a8708ca00acae86415096868eccc2effe9826db57514582433e"
CAMERAS = ("front_camera", "front_left_camera", "front_right_camera", "left_camera", "right_camera", "back_camera")


def file_digest(path: Path) -> tuple[str, int]:
    digest, crc = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            crc = zlib.crc32(block, crc)
    return digest.hexdigest(), crc & 0xFFFFFFFF


def download_range(url: str, start: int, end: int, destination: Path) -> tuple[Path, int]:
    """Validate Content-Range and size before atomically publishing a chunk."""
    expected = end - start + 1
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("url") == url and metadata.get("start") == start
                and metadata.get("end") == end and destination.stat().st_size == expected
                and file_digest(destination)[0] == metadata.get("sha256")):
            return destination, metadata["archive_size"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial, headers = destination.with_suffix(".part"), destination.with_suffix(".headers")
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
        "--connect-timeout", "30", "--max-time", "1200", "--range", f"{start}-{end}",
        "--max-filesize", str(max(expected, 1024 * 1024)),
        "--dump-header", str(headers), "--output", str(partial), url,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Range {start}-{end} failed: {result.stderr.strip()}")
    content_ranges = re.findall(r"(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$", headers.read_text(encoding="latin-1"))
    if not content_ranges or tuple(map(int, content_ranges[-1][:2])) != (start, end):
        raise RuntimeError("Server did not honor the exact HTTP byte range; refusing full-archive fallback")
    archive_size = int(content_ranges[-1][2])
    if partial.stat().st_size != expected:
        raise RuntimeError(f"Truncated range {start}-{end}")
    digest, _ = file_digest(partial)
    partial.replace(destination)
    metadata_path.write_text(json.dumps({"url": url, "start": start, "end": end, "archive_size": archive_size, "sha256": digest}), encoding="utf-8")
    headers.unlink(missing_ok=True)  # Redirect headers can contain ephemeral signed URLs.
    return destination, archive_size


class SegmentedArchive(io.RawIOBase):
    """A virtual ZIP containing cached disjoint ranges at original offsets."""

    def __init__(self, size: int, segments: list[tuple[int, int, Path]]):
        self.size, self.position = size, 0
        self.segments = sorted(segments)
        self.starts = [segment[0] for segment in self.segments]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset if whence == io.SEEK_END else -1
        if position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position) if size < 0 else min(size, max(0, self.size - self.position))
        buffers = []
        while remaining:
            index = bisect_right(self.starts, self.position) - 1
            if index < 0 or self.position > self.segments[index][1]:
                raise OSError(f"ZIP attempted to read an undownloaded range at {self.position}")
            start, end, path = self.segments[index]
            length = min(remaining, end - self.position + 1)
            with path.open("rb") as stream:
                stream.seek(self.position - start)
                block = stream.read(length)
            if len(block) != length:
                raise OSError(f"Truncated cached segment: {path}")
            buffers.append(block)
            self.position += length
            remaining -= length
        return b"".join(buffers)


def safe_destination(root: Path, name: str, sequence: str) -> Path:
    if "\\" in name or ":" in name:
        raise ValueError(f"Unsafe ZIP member: {name}")
    parts = PurePosixPath(name).parts
    if len(parts) < 2 or parts[:2] != ("pandaset", sequence) or any(part in ("..", "") for part in parts):
        raise ValueError(f"Unsafe ZIP member: {name}")
    target = root.joinpath(*parts[1:]).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"ZIP member escapes destination: {name}")
    return target


def read_index(url: str, cache: Path):
    _, size = download_range(url, 0, 0, cache / "probe.bin")
    tail_start = max(0, size - 65557)
    tail_path, _ = download_range(url, tail_start, size - 1, cache / "tail.bin")
    tail = tail_path.read_bytes()
    position = tail.rfind(b"PK\x05\x06")
    if position < 0:
        raise zipfile.BadZipFile("ZIP end record missing")
    record = struct.unpack_from("<4s4H2IH", tail, position)
    central_size, central_start = record[5:7]
    if central_start == 0xFFFFFFFF or central_size == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07", 0, position)
        if locator < 0:
            raise zipfile.BadZipFile("ZIP64 locator missing")
        zip64_offset = struct.unpack_from("<4sIQI", tail, locator)[2]
        zip64 = tail[zip64_offset - tail_start:zip64_offset - tail_start + 56]
        if len(zip64) != 56 or zip64[:4] != b"PK\x06\x06":
            raise zipfile.BadZipFile("ZIP64 record not within bounded tail")
        fields = struct.unpack("<4sQ2H2I4Q", zip64)
        central_size, central_start = fields[-2:]
    if not 0 <= central_start < size or central_size > 64 * 1024 * 1024:
        raise zipfile.BadZipFile("Invalid or unexpectedly large ZIP central directory")
    index_path, _ = download_range(url, central_start, size - 1, cache / "directory.zip")
    with zipfile.ZipFile(index_path) as archive:
        entries = archive.infolist()
    # zipfile compensates for the missing prefix in this directory-only file.
    for entry in entries:
        entry.header_offset += central_start
    return size, central_start, index_path, entries


def verify_dataset(root: Path, sequence: str, manifest: dict) -> dict:
    """Check bytes, JPEG decoding, gzip streams and sensor frame metadata.

    This is file integrity and timestamp validation, not a coordinate audit.
    Pickle payloads are intentionally not executed by the downloader.
    """
    from PIL import Image

    jpeg_count = gzip_count = json_count = 0
    for index, record in enumerate(manifest["files"], 1):
        path = safe_destination(root, "pandaset/" + record["path"], sequence)
        digest, crc = file_digest(path)
        if path.stat().st_size != record["size"] or digest != record["sha256"] or f"{crc:08x}" != record["crc32"]:
            raise RuntimeError(f"File integrity failure: {path}")
        if path.suffix.lower() == ".jpg":
            with Image.open(path) as image:
                image.load()
                if image.size[0] <= 0 or image.size[1] <= 0:
                    raise RuntimeError(f"Invalid JPEG dimensions: {path}")
            jpeg_count += 1
        elif path.name.endswith(".gz"):
            with gzip.open(path, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass
            gzip_count += 1
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            json_count += 1
        if index % 100 == 0:
            print(f"verified {index}/{len(manifest['files'])} files", flush=True)
    sensors = {}
    for sensor in (*CAMERAS, "lidar"):
        directory = root / sequence / ("lidar" if sensor == "lidar" else "camera/" + sensor)
        timestamps = json.loads((directory / "timestamps.json").read_text())
        poses = json.loads((directory / "poses.json").read_text())
        frame_files = sorted(directory.glob("*.pkl.gz" if sensor == "lidar" else "*.jpg"))
        if not len(timestamps) == len(poses) == len(frame_files) == 80:
            raise RuntimeError(f"Expected 80 consistent frames/poses/timestamps for {sensor}")
        if any(not isinstance(value, (int, float)) or not float("-inf") < value < float("inf") for value in timestamps):
            raise RuntimeError(f"Nonfinite timestamp: {sensor}")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise RuntimeError(f"Non-increasing timestamps: {sensor}")
        expected_names = [f"{number:02d}" + (".pkl.gz" if sensor == "lidar" else ".jpg") for number in range(80)]
        if [path.name for path in frame_files] != expected_names:
            raise RuntimeError(f"Missing/duplicate frame index: {sensor}")
        sensors[sensor] = {"frames": len(frame_files), "poses": len(poses), "timestamps": len(timestamps), "strictly_increasing": True, "first_timestamp": timestamps[0], "last_timestamp": timestamps[-1]}
    for kind in ("cuboids", "semseg"):
        annotation_files = sorted((root / sequence / "annotations" / kind).glob("*.pkl.gz"))
        if [path.name for path in annotation_files] != [f"{number:02d}.pkl.gz" for number in range(80)]:
            raise RuntimeError(f"Missing/duplicate annotation frame: {kind}")
    return {"status": "PASS", "scope": "file integrity, decodability and sensor timestamps; NOT geometric alignment", "verified_files": len(manifest["files"]), "jpeg_decoded": jpeg_count, "gzip_crc_verified": gzip_count, "json_parsed": json_count, "sensors": sensors, "annotations": {"cuboids": 80, "semseg": 80}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="028")
    parser.add_argument("--destination", type=Path, default=Path("data/pandaset"))
    parser.add_argument("--url", default=ORIGIN)
    parser.add_argument("--mirror", action="store_true", help="Use hf-mirror.com for the same pinned public archive")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{3}", args.sequence) or not 1 <= args.workers <= 16 or not 1 <= args.chunk_mib <= 64:
        parser.error("sequence must contain three digits, workers 1..16, chunk-mib 1..64")
    root = args.destination.resolve()
    manifest_path = root / f"download_manifest_{args.sequence}.json"
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = verify_dataset(root, args.sequence, manifest)
        (root / f"integrity_{args.sequence}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if not shutil.which("curl"):
        parser.error("curl is required for bounded HTTP Range downloads")
    url = args.url.replace("https://huggingface.co/", "https://hf-mirror.com/") if args.mirror else args.url
    cache = root / ".download-cache" / hashlib.sha256(url.encode()).hexdigest()[:16]
    size, central_start, index_path, entries = read_index(url, cache)
    prefix = f"pandaset/{args.sequence}/"
    selected = [entry for entry in entries if entry.filename.startswith(prefix)]
    files = [entry for entry in selected if not entry.is_dir()]
    if not files:
        raise RuntimeError(f"Sequence {args.sequence} not found in the archive")
    for entry in selected:
        safe_destination(root, entry.filename, args.sequence)
        if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
            raise RuntimeError("Symlink or encrypted ZIP member is unsupported")
    start = min(entry.header_offset for entry in selected)
    last = max(selected, key=lambda entry: entry.header_offset)
    # The ZIP local filename/extra fields are each at most 65535 bytes.
    end = min(central_start - 1, last.header_offset + 30 + 2 * 65535 + last.compress_size - 1)
    compressed_size = sum(entry.compress_size for entry in files)
    if end - start + 1 > compressed_size * 1.10 + 1024 * 1024:
        raise RuntimeError("Sequence entries are not contiguous; refusing excessive overdownload")
    summary = {"sequence": args.sequence, "files": len(files), "uncompressed_bytes": sum(entry.file_size for entry in files), "compressed_bytes": compressed_size, "archive_bytes": size, "range_start": start, "range_end": end, "download_bytes": end - start + 1}
    print(json.dumps(summary), flush=True)
    if args.inspect:
        return
    chunk_size = args.chunk_mib * 1024 * 1024
    ranges = [(offset, min(offset + chunk_size - 1, end)) for offset in range(start, end + 1, chunk_size)]
    segments = [(central_start, size - 1, index_path)]
    started, completed_bytes = time.perf_counter(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_range, url, first, last, cache / f"{first}-{last}.bin"): (first, last) for first, last in ranges}
        for completed, future in enumerate(as_completed(futures), 1):
            first, last = futures[future]
            chunk_path, returned_size = future.result()
            if returned_size != size:
                raise RuntimeError("Archive size changed during download")
            segments.append((first, last, chunk_path))
            completed_bytes += last - first + 1
            print(f"downloaded {completed}/{len(ranges)} chunks; {completed_bytes / 1e6:.1f}/{(end-start+1)/1e6:.1f} MB; {completed_bytes / max(time.perf_counter()-started,0.001)/1e6:.2f} MB/s", flush=True)
    exported = []
    with zipfile.ZipFile(SegmentedArchive(size, segments)) as archive:
        for index, entry in enumerate(files, 1):
            destination = safe_destination(root, entry.filename, args.sequence)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                digest, crc = file_digest(destination)
                if destination.stat().st_size != entry.file_size or crc != entry.CRC:
                    raise RuntimeError(f"Existing destination differs from archive; refusing overwrite: {destination}")
            else:
                temporary = destination.with_name(destination.name + ".download-part")
                with archive.open(entry.filename) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                digest, crc = file_digest(temporary)
                if temporary.stat().st_size != entry.file_size or crc != entry.CRC:
                    raise RuntimeError(f"ZIP size/CRC mismatch: {entry.filename}")
                temporary.replace(destination)
            exported.append({"path": destination.relative_to(root).as_posix(), "size": entry.file_size, "crc32": f"{crc:08x}", "sha256": digest})
            if index % 100 == 0:
                print(f"extracted and CRC-verified {index}/{len(files)} files", flush=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "sequence": args.sequence, "source_url": url, "canonical_source_url": ORIGIN, "source_revision": REVISION, "archive_size": size, "expected_full_archive_sha256": ARCHIVE_SHA256, "full_archive_sha256_verified": False, "verification": "each extracted file matched ZIP CRC32; SHA256 recorded; full archive was not downloaded", "summary": summary, "files": exported}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = verify_dataset(root, args.sequence, manifest)
    (root / f"integrity_{args.sequence}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
