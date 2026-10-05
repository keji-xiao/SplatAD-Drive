"""Offline checks for the selective ZIP reader and extraction path boundary."""
import importlib.util
import io
from pathlib import Path
import zipfile

import pytest


_SPEC = importlib.util.spec_from_file_location("download_pandaset", Path(__file__).parents[1] / "scripts" / "download_pandaset.py")
downloader = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(downloader)


def archive_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("pandaset/028/lidar/00.pkl.gz", b"example compressed payload" * 100)
        archive.writestr("pandaset/029/lidar/00.pkl.gz", b"another sequence")
    return stream.getvalue()


def test_segmented_archive_reads_cross_chunk_boundaries(tmp_path):
    data = archive_bytes()
    segments = []
    for start in range(0, len(data), 17):
        end = min(start + 17, len(data))
        path = tmp_path / f"{start}.bin"
        path.write_bytes(data[start:end])
        segments.append((start, end - 1, path))
    with zipfile.ZipFile(downloader.SegmentedArchive(len(data), segments)) as archive:
        assert archive.read("pandaset/028/lidar/00.pkl.gz") == b"example compressed payload" * 100


def test_segmented_archive_rejects_missing_ranges(tmp_path):
    path = tmp_path / "tail.bin"
    path.write_bytes(b"tail")
    stream = downloader.SegmentedArchive(20, [(16, 19, path)])
    with pytest.raises(OSError, match="undownloaded"):
        stream.read(1)
    stream.seek(-4, io.SEEK_END)
    assert stream.read() == b"tail"


def test_remote_index_offset_reconstruction_offline(tmp_path, monkeypatch):
    data = archive_bytes()

    def fake_download(url, start, end, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data[start:end + 1])
        return destination, len(data)

    monkeypatch.setattr(downloader, "download_range", fake_download)
    size, central_start, _, entries = downloader.read_index("https://example.test/data.zip", tmp_path)
    assert size == len(data)
    assert central_start > 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert [entry.header_offset for entry in entries] == [entry.header_offset for entry in archive.infolist()]


@pytest.mark.parametrize("name", ["pandaset/028/../../outside", "/pandaset/028/test", "pandaset/028/a:b", "pandaset/028/..\\outside", "pandaset/029/lidar/a"])
def test_zip_paths_cannot_escape_sequence(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        downloader.safe_destination(tmp_path, name, "028")


def test_expected_dataset_path(tmp_path):
    assert downloader.safe_destination(tmp_path, "pandaset/028/camera/front_camera/00.jpg", "028") == (tmp_path / "028/camera/front_camera/00.jpg").resolve()
