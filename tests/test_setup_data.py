"""Offline checks for verified, non-destructive installation of the shared sample."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from zipfile import ZipFile

import pytest

from scripts import setup_data


def csv_text(columns, count):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(sorted(columns))
    writer.writerows([[str(i)] * len(columns) for i in range(count)])
    return stream.getvalue()


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    folder = tmp_path / "downloads"
    folder.mkdir()
    for name, members in setup_data.MEMBERS.items():
        with ZipFile(folder / name, "w") as archive:
            for member in members:
                columns = setup_data.MARKET_COLUMNS if member.endswith("markets.csv") else setup_data.TRADE_COLUMNS
                archive.writestr(member, csv_text(columns, 3 if member.startswith("master/") else 1))
            archive.writestr("__MACOSX/._metadata", "ignored")
            archive.writestr("../escape.txt", "must never be extracted")
    manifest = {
        "window_start_inclusive": "2024-11-25T17:00:00-05:00",
        "window_end_inclusive": "2025-11-25T17:00:00-05:00",
        "master": {"markets": 3, "trades": 3},
        "splits": {split: {"markets": 1, "trades": 1} for split in ["train", "validation", "test"]},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        setup_data,
        "FILES",
        {name: ("fixture-id", hashlib.sha256((folder / name).read_bytes()).hexdigest()) for name in setup_data.FILES},
    )
    return folder


def test_install_and_recheck_without_extracting_unlisted_paths(downloads, tmp_path):
    files = setup_data.get_files(downloads, download=False)
    output = tmp_path / "installed"
    setup_data.install_dataset(files, output)
    setup_data.validate_dataset(output)
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    expected = {member for members in setup_data.MEMBERS.values() for member in members} | {"manifest.json"}
    assert actual == expected
    assert not list(tmp_path.rglob("escape.txt"))
    assert not list(tmp_path.glob(".kalshi_extract_*"))


def test_rejects_corrupt_download(downloads):
    (downloads / "raw_data.zip").write_bytes(b"not the expected ZIP")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        setup_data.get_files(downloads, download=False)


def test_manual_download_requires_manifest(downloads):
    (downloads / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        setup_data.get_files(downloads, download=False)


def test_existing_destination_preserved(downloads, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing data")
    with pytest.raises(FileExistsError):
        setup_data.install_dataset(setup_data.get_files(downloads, download=False), output)
    assert marker.read_text() == "existing data"


def test_failed_validation_does_not_publish_partial_dataset(downloads, tmp_path):
    files = setup_data.get_files(downloads, download=False)
    manifest = json.loads(files["manifest.json"].read_text())
    manifest["master"]["markets"] = 6
    for split in manifest["splits"].values():
        split["markets"] = 2
    files["manifest.json"].write_text(json.dumps(manifest))
    output = tmp_path / "installed"
    with pytest.raises(ValueError, match="Row count mismatch"):
        setup_data.install_dataset(files, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".kalshi_extract_*"))


def test_missing_archive_member_rejected(downloads, tmp_path):
    files = setup_data.get_files(downloads, download=False)
    with ZipFile(files["split.zip"], "w") as archive:
        archive.writestr("unexpected.csv", "bad layout")
    with pytest.raises(ValueError, match="Expected exactly one"):
        setup_data.install_dataset(files, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


def test_missing_notebook_columns_rejected(downloads, tmp_path):
    output = tmp_path / "installed"
    setup_data.install_dataset(setup_data.get_files(downloads, download=False), output)
    (output / "master/markets.csv").write_text("ticker\na\nb\nc\n")
    with pytest.raises(ValueError, match="Missing columns"):
        setup_data.validate_dataset(output)


def test_cached_files_skip_network_and_new_downloads_are_verified(downloads, tmp_path, monkeypatch):
    calls = []

    def fake_download(*, id, output, **kwargs):
        calls.append(output)
        name = setup_data.Path(output).name.removesuffix(".part")
        setup_data.Path(output).write_bytes((downloads / name).read_bytes())
        return output

    monkeypatch.setattr(setup_data.gdown, "download", fake_download)
    cache = tmp_path / "cache"
    first = setup_data.get_files(cache, download=True)
    assert len(calls) == 3
    second = setup_data.get_files(cache, download=True)
    assert len(calls) == 3
    assert first == second


def test_download_failure_explains_manual_fallback(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Drive download limit")

    monkeypatch.setattr(setup_data.gdown, "download", fail)
    with pytest.raises(RuntimeError, match="--from-dir"):
        setup_data.get_files(tmp_path, download=True)
    assert not (tmp_path / "raw_data.zip").exists()
