"""Install the shared Kalshi CSV sample used by both research notebooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import duckdb
import gdown

ROOT = Path(__file__).resolve().parents[1]
FOLDER_URL = "https://drive.google.com/drive/folders/1WOUfs-T62UJGP6rafvvSuSlxwAPmMFc8"
# Fixed file IDs and SHA-256 hashes identify the delivered sample, not a moving extract.
FILES = {
    "raw_data.zip": (
        "18WBh9v1o0EgfzpuABzwiJTzEQ6m7cN6r",
        "c2769e8bc3ad7fac3dbcdc2a9270b917a89daf12eba2b1dc55c973c494391e72",
    ),
    "split.zip": (
        "1wpAdJXb29nVuGYnt36t5ZdXqVZJqRmuu",
        "9be1da820de07d42ec292c757e439779eb36cb66d95732a38daa809b591608d9",
    ),
    "manifest.json": (
        "1jbt8Tb3U6OYpdcA0OvBZQJuruXEXeOFS",
        "01690719a3f30e2872405c75dbda51d5de1935a68fb267a334a02d457ed1cdc6",
    ),
}
MEMBERS = {
    "raw_data.zip": [f"master/{table}.csv" for table in ["markets", "trades"]],
    "split.zip": [
        f"split/{split}/{table}.csv" for split in ["train", "validation", "test"] for table in ["markets", "trades"]
    ],
}
MARKET_COLUMNS = {
    "ticker",
    "event_ticker",
    "market_type",
    "title",
    "yes_sub_title",
    "open_time",
    "close_time",
    "status",
    "result",
}
TRADE_COLUMNS = {"trade_id", "ticker", "count", "yes_price", "no_price", "taker_side", "created_time"}


def verify_checksum(path: Path, expected: str) -> None:
    """Reject incomplete downloads or files from a different sample."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}. Move this file aside and download the shared version again.")


def get_files(directory: Path, download: bool) -> dict[str, Path]:
    """Reuse verified downloads; keep incomplete transfers separate from final files."""
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (file_id, checksum) in FILES.items():
        path = directory / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}. Download all three files from {FOLDER_URL}.")
            partial = directory / f"{name}.part"
            print(f"Downloading {name} ...", flush=True)
            try:
                result = gdown.download(id=file_id, output=str(partial), quiet=False, resume=True, use_cookies=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {name}. Retry, or download all three files from {FOLDER_URL} "
                    "in your browser and run this script with --from-dir PATH."
                ) from exc
            if result is None:
                raise RuntimeError(f"Download failed for {name}. Use --from-dir with browser-downloaded files.")
            verify_checksum(partial, checksum)
            partial.rename(path)
        else:
            verify_checksum(path, checksum)
        paths[name] = path
    return paths


def validate_dataset(directory: Path) -> None:
    """Check the notebook columns and all CSV row counts against the manifest."""
    manifest = json.loads((directory / "manifest.json").read_text())
    start = datetime.fromisoformat(manifest["window_start_inclusive"])
    end = datetime.fromisoformat(manifest["window_end_inclusive"])
    if start >= end:
        raise ValueError("Invalid sample window in manifest.json.")
    for table in ["markets", "trades"]:
        if (
            sum(manifest["splits"][split][table] for split in ["train", "validation", "test"])
            != manifest["master"][table]
        ):
            raise ValueError(f"Manifest split counts do not sum to the master {table} count.")
    with duckdb.connect() as con:
        con.execute("SET threads = 2")
        con.execute("SET memory_limit = '1GB'")
        for members in MEMBERS.values():
            for relative in members:
                path = directory / relative
                table = path.stem
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    header = next(csv.reader(stream), [])
                required = MARKET_COLUMNS if table == "markets" else TRADE_COLUMNS
                if not required.issubset(header):
                    raise ValueError(f"Missing columns in {path}: {sorted(required - set(header))}")
                expected = (
                    manifest["master"][table]
                    if relative.startswith("master/")
                    else manifest["splits"][path.parent.name][table]
                )
                actual = con.execute("SELECT count(*) FROM read_csv(?, all_varchar = true)", [str(path)]).fetchone()[0]
                if actual != expected:
                    raise ValueError(f"Row count mismatch in {path}: expected {expected:,}, found {actual:,}.")
                print(f"Verified {relative}: {actual:,} rows", flush=True)


def install_dataset(paths: dict[str, Path], output: Path) -> None:
    """Validate a staged extraction before making the destination available."""
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}. Use --check or choose a new --output directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".kalshi_extract_", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        for name, members in MEMBERS.items():
            with ZipFile(paths[name]) as archive:
                names = archive.namelist()
                for relative in members:
                    if names.count(relative) != 1:
                        raise ValueError(f"Expected exactly one {relative} in {name}.")
                    # Only these fixed paths are extracted; macOS metadata and arbitrary paths are ignored.
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Extracting {relative} ...", flush=True)
                    with archive.open(relative) as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        shutil.copyfile(paths["manifest.json"], stage / "manifest.json")
        validate_dataset(stage)
        if output.exists():
            raise FileExistsError(f"Destination appeared during extraction; refusing to replace {output}.")
        stage.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/kalshi_recent_year_csv")
    parser.add_argument(
        "--from-dir", type=Path, help="Use raw_data.zip, split.zip, and manifest.json already downloaded here."
    )
    parser.add_argument("--check", action="store_true", help="Validate existing CSVs and manifest without downloading.")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.check or output.exists():
        validate_dataset(output)
        print(f"Existing dataset verified; no files changed: {output}")
        return
    downloads = args.from_dir.resolve() if args.from_dir else output.parent / ".kalshi_downloads"
    paths = get_files(downloads, download=args.from_dir is None)
    install_dataset(paths, output)
    print(f"Dataset ready for both notebooks: {output}")
    print(f"Verified downloads retained in: {downloads}")


if __name__ == "__main__":
    main()
