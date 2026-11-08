"""Notebook plumbing only: paths, provenance and bounded artifact inspection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

NOTEBOOKS = [
    "01_data_and_clustering.ipynb",
    "02_measuring_mispricing.ipynb",
    "03_historical_and_supervised_prediction.ipynb",
    "04_evaluation_and_conclusions.ipynb",
]


def read_completed(directory: Path, source: Path | None = None, stage: str | None = None) -> dict:
    """Reject incomplete, wrong-stage or changed-source artifacts rather than relabel them."""
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("completed") is not True:
        raise ValueError(f"Incomplete run: {directory}")
    if stage is not None and manifest.get("stage") != stage:
        raise ValueError(f"Expected stage {stage}: {directory}")
    if source is not None:
        for name, expected in manifest.get("source_hashes", {}).items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Artifact implementation changed: {name}")
    return manifest


def published_paths(root: Path) -> tuple[dict, Path, Path, Path, Path]:
    result = root / "extension/clustering/supervised/results"
    manifest = read_completed(result, root / "extension/clustering/supervised")

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    config = manifest["config"]
    return manifest, resolve(config["data"]), resolve(config["assignments"]), resolve(config["stats"]), result


def output_dir(root: Path, run_name: str, stage: str) -> Path:
    """Restrict notebook writes to a dedicated ignored subtree, including symlink checks."""
    base = (root / "extension/notebooks/results").resolve()
    path = (base / run_name / stage).resolve()
    if base not in path.parents:
        raise ValueError("Notebook outputs must be below extension/notebooks/results/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def notebook_digest(path: Path) -> str:
    """Hash authored cells, excluding execution outputs and transient kernel metadata."""
    notebook = json.loads(path.read_text())
    cells = [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]
    return hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest()


def begin_stage(root: Path, directory: Path, stage: str, inputs: list[Path], config: dict) -> dict:
    from extension.clustering.mispricing.pipeline import fingerprint

    base = (root / "extension/notebooks/results").resolve()
    if base not in directory.resolve().parents:
        raise ValueError("Unsafe notebook output directory")
    manifest = {
        "completed": False,
        "interface": "completion_notebook",
        "stage": stage,
        "config": config,
        "inputs": fingerprint(inputs),
        "notebook_cell_hashes": {name: notebook_digest(root / "extension/notebooks" / name) for name in NOTEBOOKS},
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "extension/clustering").rglob("*.py"))
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "duckdb"]
        },
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def finish_stage(directory: Path, manifest: dict, **extra) -> None:
    manifest.update(extra)
    manifest["completed"] = True
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def prediction_sample(path: Path, columns: list[str], rows: int = 2000) -> pd.DataFrame:
    """Read a bounded leading sample, explicitly not a population estimate."""
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=rows, columns=columns))
    return batch.to_pandas()
