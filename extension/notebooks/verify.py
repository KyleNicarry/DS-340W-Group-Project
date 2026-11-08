"""Validate and execute the completion notebooks against existing real artifacts.

Default: fresh-kernel review execution for all four notebooks. Optional:
--recompute-prediction executes stages 03/04 in isolated directories and checks
numerical parity with the completed CLI experiment. No synthetic data substitution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extension.notebooks.support import NOTEBOOKS, notebook_digest, published_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-prediction", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "extension/notebooks/results/validation"
    destination.mkdir(parents=True, exist_ok=True)
    published, _, _, _, reference = published_paths(ROOT)
    record = {
        "completed": False,
        "mode": "real_artifact_review",
        "notebooks": {},
        "reference_manifest_sha256": hashlib.sha256((reference / "run_manifest.json").read_bytes()).hexdigest(),
        "reference_completed": published["completed"],
    }
    manifest_path = destination / "verification.json"
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    for name in NOTEBOOKS:
        path = ROOT / "extension/notebooks" / name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, name, "exec")
                if cell.outputs or cell.execution_count is not None:
                    raise ValueError(f"Authored notebook must be output-free: {name}")
        print("SCHEMA / COMPILE:", name, flush=True)
        client = NotebookClient(
            notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
        )
        client.execute()
        nbformat.write(notebook, destination / name)
        record["notebooks"][name] = {"cell_source_sha256": notebook_digest(path), "fresh_kernel_review": True}
        print("REAL-ARTIFACT REVIEW EXECUTED:", name, flush=True)
    if args.recompute_prediction:
        record["recompute_prediction"] = False
        for name in NOTEBOOKS[2:]:
            notebook = nbformat.read(ROOT / "extension/notebooks" / name, as_version=4)
            parameters = next(
                cell for cell in notebook.cells if cell.cell_type == "code" and "RUN_NAME =" in cell.source
            )
            parameters.source = parameters.source.replace("RECOMPUTE = False", "RECOMPUTE = True", 1)
            if name.startswith("04"):
                parameters.source = parameters.source.replace(
                    "USE_REBUILT_UPSTREAM = False", "USE_REBUILT_UPSTREAM = True", 1
                )
            client = NotebookClient(
                notebook, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
            )
            client.execute()
            nbformat.write(notebook, destination / ("recomputed_" + name))
            print("REAL-DATA RECOMPUTATION EXECUTED:", name, flush=True)
        result = ROOT / "extension/notebooks/results/completion_walkthrough/evaluation"
        for filename, keys in [
            ("metrics.csv", ["split", "model"]),
            ("holdout_coverage.csv", ["split"]),
            ("wedge_metrics.csv", ["split", "model"]),
            ("comparisons.csv", ["split", "candidate", "baseline", "dependence_unit"]),
        ]:
            actual = pd.read_csv(result / filename).sort_values(keys).reset_index(drop=True)
            expected = pd.read_csv(reference / filename).sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-12)
        current_settings = json.loads((result / "historical_selected.json").read_text())
        expected_settings = json.loads((reference / "historical_selected.json").read_text())
        for scope in current_settings:
            assert current_settings[scope]["strength"] == expected_settings[scope]["strength"]
            assert current_settings[scope]["width"] == expected_settings[scope]["width"]
            np.testing.assert_allclose(
                current_settings[scope]["validation_brier"], expected_settings[scope]["validation_brier"], atol=1e-14
            )
        record["recompute_prediction"] = True
        record["parity_tables"] = ["metrics.csv", "holdout_coverage.csv", "wedge_metrics.csv", "comparisons.csv"]
        record["parameter_overrides"] = "03 RECOMPUTE=True; 04 RECOMPUTE=True and USE_REBUILT_UPSTREAM=True"
        print("CLI ARTIFACT PARITY: four tables and historical settings match.", flush=True)
    record["completed"] = True
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    print("VERIFICATION COMPLETE:", manifest_path, flush=True)


if __name__ == "__main__":
    main()
