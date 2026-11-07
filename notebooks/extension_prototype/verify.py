import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

root = Path.cwd()
base = root / "notebooks/extension_prototype"
for p in sorted(base.glob("*.ipynb")):
    nb = nbformat.read(p, as_version=4)
    nbformat.validate(nb)
    ns = {"__doc__": "Notebook validation"}
    for c in nb.cells:
        if c.cell_type == "code":
            compile(c.source, str(p), "exec")
    for i in [2, 4, 6]:
        exec(nb.cells[i].source, ns)
    assert (root / "output/notebook_extension").resolve() in ns["args"].output.resolve().parents
    print("SCHEMA, COMPILE, IMPORT, CONFIG:", p.name, flush=True)

data = root / "output/notebook_extension/validation/synthetic_csv"
for split in ["train", "validation", "test"]:
    rows = []
    for i in range(40):
        rows.append(
            {
                "ticker": f"{split}-{i}",
                "event_ticker": f"KXBTCD-{split}{i}",
                "market_type": "binary",
                "title": f"Bitcoin price above {90000 + (i % 8) * 1000}?",
                "yes_sub_title": f"{90000 + (i % 8) * 1000} or above",
                "open_time": "",
                "close_time": "",
            }
        )
    d = data / "split" / split
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "markets.csv", index=False)
p = base / "02_csv_clustering.ipynb"
nb = nbformat.read(p, as_version=4)
nbout = root / "output/notebook_extension/validation/notebook_clustering"
cliout = root / "output/notebook_extension/validation/cli_clustering"
fixture_settings = ["--min-contracts", "4", "--min-events", "4"]
nb.cells[4].source = "NOTEBOOK_ARGUMENTS = " + repr(["--data", str(data), "--output", str(nbout), *fixture_settings])
# Execute real Jupyter cells in a kernel; keep all generated outputs ignored.
NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(nb, root / "output/notebook_extension/validation/executed_clustering.ipynb")
subprocess.run(
    [
        sys.executable,
        "-m",
        "extension.clustering.pipeline",
        "--data",
        str(data),
        "--output",
        str(cliout),
        *fixture_settings,
    ],
    check=True,
)
for name in ["assignments.csv", "coverage.csv", "cluster_profiles.csv", "review_examples.csv"]:
    pd.testing.assert_frame_equal(pd.read_csv(nbout / name), pd.read_csv(cliout / name))
assert pd.read_csv(nbout / "assignments.csv").status.eq("assigned").any()
print("FULL SYNTHETIC NOTEBOOK EXECUTION + CLI PARITY: 4 CSV outputs match", flush=True)
