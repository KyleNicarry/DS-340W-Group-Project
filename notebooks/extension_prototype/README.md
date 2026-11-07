# Experimental notebook extension

This is a separate, reviewable notebook interface to the existing extension. It
has not replaced the Python workflows, and it is not an independent replication
of their algorithms. No existing analysis modules or delivered results are edited.
The notebook cells expose the orchestration and call the same reusable functions
as the command-line pipelines. Methodology remains documented in
[the Parquet extension](../../extension/README.md) and
[the CSV workflow](../../extension/clustering/METHODOLOGY.md).

## Open and run

From the repository root:

```bash
uv sync
uv pip install --python .venv/bin/python -r notebooks/extension_prototype/requirements.txt
.venv/bin/python -m jupyterlab notebooks/extension_prototype
```

Choose the project `.venv` Python kernel. If your Jupyter installation uses another
Python, register this environment first:

```bash
.venv/bin/python -m ipykernel install --user --name ds340-extension --display-name "DS 340W extension"
```

Notebooks locate the repository from their working directory. Start Jupyter
inside the repository. Run each notebook top to bottom in a **fresh kernel**;
after changing an upstream cell, restart and rerun downstream cells. Do not rely
on hidden state or cells executed out of order. Saved source notebooks intentionally
have no execution outputs or private data.

| Notebook | Input | Main replaceable stages |
| --- | --- | --- |
| `01_parquet_prediction.ipynb` | `data/kalshi/{markets,trades}/*.parquet` | Early-window preparation; training-only representations; exact-cost labels and models; figures/report |
| `02_csv_clustering.ipynb` | `data/kalshi_recent_year_csv/split/{train,validation,test}/markets.csv` | Eligible metadata; text encoder; frozen family assignment; review tables |
| `03_csv_mispricing.ipynb` | Master market/trade CSVs and notebook 02 assignments | Audited sufficient statistics; populations; exact-cost curves; cluster inference; report |
| `04_structural_families.ipynb` | Master CSVs and notebook 03 audited cache | Rule-only families; matched populations; inference; report |

Notebook 01 is independent. For the CSV route, run **02 → 03 → 04**.
See [input preparation](../../scripts/README.md) for missing CSVs. Notebooks do not
download data, create the CSV splits, or silently substitute synthetic data.
Full runs retain the existing resource requirements and expensive inference.

## Separation and modularity

- Configuration is one cell using the existing CLI parameter definitions and
  defaults. Only the input/output wiring differs from the CLI defaults.
- Inspectable stage cells call existing functions directly, rather than launch
  the entire pipeline with a single shell command. Follow their imports to the
  authoritative implementation. Substitute a function with the same input/output
  contract in a copied notebook to investigate a variation.
- Outputs must be in a subdirectory of `output/notebook_extension/`. The defaults
  use `parquet`, `clustering`, `mispricing`, and `structural`. Reports **and figures**
  go there, including reports that the original CLI normally writes beside source.
  Use a different subdirectory per sensitivity run and update downstream paths.
- Rerunning overwrites files in that notebook's output directory. Configuration
  invalidates its completion marker; completion is set only after analysis and
  reporting succeed. An incomplete run may still contain files from an earlier run.
- Manifests preserve original pipeline source provenance and additionally identify
  the notebook source hash. An executed copy with edited cells must be saved at
  its source path before running for that hash to reflect the edits.
- Splits, training-only fitting, matched-price definitions, sparse-cell rejection,
  uncertainty procedures and default seeds remain those of the Python extension.
  Notebook 01 uses an unseen-series holdout; the CSV workflows preserve their
  supplied event-disjoint splits. Neither is a chronological trading backtest.

## Verification and adoption criteria

Run the small, offline notebook-interface check from the repository root:

```bash
.venv/bin/python notebooks/extension_prototype/verify.py
```

It validates all four notebook schemas, compiles all code cells, executes their
imports and configuration, then runs notebook 02 in a real Jupyter kernel on
explicitly synthetic CSV metadata. It runs the existing CLI on the same fixture
and requires all four clustering CSV outputs to match. Synthetic artifacts and
an executed notebook go under `output/notebook_extension/validation/`. This checks
interface behavior and parity, not economic validity or a successful real-data run.
The kernel needs permission to bind local connection ports.

Before adopting this route, execute all applicable notebooks on the real inputs,
run each corresponding CLI with identical parameters into another isolated output
directory, and compare tables, inclusion counts, splits and numeric results
(allowing documented floating-point tolerance). Compare completion manifests and
inspect figures. The verification script does **not** establish full real-data
parity for notebook 01, 03, or 04. Sparse synthetic inputs also do not validate
inference behavior for the production population.

The core correctness suite is still `uv run pytest tests/extension -q`. A working
notebook interface does not resolve the methodological limitations of the original
extension or establish executable trading profits.
