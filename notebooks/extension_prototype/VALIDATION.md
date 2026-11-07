# Prototype validation record

Validated on 2026-11-06 with the repository's Python 3.9 environment.

| Check | Result |
| --- | --- |
| Notebook schema validation (`nbformat`) | All four passed |
| Compile every code cell | All four passed |
| Execute imports and configuration | All four passed |
| Notebook 02, fresh Jupyter kernel, synthetic CSV metadata | Completed |
| Notebook 02 versus original CSV clustering CLI | All four exported CSV tables matched exactly |
| `make test` | 152 passed; 5 warnings |
| Ruff lint and formatting for `notebooks/extension_prototype/` | Passed |
| Repository-wide lint | 87 existing errors in unchanged files; not modified here |

The synthetic fixture contains 120 contracts across three event-disjoint splits.
It uses four-contract/four-event support thresholds in both notebook and CLI to
exercise assignment with a small fixture. Delivered notebook defaults retain the
original 20-contract/10-event thresholds. Synthetic results are not research findings.

The executed validation notebook and fixture outputs are generated artifacts under
`output/notebook_extension/validation/`; they are not part of the submitted source.

## Not yet established

- Full real-data execution and CLI parity for the Parquet prediction notebook.
- Full real-data execution and inference parity for CSV mispricing and structural families.
- CSV production runs: the default prepared CSV input directory was not present locally.
- Scientific validity, chronological availability of training outcomes, or trading profitability.

The production Python extension and its results remain unchanged. Follow the README's
real-data comparison procedure before treating the notebooks as a replacement.
