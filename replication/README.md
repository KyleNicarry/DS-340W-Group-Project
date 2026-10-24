# Article replication workflow

This team-created workflow reproduces an editable version of Jonathan Becker’s *The Microstructure of Wealth Transfer in Prediction Markets*. The original research is Becker’s; the separate [research extension](../extension/README.md) contains our new analysis.

Run from the repository root:

```bash
uv sync
uv run python replication/build_article.py
```

The build requires `xelatex` on your PATH and the figures referenced by `replicate_article.tex` in `output/`. It writes `output/pdf/replicate_article.pdf`.

To run all registered inherited analyses before compiling (requires their input datasets):

```bash
uv run python replication/build_article.py --run-analyses
```

The Markdown article in this directory contains editable article text; `replicate_article.tex` controls the compiled article and layout. Review both when changing the article: the build compiles the LaTeX source directly and does not convert the Markdown file. `generate_replica_figures.py` supplies an additional figure-generation script.

See [contributions and provenance](../docs/contributions.md) for the upstream baseline and attribution.
