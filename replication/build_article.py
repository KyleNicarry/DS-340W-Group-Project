"""Build the editable prediction-market article PDF.

Usage:
    python build_article.py                 # compile the LaTeX article
    python build_article.py --run-analyses  # regenerate figures first, then compile

Edit ``The Microstructure of Wealth Transfer in Prediction Markets.md`` for
article text and ``replicate_article.tex`` for layout/typography. The figure
paths are the PNG files in ``output/`` produced by ``src/analysis``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "replication" / "replicate_article.tex"
OUT = ROOT / "output" / "pdf"


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/prediction-market-mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/prediction-market-cache")
    env.setdefault("UV_CACHE_DIR", "/tmp/prediction-market-uv")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-analyses", action="store_true", help="run all registered src analyses first")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.run_analyses:
        run(["uv", "run", "main.py", "analyze", "all"])
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(OUT), str(TEX)])
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(OUT), str(TEX)])
    result = OUT / "replicate_article.pdf"
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
