"""Guard notebook artifact provenance and isolated writes without requiring Jupyter."""

import hashlib
import json

import pytest

from extension.notebooks.support import notebook_digest, output_dir, read_completed


def test_review_rejects_incomplete_and_changed_source(tmp_path):
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"completed": False}))
    with pytest.raises(ValueError, match="Incomplete"):
        read_completed(tmp_path)
    source = tmp_path / "analysis.py"
    source.write_text("original")
    manifest_path.write_text(
        json.dumps(
            {
                "completed": True,
                "stage": "prediction_fit",
                "source_hashes": {"analysis.py": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
        )
    )
    read_completed(tmp_path, source=tmp_path, stage="prediction_fit")
    with pytest.raises(ValueError, match="Expected stage"):
        read_completed(tmp_path, stage="evaluation")
    source.write_text("changed")
    with pytest.raises(ValueError, match="implementation changed"):
        read_completed(tmp_path, source=tmp_path)


def test_notebook_output_cannot_escape_ignored_subtree(tmp_path):
    assert output_dir(tmp_path, "example", "models").is_dir()
    with pytest.raises(ValueError, match="must be below"):
        output_dir(tmp_path, "../../../../outside", "models")


def test_cell_hash_tracks_authored_code_but_not_saved_outputs(tmp_path):
    path = tmp_path / "example.ipynb"
    content = {"cells": [{"cell_type": "code", "source": ["print(1)"], "outputs": []}]}
    path.write_text(json.dumps(content))
    before = notebook_digest(path)
    content["cells"][0]["outputs"] = [{"text": "1"}]
    path.write_text(json.dumps(content))
    assert notebook_digest(path) == before
    content["cells"][0]["source"] = ["print(2)"]
    path.write_text(json.dumps(content))
    assert notebook_digest(path) != before
