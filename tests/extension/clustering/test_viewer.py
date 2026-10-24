"""Validate embedded title content and safe, reproducible offline output."""

import csv

import pytest

from extension.clustering.viewer import FIELDS, build_viewer


def test_viewer_embeds_titles_without_script_injection(tmp_path):
    source = tmp_path / "assignments.csv"
    title = '</script><script>alert("title")</script>'
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({**dict.fromkeys(FIELDS, ""), "ticker": "A", "title": title, "yes_sub_title": "100 or above"})
    output = tmp_path / "review.html"
    assert build_viewer(source, output) == 1
    html = output.read_text()
    assert title not in html
    assert "\\u003c/script>" in html
    assert "100 or above" in html
    assert "__DATA__" not in html and "__RUN__" not in html
    assert html.count("<script>") == 1
    assert build_viewer(source, output) == 1
    assert output.read_text() == html


def test_viewer_rejects_wrong_schema(tmp_path):
    source = tmp_path / "wrong.csv"
    source.write_text("ticker,title\nA,hello\n")
    with pytest.raises(ValueError, match="missing required"):
        build_viewer(source, tmp_path / "review.html")
