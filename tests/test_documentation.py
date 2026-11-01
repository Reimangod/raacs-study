"""Regression checks for documentation and non-destructive saved-data parity."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


docs = load_script("check_docs")
reproduction = load_script("check_reproduction")


def test_current_documentation():
    result = docs.check_documentation(ROOT)
    assert result["passed"], result["errors"]
    assert result["external_links_fetched"] == 0


@pytest.mark.parametrize("target", ["missing.md", "../outside.md", "README.md#absent"])
def test_broken_local_links_fail(tmp_path, target):
    (tmp_path / "README.md").write_text(f"# Guide\n[Link]({target})\n")
    assert not docs.check_documentation(tmp_path)["passed"]


def test_valid_anchor_and_external_link(tmp_path):
    (tmp_path / "README.md").write_text("# Guide\n[Here](#guide) [Remote](https://example.invalid/offline)\n")
    result = docs.check_documentation(tmp_path)
    assert result["passed"] and result["local_links_checked"] == 1


def test_archived_original_language_is_not_rewritten(tmp_path):
    (tmp_path / "README.md").write_text("# English guide\n")
    old = tmp_path / "historical"
    old.mkdir()
    # Deliberately build an original-language fixture without non-English source prose.
    (old / "README.md").write_text(chr(0x7814) + chr(0x7A76))
    assert docs.check_documentation(tmp_path)["passed"]
    (tmp_path / "README.md").write_text(chr(0x7814) + chr(0x7A76))
    assert not docs.check_documentation(tmp_path)["passed"]


@pytest.mark.parametrize("payload", ["<script>bad()</script>", '<image href="https://example.invalid/x"/>', '<rect onclick="bad()"/>'])
def test_unsafe_documentation_svg_fails(tmp_path, payload):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "bad.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><title>Title</title><desc>Description</desc>' + payload + '</svg>')
    assert not docs.check_documentation(tmp_path)["passed"]


def test_report_cannot_overwrite_evidence(tmp_path):
    with pytest.raises(ValueError):
        docs.write_report(tmp_path, tmp_path / "data/report.json", {"passed": True})


def test_all_portable_products_match():
    result = reproduction.check_reproduction(ROOT)
    assert result["passed"], result["errors"]
    assert result["files_checked"] == 42 and result["figures_checked"] == 40
    assert result["new_model_samples"] == 0 and result["new_CI_solves"] == 0


def test_changed_missing_and_extra_products_fail(tmp_path):
    generated, committed = tmp_path / "generated", tmp_path / "committed"
    for folder in (generated, committed):
        (folder / "figures").mkdir(parents=True)
        (folder / "endpoint-summary.csv").write_text("same")
        (folder / "reproduction.json").write_text("{}")
        (folder / "figures/a.svg").write_text("same")
    assert reproduction.compare_products(generated, committed)["passed"]
    (generated / "figures/a.svg").write_text("changed")
    assert not reproduction.compare_products(generated, committed)["passed"]
    (generated / "figures/a.svg").unlink()
    assert not reproduction.compare_products(generated, committed)["passed"]
    (generated / "figures/a.svg").write_text("same")
    (generated / "figures/extra.svg").write_text("extra")
    assert not reproduction.compare_products(generated, committed)["passed"]
