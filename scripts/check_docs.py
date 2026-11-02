"""Offline checks for maintained English documentation, links, and SVGs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

LINK = re.compile(r"\[[^\]\n]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
NON_ENGLISH_PROSE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def anchors(markdown: str) -> set[str]:
    result = set()
    counts: dict[str, int] = {}
    for line in markdown.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        base = re.sub(r"[^\w -]", "", match[1].lower()).replace(" ", "-")
        ordinal = counts.get(base, 0)
        counts[base] = ordinal + 1
        result.add(base if ordinal == 0 else f"{base}-{ordinal}")
    return result


def documentation_files(repo: Path) -> list[Path]:
    # Exclude captured research snapshots and fixed data by design.
    guides = [repo / folder / "README.md" for folder in ("results", "data", "reference")]
    return sorted({*repo.glob("*.md"), *(repo / "docs").rglob("*.md"),
                   *(path for path in guides if path.exists())})


def check_documentation(repo: Path) -> dict:
    repo = repo.resolve()
    errors: list[str] = []
    links_checked = 0
    files = documentation_files(repo)
    for path in files:
        name = path.relative_to(repo).as_posix()
        text = path.read_text(encoding="utf-8")
        if NON_ENGLISH_PROSE.search(text):
            errors.append(f"{name}: current prose must be English; do not alter archived evidence")
        for match in LINK.finditer(text):
            url = urlsplit(match[1].strip("<>"))
            if url.scheme or url.netloc:
                continue  # No network access or flaky external-link checks in CI.
            links_checked += 1
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(repo):
                errors.append(f"{name}: link escapes repository: {match[1]}")
            elif not target.exists():
                errors.append(f"{name}: missing link target: {match[1]}")
            elif url.fragment and target.suffix == ".md":
                if unquote(url.fragment) not in anchors(target.read_text(encoding="utf-8")):
                    errors.append(f"{name}: missing heading anchor: {match[1]}")
    svg_files = sorted((repo / "docs").rglob("*.svg"))
    for path in svg_files:
        name = path.relative_to(repo).as_posix()
        raw = path.read_text(encoding="utf-8")
        if re.search(r"<!DOCTYPE|<!ENTITY|<\?xml-stylesheet|@import|url\(\s*['\"]?(?:https?:|//)", raw, re.I):
            errors.append(f"{name}: external or active SVG content")
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            errors.append(f"{name}: invalid SVG: {exc}")
            continue
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            errors.append(f"{name}: not an SVG document")
        if not root.findtext("{http://www.w3.org/2000/svg}title") or not root.findtext("{http://www.w3.org/2000/svg}desc"):
            errors.append(f"{name}: missing accessible title or description")
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] in {"script", "foreignObject"}:
                errors.append(f"{name}: active SVG element")
            for key, value in node.attrib.items():
                local = key.rsplit("}", 1)[-1]
                if local.startswith("on") or (local == "href" and not value.startswith("#")):
                    errors.append(f"{name}: unsafe SVG attribute")
        if NON_ENGLISH_PROSE.search("".join(root.itertext())):
            errors.append(f"{name}: current diagram text must be English")
    return {"passed": not errors, "markdown_files_checked": len(files),
            "local_links_checked": links_checked, "documentation_svgs_checked": len(svg_files),
            "external_links_fetched": 0, "errors": errors,
            "scope": "maintained documentation only; original source code and scientific evidence are excluded"}


def write_report(repo: Path, output: Path | None, report: dict) -> None:
    raw = json.dumps(report, indent=2) + "\n"
    if output is not None:
        target = output.resolve()
        if target.is_relative_to(repo.resolve()) and not target.is_relative_to((repo / ".ci-reports").resolve()):
            raise ValueError("Repository reports must be written under .ci-reports/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(raw, encoding="utf-8")
    print(raw, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_documentation(args.repo)
    write_report(args.repo, args.output, result)
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
