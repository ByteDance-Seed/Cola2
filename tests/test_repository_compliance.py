# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("src", "benchmarks", "scripts", "examples", "tests")
COPYRIGHT = "# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates"
SPDX = "# SPDX-License-Identifier: Apache-2.0"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _source_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = REPO_ROOT / root_name
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
        files.extend(path for path in root.rglob("*.sh") if "__pycache__" not in path.parts)
    return sorted(set(files))


def test_source_files_have_apache_spdx_header() -> None:
    failures: list[str] = []
    for path in _source_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        offset = 1 if lines and lines[0].startswith("#!") else 0
        if lines[offset : offset + 2] != [COPYRIGHT, SPDX]:
            failures.append(str(path.relative_to(REPO_ROOT)))
    assert not failures, "missing Apache-2.0 source header: " + ", ".join(failures)


def test_root_license_is_apache_2() -> None:
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text


def test_local_markdown_links_resolve() -> None:
    failures: list[str] = []
    markdown_files = sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if "cola2_arxiv" not in path.parts
        and ".pytest_cache" not in path.parts
    )
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0]
            if (
                not target
                or target.startswith(("#", "http://", "https://", "mailto:"))
            ):
                continue
            relative = unquote(target.split("#", 1)[0])
            destination = (path.parent / relative).resolve()
            if not destination.exists():
                failures.append(
                    f"{path.relative_to(REPO_ROOT)} -> {target}"
                )
    assert not failures, "broken local Markdown links: " + ", ".join(failures)


def test_bilingual_document_pairs_exist() -> None:
    pairs = [
        ("README.md", "README_zh.md"),
        ("docs/README.md", "docs/README_zh.md"),
        ("docs/paper.md", "docs/paper_zh.md"),
        ("docs/architecture.md", "docs/architecture_zh.md"),
        ("docs/inference.md", "docs/inference_zh.md"),
        ("docs/configuration.md", "docs/configuration_zh.md"),
        ("docs/benchmark.md", "docs/benchmark_zh.md"),
        ("docs/reproduction.md", "docs/reproduction_zh.md"),
        ("docs/model_card.md", "docs/model_card_zh.md"),
        ("eval_data/README.md", "eval_data/README_zh.md"),
    ]
    for english, chinese in pairs:
        assert (REPO_ROOT / english).is_file()
        assert (REPO_ROOT / chinese).is_file()
