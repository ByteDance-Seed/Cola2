#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Write unified benchmark summary.json, summary.csv, and summary.md."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

COLA2_ROOT = Path(__file__).resolve().parents[1]
if str(COLA2_ROOT) not in sys.path:
    sys.path.insert(0, str(COLA2_ROOT))

from benchmarks.contracts import MODEL_SPECS, TASK_SIZES, parse_names  # noqa: E402
from benchmarks.summary import build_summary  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    root = Path(
        os.environ.get(
            "COLA2_BENCHMARK_ROOT",
            str(COLA2_ROOT / "benchmark_artifacts"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--summary-dir", type=Path, default=root / "summary")
    parser.add_argument("--models", default=" ".join(MODEL_SPECS))
    parser.add_argument("--tasks", default=" ".join(TASK_SIZES))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return a non-zero status unless every requested cell is complete.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_summary(
            args.data_dir,
            args.output_dir,
            args.summary_dir,
            models=parse_names(args.models, list(MODEL_SPECS)),
            tasks=parse_names(args.tasks, list(TASK_SIZES)),
            max_samples=args.max_samples,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": summary["complete"],
                "completed_cells": summary["completed_cells"],
                "expected_cells": summary["expected_cells"],
                "summary_dir": str(args.summary_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_complete and not summary["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
