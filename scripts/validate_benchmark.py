#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Validate prepared data and prediction IDs/counts for benchmark cells."""

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
from benchmarks.io_utils import atomic_write_json  # noqa: E402
from benchmarks.validation import (  # noqa: E402
    validate_grid,
    validate_scored,
    write_completion,
)


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
    parser.add_argument("--models", default=" ".join(MODEL_SPECS))
    parser.add_argument("--tasks", default=" ".join(TASK_SIZES))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--level",
        choices=("predictions", "scored", "complete"),
        default="complete",
    )
    parser.add_argument(
        "--write-complete",
        action="store_true",
        help="Write completion markers after scored validation succeeds.",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        models = parse_names(args.models, list(MODEL_SPECS))
        tasks = parse_names(args.tasks, list(TASK_SIZES))
        if args.write_complete:
            for model in models:
                for task in tasks:
                    write_completion(
                        args.data_dir,
                        args.output_dir,
                        model=model,
                        task=task,
                        max_samples=args.max_samples,
                    )
            level = "complete"
        else:
            level = args.level

        if level in {"predictions", "complete"}:
            report = validate_grid(
                args.data_dir,
                args.output_dir,
                models=models,
                tasks=tasks,
                max_samples=args.max_samples,
                require_completion=level == "complete",
            )
        else:
            cells = []
            errors = []
            for model in models:
                for task in tasks:
                    try:
                        cells.append(
                            validate_scored(
                                args.data_dir,
                                args.output_dir,
                                model=model,
                                task=task,
                                max_samples=args.max_samples,
                            )
                        )
                    except (FileNotFoundError, ValueError, OSError) as error:
                        errors.append(
                            {"model": model, "task": task, "error": str(error)}
                        )
            report = {
                "ok": not errors,
                "level": "scored",
                "valid_cells": len(cells),
                "expected_cells": len(models) * len(tasks),
                "cells": cells,
                "errors": errors,
            }
    except (FileNotFoundError, ValueError, OSError) as error:
        report = {"ok": False, "errors": [{"error": str(error)}]}

    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
