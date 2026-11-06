#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Score canonical benchmark predictions with the fixed shared scorer."""

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
from benchmarks.scoring import score_grid  # noqa: E402


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
        "--complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = score_grid(
            args.data_dir,
            args.output_dir,
            models=parse_names(args.models, list(MODEL_SPECS)),
            tasks=parse_names(args.tasks, list(TASK_SIZES)),
            max_samples=args.max_samples,
            complete=args.complete,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        result = {"ok": False, "errors": [{"error": str(error)}]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
