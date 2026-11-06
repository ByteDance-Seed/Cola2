#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepare the fixed benchmark JSONL files and materialize identical prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

COLA2_ROOT = Path(__file__).resolve().parents[1]
if str(COLA2_ROOT) not in sys.path:
    sys.path.insert(0, str(COLA2_ROOT))

from benchmarks.contracts import TASK_SIZES, parse_names  # noqa: E402
from benchmarks.data import ensure_eval_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    benchmark_root = Path(
        os.environ.get(
            "COLA2_BENCHMARK_ROOT",
            str(COLA2_ROOT / "benchmark_artifacts"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "GENERATE_TASK_DATA_DIR",
                str(COLA2_ROOT / "eval_data"),
            )
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=benchmark_root / "data")
    parser.add_argument("--tasks", default=" ".join(TASK_SIZES))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 keeps the full fixed task sizes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = parse_names(args.tasks, list(TASK_SIZES))
        manifest = ensure_eval_data(
            args.source_dir,
            args.output_dir,
            tasks=tasks,
            max_samples=args.max_samples,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
