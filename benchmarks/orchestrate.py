# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed eight-task paper benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .contracts import (
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_SAMPLE_SEED,
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    SAMPLE_SEED_SCHEME,
    TASK_SIZES,
    parse_names,
    shard,
)
from .data import ensure_eval_data, load_task_data
from .inference import InferenceOptions, run_inference
from .sample_rng import validate_base_seed
from .scoring import score_cell
from .summary import build_summary
from .validation import (
    RUN_CONFIG_NAME,
    cell_dir,
    is_cell_complete,
    validate_predictions,
)


COLA2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path(
    os.environ.get(
        "COLA2_BENCHMARK_ROOT",
        str(COLA2_ROOT / "benchmark_artifacts"),
    )
)
DEFAULT_SOURCE_DATA = Path(
    os.environ.get(
        "GENERATE_TASK_DATA_DIR",
        str(COLA2_ROOT / "eval_data"),
    )
)

MODEL_PATH_ENV = {
    "meta-llama-3-8b": ("LLAMA3_MODEL_PATH",),
    "qwen2.5-7b": ("QWEN25_MODEL_PATH",),
    "qwen3-8b-base": ("QWEN3_MODEL_PATH",),
    "llada-8b-base": ("LLADA_MODEL_PATH",),
    "cola2": ("COLA2_MODEL_PATH",),
}


def _key_value(values: list[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise ValueError(f"{option} requires MODEL=VALUE, got {value!r}")
        if key not in MODEL_SPECS:
            raise ValueError(f"{option}: unknown model {key!r}")
        parsed[key] = item
    return parsed


def _model_source(model: str, overrides: dict[str, str]) -> str:
    if model in overrides:
        return overrides[model]
    for environment_name in MODEL_PATH_ENV[model]:
        value = os.environ.get(environment_name)
        if value:
            return value
    default = MODEL_SPECS[model].default_source
    if default:
        return default
    raise ValueError(
        f"{model}: set --model-path {model}=PATH or "
        f"{'/'.join(MODEL_PATH_ENV[model])}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--models", default=" ".join(MODEL_SPECS))
    parser.add_argument("--tasks", default=" ".join(TASK_SIZES))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model-shard-index", type=int, default=0)
    parser.add_argument("--model-shard-count", type=int, default=1)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="MODEL=PATH",
    )
    parser.add_argument(
        "--batch-size",
        action="append",
        default=[],
        metavar="MODEL=N",
    )
    parser.add_argument(
        "--cola2-command",
        default=os.environ.get("COLA2_BENCHMARK_COMMAND"),
        help=(
            "Optional Cola2 CLI adapter instead of the installed package; "
            "see benchmarks.run_inference --help."
        ),
    )
    parser.add_argument("--external-cwd", type=Path, default=COLA2_ROOT)
    parser.add_argument(
        "--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=int(os.environ.get("COLA2_BENCHMARK_SEED", DEFAULT_SAMPLE_SEED)),
        help=(
            "Base seed for deterministic per-record Cola2 latent noise. "
            "Ignored by other backends."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    work_dir = args.work_dir.expanduser().resolve()
    data_dir = (args.data_dir or work_dir / "data").expanduser().resolve()
    output_dir = (args.output_dir or work_dir / "outputs").expanduser().resolve()
    summary_dir = (args.summary_dir or work_dir / "summary").expanduser().resolve()
    return data_dir, output_dir, summary_dir


def _dry_run(
    data_dir: Path,
    output_dir: Path,
    models: list[str],
    tasks: list[str],
    max_samples: int,
    sample_seed: int,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for model in models:
        for task in tasks:
            complete = False
            reason = "prepared data is unavailable"
            if (data_dir / f"{task}.jsonl").is_file():
                complete, reason = is_cell_complete(
                    data_dir,
                    output_dir,
                    model=model,
                    task=task,
                    max_samples=max_samples,
                )
                if complete:
                    complete, reason = _seed_matches(
                        output_dir,
                        model=model,
                        task=task,
                        sample_seed=sample_seed,
                    )
            cells.append(
                {
                    "model": model,
                    "task": task,
                    "action": "skip" if complete else "run",
                    "reason": None if complete else reason,
                }
            )
    return {
        "ok": True,
        "dry_run": True,
        "models": models,
        "tasks": tasks,
        "cells": cells,
    }


def _seed_matches(
    output_dir: Path,
    *,
    model: str,
    task: str,
    sample_seed: int,
) -> tuple[bool, str | None]:
    if MODEL_SPECS[model].backend != "cola2":
        return True, None
    path = cell_dir(output_dir, model, task) / RUN_CONFIG_NAME
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        return False, str(error)
    if not isinstance(config, dict):
        return False, f"{path}: expected a JSON object"
    expected_seed = validate_base_seed(sample_seed)
    if config.get("sample_seed") != expected_seed:
        return (
            False,
            f"{path}: sample_seed changed; rerun with --force",
        )
    if config.get("sample_seed_scheme") != SAMPLE_SEED_SCHEME:
        return (
            False,
            f"{path}: sample seed scheme changed; rerun with --force",
        )
    return True, None


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    requested_models = parse_names(args.models, list(MODEL_SPECS))
    requested_tasks = parse_names(args.tasks, list(TASK_SIZES))
    if "cola2" in requested_models:
        validate_base_seed(args.sample_seed)
    selected_models = shard(
        requested_models, args.model_shard_index, args.model_shard_count
    )
    selected_tasks = shard(
        requested_tasks, args.task_shard_index, args.task_shard_count
    )
    data_dir, output_dir, summary_dir = _resolved_paths(args)

    if args.dry_run:
        return _dry_run(
            data_dir,
            output_dir,
            selected_models,
            selected_tasks,
            args.max_samples,
            args.sample_seed,
        )

    if args.skip_prepare:
        for task in requested_tasks:
            load_task_data(data_dir, task, max_samples=args.max_samples)
    else:
        ensure_eval_data(
            args.source_data_dir,
            data_dir,
            tasks=TASK_SIZES,
            max_samples=args.max_samples,
        )

    model_path_overrides = _key_value(args.model_path, option="--model-path")
    raw_batch_sizes = _key_value(args.batch_size, option="--batch-size")
    batch_sizes: dict[str, int] = {}
    for model, raw_value in raw_batch_sizes.items():
        size = int(raw_value)
        if size < 1:
            raise ValueError(f"--batch-size {model} must be at least 1")
        batch_sizes[model] = size

    errors: list[dict[str, str]] = []
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for model in selected_models:
        tasks_to_process: list[str] = []
        for task in selected_tasks:
            is_complete, _ = is_cell_complete(
                data_dir,
                output_dir,
                model=model,
                task=task,
                max_samples=args.max_samples,
            )
            if is_complete:
                is_complete, _ = _seed_matches(
                    output_dir,
                    model=model,
                    task=task,
                    sample_seed=args.sample_seed,
                )
            if args.resume and not args.force and is_complete:
                skipped.append({"model": model, "task": task})
            else:
                tasks_to_process.append(task)
        if not tasks_to_process:
            continue

        try:
            source = _model_source(model, model_path_overrides)
            options = InferenceOptions(
                model=model,
                model_source=source,
                data_dir=data_dir,
                output_dir=output_dir,
                tasks=tuple(tasks_to_process),
                max_samples=args.max_samples,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=batch_sizes.get(
                    model, MODEL_SPECS[model].default_batch_size
                ),
                max_prompt_length=args.max_prompt_length,
                dtype=args.dtype,
                device=args.device,
                trust_remote_code=args.trust_remote_code,
                resume=args.resume,
                force=args.force,
                sample_seed=args.sample_seed,
                external_command=args.cola2_command,
                external_cwd=args.external_cwd,
            )
            run_inference(options)
        except Exception as error:  # keep independent model shards progressing
            errors.append(
                {"model": model, "task": "*", "error": f"inference: {error}"}
            )
            continue

        for task in tasks_to_process:
            try:
                validate_predictions(
                    data_dir,
                    output_dir,
                    model=model,
                    task=task,
                    max_samples=args.max_samples,
                )
                metrics = score_cell(
                    data_dir,
                    output_dir,
                    model=model,
                    task=task,
                    max_samples=args.max_samples,
                    complete=True,
                )
                completed.append(
                    {
                        "model": model,
                        "task": task,
                        "count": metrics["total"],
                        "acc": metrics["acc"],
                    }
                )
            except Exception as error:
                errors.append(
                    {"model": model, "task": task, "error": f"score: {error}"}
                )

    summary = build_summary(
        data_dir,
        output_dir,
        summary_dir,
        models=requested_models,
        tasks=requested_tasks,
        max_samples=args.max_samples,
    )
    return {
        "ok": not errors,
        "selected_models": selected_models,
        "selected_tasks": selected_tasks,
        "sample_seed": args.sample_seed if "cola2" in selected_models else None,
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "summary": {
            "directory": str(summary_dir),
            "completed_cells": summary["completed_cells"],
            "expected_cells": summary["expected_cells"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
