# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Apply the fixed scorer to canonical prediction files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .contracts import MODEL_SPECS
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    clean_generation,
    read_jsonl,
    sha256_file,
)
from .task_scoring import score_records
from .validation import (
    COMPLETE_NAME,
    METRICS_NAME,
    PREDICTIONS_NAME,
    SCORED_NAME,
    cell_dir,
    validate_predictions,
    write_completion,
)


def score_cell(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    task: str,
    max_samples: int,
    complete: bool = True,
) -> dict[str, Any]:
    validated = validate_predictions(
        data_dir,
        output_dir,
        model=model,
        task=task,
        max_samples=max_samples,
    )
    directory = cell_dir(output_dir, model, task)
    completion_path = directory / COMPLETE_NAME
    completion_path.unlink(missing_ok=True)

    records = read_jsonl(directory / PREDICTIONS_NAME)
    scored_records: list[dict[str, Any]] = []
    for prediction in records:
        record = dict(prediction)
        cleaned, remainder = clean_generation(record.get("generate", ""))
        record["generate"] = cleaned
        if remainder is not None:
            record["others"] = remainder
        scored_records.append(record)

    metrics = score_records(task, scored_records)
    metrics.update(
        {
            "model": model,
            "model_label": MODEL_SPECS[model].label,
            "max_samples": max_samples,
            "ids_sha256": validated["ids_sha256"],
            "predictions_sha256": validated["predictions_sha256"],
            "scorer_sha256": sha256_file(Path(__file__).with_name("task_scoring.py")),
        }
    )
    atomic_write_jsonl(directory / SCORED_NAME, scored_records)
    atomic_write_json(directory / METRICS_NAME, metrics)
    if complete:
        write_completion(
            data_dir,
            output_dir,
            model=model,
            task=task,
            max_samples=max_samples,
        )
    return metrics


def score_grid(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    models: Iterable[str],
    tasks: Iterable[str],
    max_samples: int,
    complete: bool = True,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model in models:
        for task in tasks:
            try:
                results.append(
                    score_cell(
                        data_dir,
                        output_dir,
                        model=model,
                        task=task,
                        max_samples=max_samples,
                        complete=complete,
                    )
                )
            except (FileNotFoundError, ValueError, OSError) as error:
                errors.append(
                    {"model": model, "task": task, "error": str(error)}
                )
    return {"ok": not errors, "results": results, "errors": errors}
