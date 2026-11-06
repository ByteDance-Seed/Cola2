# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Count, identifier, metadata, and completion validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import CONTRACT_VERSION, MODEL_SPECS, TASK_SIZES
from .data import load_task_data
from .io_utils import (
    atomic_write_json,
    canonical_id,
    ids_sha256,
    read_jsonl,
    sha256_file,
    unique_by_id,
)


PREDICTIONS_NAME = "predictions.jsonl"
SCORED_NAME = "scored.jsonl"
METRICS_NAME = "metrics.json"
RUN_CONFIG_NAME = "run_config.json"
COMPLETE_NAME = "complete.json"


def cell_dir(output_dir: str | Path, model: str, task: str) -> Path:
    return Path(output_dir) / model / task


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_predictions(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    task: str,
    max_samples: int,
) -> dict[str, Any]:
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown model: {model}")
    inputs = load_task_data(data_dir, task, max_samples=max_samples)
    directory = cell_dir(output_dir, model, task)
    predictions_path = directory / PREDICTIONS_NAME
    if not predictions_path.is_file():
        raise FileNotFoundError(f"missing predictions: {predictions_path}")
    predictions = read_jsonl(predictions_path)
    input_index = unique_by_id(inputs, source=f"input:{task}")
    prediction_index = unique_by_id(predictions, source=str(predictions_path))

    missing = [key for key in input_index if key not in prediction_index]
    extra = [key for key in prediction_index if key not in input_index]
    if missing or extra:
        raise ValueError(
            f"{predictions_path}: id mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    if len(predictions) != len(inputs):
        raise ValueError(
            f"{predictions_path}: expected {len(inputs)} records, found {len(predictions)}"
        )

    for key, source in input_index.items():
        prediction = prediction_index[key]
        if prediction.get("task") != task:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} has task="
                f"{prediction.get('task')!r}"
            )
        if prediction.get("model") != model:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} has model="
                f"{prediction.get('model')!r}"
            )
        if not isinstance(prediction.get("generate"), str):
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} has no string generate"
            )
        for field in ("prompt", "ground_truth"):
            if prediction.get(field) != source.get(field):
                raise ValueError(
                    f"{predictions_path}: id={source['id']!r} changed {field}"
                )
        if "choices" in source and prediction.get("choices") != source["choices"]:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} changed choices"
            )
        if "source_id" in source and prediction.get("source_id") != source["source_id"]:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} changed source_id"
            )
        if task == "truthfulqa" and not prediction.get("choices"):
            raise ValueError(
                f"{predictions_path}: truthfulqa id={source['id']!r} lost choices"
            )

    return {
        "model": model,
        "task": task,
        "count": len(predictions),
        "ids_sha256": ids_sha256(inputs),
        "predictions_sha256": sha256_file(predictions_path),
    }


def validate_scored(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    task: str,
    max_samples: int,
) -> dict[str, Any]:
    base = validate_predictions(
        data_dir,
        output_dir,
        model=model,
        task=task,
        max_samples=max_samples,
    )
    directory = cell_dir(output_dir, model, task)
    scored_path = directory / SCORED_NAME
    metrics_path = directory / METRICS_NAME
    run_config_path = directory / RUN_CONFIG_NAME
    for path in (scored_path, metrics_path, run_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing completed artifact: {path}")

    scored = read_jsonl(scored_path)
    scored_index = unique_by_id(scored, source=str(scored_path))
    inputs = load_task_data(data_dir, task, max_samples=max_samples)
    input_keys = {canonical_id(record["id"]) for record in inputs}
    if set(scored_index) != input_keys or len(scored) != len(inputs):
        raise ValueError(f"{scored_path}: scored IDs/count do not match input")
    for record in scored:
        scoring = record.get("scoring")
        if not isinstance(scoring, dict) or not isinstance(
            scoring.get("correct"), bool
        ):
            raise ValueError(
                f"{scored_path}: id={record.get('id')!r} has invalid scoring"
            )

    metrics = _load_json_object(metrics_path)
    if metrics.get("model") != model or metrics.get("task") != task:
        raise ValueError(f"{metrics_path}: model/task contract mismatch")
    if metrics.get("total") != len(inputs):
        raise ValueError(
            f"{metrics_path}: total={metrics.get('total')}, expected {len(inputs)}"
        )
    correct = sum(int(record["scoring"]["correct"]) for record in scored)
    if metrics.get("correct") != correct:
        raise ValueError(
            f"{metrics_path}: correct={metrics.get('correct')}, expected {correct}"
        )
    expected_accuracy = round(100.0 * correct / len(inputs), 2) if inputs else None
    if metrics.get("acc") != expected_accuracy:
        raise ValueError(
            f"{metrics_path}: acc={metrics.get('acc')}, expected {expected_accuracy}"
        )
    if metrics.get("degraded_no_choices"):
        raise ValueError(f"{metrics_path}: scorer degraded because choices are missing")

    base.update(
        {
            "scored_sha256": sha256_file(scored_path),
            "metrics_sha256": sha256_file(metrics_path),
            "run_config_sha256": sha256_file(run_config_path),
            "correct": correct,
            "acc": expected_accuracy,
        }
    )
    return base


def expected_completion(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    task: str,
    max_samples: int,
) -> dict[str, Any]:
    validated = validate_scored(
        data_dir,
        output_dir,
        model=model,
        task=task,
        max_samples=max_samples,
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "model": model,
        "task": task,
        "max_samples": max_samples,
        **validated,
    }


def write_completion(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    task: str,
    max_samples: int,
) -> dict[str, Any]:
    marker = expected_completion(
        data_dir,
        output_dir,
        model=model,
        task=task,
        max_samples=max_samples,
    )
    atomic_write_json(cell_dir(output_dir, model, task) / COMPLETE_NAME, marker)
    return marker


def validate_completion(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    task: str,
    max_samples: int,
) -> dict[str, Any]:
    path = cell_dir(output_dir, model, task) / COMPLETE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing completion marker: {path}")
    actual = _load_json_object(path)
    expected = expected_completion(
        data_dir,
        output_dir,
        model=model,
        task=task,
        max_samples=max_samples,
    )
    if actual != expected:
        raise ValueError(f"{path}: stale or mismatched completion marker")
    return actual


def is_cell_complete(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    task: str,
    max_samples: int,
) -> tuple[bool, str | None]:
    try:
        validate_completion(
            data_dir,
            output_dir,
            model=model,
            task=task,
            max_samples=max_samples,
        )
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
        return False, str(error)
    return True, None


def validate_grid(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    models: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    max_samples: int = 0,
    require_completion: bool = True,
) -> dict[str, Any]:
    selected_models = list(models or MODEL_SPECS)
    selected_tasks = list(tasks or TASK_SIZES)
    cells: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model in selected_models:
        for task in selected_tasks:
            try:
                if require_completion:
                    result = validate_completion(
                        data_dir,
                        output_dir,
                        model=model,
                        task=task,
                        max_samples=max_samples,
                    )
                else:
                    result = validate_predictions(
                        data_dir,
                        output_dir,
                        model=model,
                        task=task,
                        max_samples=max_samples,
                    )
                cells.append(result)
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
                errors.append(
                    {"model": model, "task": task, "error": str(error)}
                )
    return {
        "contract_version": CONTRACT_VERSION,
        "max_samples": max_samples,
        "models": selected_models,
        "tasks": selected_tasks,
        "valid_cells": len(cells),
        "expected_cells": len(selected_models) * len(selected_tasks),
        "ok": not errors,
        "cells": cells,
        "errors": errors,
    }
