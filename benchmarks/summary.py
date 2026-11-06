# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Create aligned JSON, CSV, and Markdown benchmark summaries."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    CONTRACT_VERSION,
    DECODING,
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    SAMPLE_SEED_SCHEME,
    TASK_SIZES,
    expected_count,
)
from .io_utils import atomic_write_json, atomic_write_text
from .validation import (
    METRICS_NAME,
    RUN_CONFIG_NAME,
    cell_dir,
    is_cell_complete,
)


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build_summary(
    data_dir: str | Path,
    output_dir: str | Path,
    summary_dir: str | Path,
    *,
    models: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    import fcntl

    destination = Path(summary_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".summary.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _build_summary_unlocked(
            data_dir,
            output_dir,
            destination,
            models=models,
            tasks=tasks,
            max_samples=max_samples,
        )


def _build_summary_unlocked(
    data_dir: str | Path,
    output_dir: str | Path,
    summary_dir: str | Path,
    *,
    models: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    selected_models = list(models or MODEL_SPECS)
    selected_tasks = list(tasks or TASK_SIZES)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    completed_cells = 0
    sample_seeds: dict[str, set[int]] = {
        model: set() for model in selected_models
    }

    for task in selected_tasks:
        values: dict[str, Any] = {}
        for model in selected_models:
            complete, error = is_cell_complete(
                data_dir,
                output_dir,
                model=model,
                task=task,
                max_samples=max_samples,
            )
            if not complete:
                values[model] = None
                errors.append(
                    {"model": model, "task": task, "error": error or "incomplete"}
                )
                continue
            metrics = _load_metrics(
                cell_dir(output_dir, model, task) / METRICS_NAME
            )
            run_config = _load_metrics(
                cell_dir(output_dir, model, task) / RUN_CONFIG_NAME
            )
            sample_seed = run_config.get("sample_seed")
            if sample_seed is not None:
                sample_seeds[model].add(int(sample_seed))
            values[model] = {
                "acc": metrics["acc"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                **(
                    {"sample_seed": int(sample_seed)}
                    if sample_seed is not None
                    else {}
                ),
                **{
                    key: metrics[key]
                    for key in ("chance", "fidelity", "f1")
                    if key in metrics
                },
            }
            completed_cells += 1
        rows.append(
            {
                "task": task,
                "N": expected_count(task, max_samples),
                "models": values,
            }
        )

    means: dict[str, float | None] = {}
    for model in selected_models:
        accuracies = [
            row["models"][model]["acc"]
            for row in rows
            if row["models"][model] is not None
        ]
        means[model] = (
            round(sum(accuracies) / len(accuracies), 2)
            if len(accuracies) == len(selected_tasks)
            else None
        )

    serialized_sample_seeds: dict[str, int | None] = {}
    for model, values in sample_seeds.items():
        if len(values) > 1:
            errors.append(
                {
                    "model": model,
                    "task": "*",
                    "error": (
                        "completed cells use inconsistent sample seeds: "
                        + ", ".join(str(value) for value in sorted(values))
                    ),
                }
            )
            serialized_sample_seeds[model] = None
        else:
            serialized_sample_seeds[model] = (
                next(iter(values)) if values else None
            )

    summary: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "settings": {
            "max_samples": max_samples,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": DECODING,
            "scorer": "benchmarks/task_scoring.py",
            "sample_seeds": serialized_sample_seeds,
            "sample_seed_scheme": SAMPLE_SEED_SCHEME,
        },
        "models": [
            {"slug": model, "label": MODEL_SPECS[model].label}
            for model in selected_models
        ],
        "tasks": selected_tasks,
        "completed_cells": completed_cells,
        "expected_cells": len(selected_models) * len(selected_tasks),
        "complete": not errors,
        "rows": rows,
        "mean": means,
        "errors": errors,
    }

    destination = Path(summary_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "summary.json", summary)
    atomic_write_text(destination / "summary.csv", _render_csv(summary))
    atomic_write_text(destination / "summary.md", _render_markdown(summary))
    return summary


def _format_accuracy(value: dict[str, Any] | None) -> str:
    return "" if value is None else f"{float(value['acc']):.2f}"


def _render_csv(summary: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    models = [model["slug"] for model in summary["models"]]
    labels = [model["label"] for model in summary["models"]]
    writer.writerow(["task", "N", *labels])
    for row in summary["rows"]:
        writer.writerow(
            [
                row["task"],
                row["N"],
                *[_format_accuracy(row["models"][model]) for model in models],
            ]
        )
    writer.writerow(
        [
            "mean",
            "",
            *[
                ""
                if summary["mean"][model] is None
                else f"{summary['mean'][model]:.2f}"
                for model in models
            ],
        ]
    )
    return buffer.getvalue()


def _render_markdown(summary: dict[str, Any]) -> str:
    models = [model["slug"] for model in summary["models"]]
    labels = [model["label"] for model in summary["models"]]
    lines = [
        "# Full benchmark summary",
        "",
        (
            f"Contract: greedy decoding, max_new_tokens={MAX_NEW_TOKENS}, "
            f"max_samples={summary['settings']['max_samples']}. "
            f"Completed cells: {summary['completed_cells']}/"
            f"{summary['expected_cells']}."
        ),
    ]
    seeded = {
        model: seed
        for model, seed in summary["settings"]["sample_seeds"].items()
        if seed is not None
    }
    if seeded:
        lines.extend(
            [
                "",
                "Latent-noise base seeds: "
                + ", ".join(f"{model}={seed}" for model, seed in seeded.items())
                + f" ({SAMPLE_SEED_SCHEME}).",
            ]
        )
    lines.extend(
        [
            "",
            "| task | N | " + " | ".join(labels) + " |",
            "|---|---:|" + "|".join("---:" for _ in labels) + "|",
        ]
    )
    for row in summary["rows"]:
        values = [
            "—"
            if row["models"][model] is None
            else f"{float(row['models'][model]['acc']):.2f}"
            for model in models
        ]
        lines.append(
            f"| {row['task']} | {row['N']} | " + " | ".join(values) + " |"
        )
    mean_values = [
        "—"
        if summary["mean"][model] is None
        else f"{summary['mean'][model]:.2f}"
        for model in models
    ]
    lines.append("| **mean** |  | " + " | ".join(mean_values) + " |")
    lines.append("")
    lines.append("Values are percentages. A mean is emitted only for a complete column.")
    return "\n".join(lines) + "\n"
