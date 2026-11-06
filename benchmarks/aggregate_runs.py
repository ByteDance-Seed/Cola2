# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Aggregate independent benchmark summaries into mean/std paper results."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from .contracts import CONTRACT_VERSION, MODEL_SPECS, SAMPLE_SEED_SCHEME
from .io_utils import atomic_write_json, atomic_write_text


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _round_std(values: list[float]) -> float:
    return round(statistics.pstdev(values), 4)


def aggregate_summaries(
    summary_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    model: str = "cola2",
) -> dict[str, Any]:
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown model: {model}")
    paths = [Path(path).expanduser().resolve() for path in summary_paths]
    if not paths:
        raise ValueError("at least one --summary path is required")
    if len(set(paths)) != len(paths):
        raise ValueError("summary paths must be unique")

    loaded = [_load_json_object(path) for path in paths]
    first = loaded[0]
    tasks = list(first.get("tasks") or [])
    if not tasks:
        raise ValueError(f"{paths[0]}: summary has no tasks")
    common_settings = {
        key: first.get("settings", {}).get(key)
        for key in ("max_samples", "max_new_tokens", "decoding")
    }

    labels = {
        item.get("slug"): item.get("label")
        for item in first.get("models", [])
        if isinstance(item, dict)
    }
    if model not in labels:
        raise ValueError(f"{paths[0]}: model {model!r} is absent")

    seeds: list[int] = []
    run_rows: list[dict[str, dict[str, Any]]] = []
    run_means: list[float] = []
    for path, summary in zip(paths, loaded):
        if summary.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(
                f"{path}: contract version differs from {CONTRACT_VERSION}"
            )
        if summary.get("complete") is not True:
            raise ValueError(f"{path}: summary is not complete")
        if list(summary.get("tasks") or []) != tasks:
            raise ValueError(f"{path}: task order differs from the first summary")
        settings = summary.get("settings")
        if not isinstance(settings, dict):
            raise ValueError(f"{path}: missing settings")
        observed_settings = {
            key: settings.get(key)
            for key in ("max_samples", "max_new_tokens", "decoding")
        }
        if observed_settings != common_settings:
            raise ValueError(f"{path}: benchmark settings differ")
        if settings.get("sample_seed_scheme") != SAMPLE_SEED_SCHEME:
            raise ValueError(f"{path}: unsupported sample seed scheme")
        sample_seeds = settings.get("sample_seeds")
        if not isinstance(sample_seeds, dict) or not isinstance(
            sample_seeds.get(model), int
        ):
            raise ValueError(f"{path}: missing integer sample seed for {model}")
        seed = int(sample_seeds[model])
        if seed in seeds:
            raise ValueError(f"{path}: duplicate sample seed {seed}")
        seeds.append(seed)

        indexed_rows: dict[str, dict[str, Any]] = {}
        for row in summary.get("rows", []):
            if isinstance(row, dict) and isinstance(row.get("task"), str):
                indexed_rows[row["task"]] = row
        model_rows: dict[str, dict[str, Any]] = {}
        for task in tasks:
            row = indexed_rows.get(task)
            value = (
                row.get("models", {}).get(model)
                if isinstance(row, dict)
                and isinstance(row.get("models"), dict)
                else None
            )
            if not isinstance(value, dict) or not isinstance(
                value.get("acc"), (int, float)
            ):
                raise ValueError(f"{path}: incomplete {model}/{task} cell")
            model_rows[task] = {
                "N": row.get("N"),
                "acc": float(value["acc"]),
            }
        run_rows.append(model_rows)

        mean_value = summary.get("mean", {}).get(model)
        if not isinstance(mean_value, (int, float)):
            raise ValueError(f"{path}: missing complete mean for {model}")
        run_means.append(
            sum(model_rows[task]["acc"] for task in tasks) / len(tasks)
        )

    rows: list[dict[str, Any]] = []
    task_means: list[float] = []
    for task in tasks:
        counts = {run[task]["N"] for run in run_rows}
        if len(counts) != 1:
            raise ValueError(f"task {task}: sample counts differ across runs")
        accuracies = [run[task]["acc"] for run in run_rows]
        task_mean = sum(accuracies) / len(accuracies)
        task_means.append(task_mean)
        rows.append(
            {
                "task": task,
                "N": next(iter(counts)),
                "runs": [
                    {"sample_seed": seed, "acc": accuracy}
                    for seed, accuracy in zip(seeds, accuracies)
                ],
                "mean": round(task_mean, 2),
                "std": _round_std(accuracies),
            }
        )

    aggregate = {
        "contract_version": CONTRACT_VERSION,
        "aggregation": "mean_over_independent_latent_noise_runs",
        "model": {"slug": model, "label": labels[model]},
        "run_count": len(paths),
        "sample_seed_scheme": SAMPLE_SEED_SCHEME,
        "sample_seeds": seeds,
        "settings": common_settings,
        "tasks": tasks,
        "rows": rows,
        "mean": {
            "runs": [
                {"sample_seed": seed, "acc": round(value, 2)}
                for seed, value in zip(seeds, run_means)
            ],
            "mean": round(sum(task_means) / len(task_means), 2),
            "std": _round_std(run_means),
        },
        "source_summaries": [str(path) for path in paths],
    }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "summary.json", aggregate)
    atomic_write_text(destination / "summary.csv", _render_csv(aggregate))
    atomic_write_text(destination / "summary.md", _render_markdown(aggregate))
    return aggregate


def _render_csv(aggregate: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    seeds = aggregate["sample_seeds"]
    writer.writerow(
        ["task", "N", *[f"seed_{seed}" for seed in seeds], "mean", "std"]
    )
    for row in aggregate["rows"]:
        writer.writerow(
            [
                row["task"],
                row["N"],
                *[f"{run['acc']:.2f}" for run in row["runs"]],
                f"{row['mean']:.2f}",
                f"{row['std']:.4f}",
            ]
        )
    writer.writerow(
        [
            "mean",
            "",
            *[f"{run['acc']:.2f}" for run in aggregate["mean"]["runs"]],
            f"{aggregate['mean']['mean']:.2f}",
            f"{aggregate['mean']['std']:.4f}",
        ]
    )
    return buffer.getvalue()


def _render_markdown(aggregate: dict[str, Any]) -> str:
    seeds = aggregate["sample_seeds"]
    label = aggregate["model"]["label"]
    lines = [
        f"# {label} repeated benchmark summary",
        "",
        (
            f"Mean and population standard deviation over "
            f"{aggregate['run_count']} independent latent-noise runs. "
            f"Seeds: {', '.join(str(seed) for seed in seeds)}."
        ),
        "",
        "| task | N | "
        + " | ".join(f"seed {seed}" for seed in seeds)
        + " | mean | std |",
        "|---|---:|" + "|".join("---:" for _ in range(len(seeds) + 2)) + "|",
    ]
    for row in aggregate["rows"]:
        run_values = " | ".join(f"{run['acc']:.2f}" for run in row["runs"])
        lines.append(
            f"| {row['task']} | {row['N']} | {run_values} | "
            f"{row['mean']:.2f} | {row['std']:.4f} |"
        )
    mean_values = " | ".join(
        f"{run['acc']:.2f}" for run in aggregate["mean"]["runs"]
    )
    lines.append(
        f"| **mean** |  | {mean_values} | "
        f"**{aggregate['mean']['mean']:.2f}** | "
        f"{aggregate['mean']['std']:.4f} |"
    )
    lines.extend(
        [
            "",
            "Values are percentages. Task means are averaged without sample-count weighting.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
        help="Path to one completed benchmark summary.json; repeat per run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_SPECS, default="cola2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = aggregate_summaries(
            args.summary,
            args.output_dir,
            model=args.model,
        )
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "run_count": result["run_count"],
                "sample_seeds": result["sample_seeds"],
                "mean": result["mean"]["mean"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
