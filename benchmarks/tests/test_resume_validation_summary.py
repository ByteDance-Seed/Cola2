# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks import orchestrate
from benchmarks.contracts import expected_count, shard
from benchmarks.data import _disambiguate_duplicate_ids, prepare_record
from benchmarks.inference import InferenceOptions, run_inference
from benchmarks.io_utils import atomic_write_jsonl, read_jsonl
from benchmarks.scoring import score_cell
from benchmarks.summary import build_summary
from benchmarks.validation import (
    PREDICTIONS_NAME,
    cell_dir,
    validate_completion,
)


def _story_record(identifier: int, correct: str) -> dict:
    return {
        "id": identifier,
        "task": "story_cloze",
        "question": f"Story {identifier}",
        "prompt": f"Story: Story {identifier}\n(A) bad\n(B) {correct}\nEnd:",
        "choices": ["bad", correct],
        "ground_truth": correct,
    }


def _fake_external_script(path: Path) -> None:
    path.write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model-path")
parser.add_argument("--task")
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--max-new-tokens")
parser.add_argument("--decoding")
parser.add_argument("--decoder-cfg")
parser.add_argument("--dit-cfg")
parser.add_argument("--dit-timestep-schedule")
parser.add_argument("--sample-seed")
parser.add_argument("--sample-seed-scheme")
args = parser.parse_args()
with open(args.input, encoding="utf-8") as source, open(args.output, "w", encoding="utf-8") as output:
    for line in source:
        record = json.loads(line)
        output.write(json.dumps({"id": record["id"], "generate": record["ground_truth"]}) + "\\n")
""".lstrip(),
        encoding="utf-8",
    )


def test_external_runner_recovers_partial_jsonl_and_completes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    records = [_story_record(1, "good one"), _story_record(2, "good two")]
    atomic_write_jsonl(data_dir / "story_cloze.jsonl", records)
    fake = tmp_path / "fake_generate.py"
    _fake_external_script(fake)
    command = (
        f"{{python}} {shlex.quote(str(fake))} "
        "--model-path {model_path} --task {task} --input {input} --output {output} "
        "--max-new-tokens {max_new_tokens} --decoding {decoding} "
        "--decoder-cfg {decoder_cfg} --dit-cfg {dit_cfg} "
        "--dit-timestep-schedule {dit_timestep_schedule} "
        "--sample-seed {sample_seed} "
        "--sample-seed-scheme {sample_seed_scheme}"
    )
    options = InferenceOptions(
        model="cola2",
        model_source="checkpoint",
        data_dir=data_dir,
        output_dir=output_dir,
        tasks=("story_cloze",),
        max_samples=2,
        external_command=command,
        external_cwd=tmp_path,
    )

    bad_command = command.replace(
        " --sample-seed-scheme {sample_seed_scheme}",
        "",
    )
    with pytest.raises(ValueError, match="sample_seed_scheme"):
        run_inference(
            replace(
                options,
                output_dir=tmp_path / "bad-outputs",
                external_command=bad_command,
            )
        )

    run_inference(options)
    predictions_path = (
        cell_dir(output_dir, options.model, "story_cloze") / PREDICTIONS_NAME
    )
    first = read_jsonl(predictions_path)[0]
    atomic_write_jsonl(predictions_path, [first])
    with predictions_path.open("ab") as handle:
        handle.write(b'{"id":')

    run_inference(options)
    predictions = read_jsonl(predictions_path)
    assert [record["id"] for record in predictions] == [1, 2]

    metrics = score_cell(
        data_dir,
        output_dir,
        model=options.model,
        task="story_cloze",
        max_samples=2,
    )
    assert metrics["acc"] == 100.0
    marker = validate_completion(
        data_dir,
        output_dir,
        model=options.model,
        task="story_cloze",
        max_samples=2,
    )
    assert marker["count"] == 2

    summary = build_summary(
        data_dir,
        output_dir,
        tmp_path / "summary",
        models=[options.model],
        tasks=["story_cloze"],
        max_samples=2,
    )
    assert summary["complete"] is True
    assert (tmp_path / "summary" / "summary.csv").is_file()
    assert (tmp_path / "summary" / "summary.json").is_file()
    assert (tmp_path / "summary" / "summary.md").is_file()

    with pytest.raises(ValueError, match="configuration changed"):
        run_inference(replace(options, sample_seed=43))


def test_truthfulqa_preparation_keeps_incorrect_choices() -> None:
    source = {
        "id": "tqa",
        "context": "",
        "question": "Can this happen?",
        "answer": ["No"],
        "choices": ["Yes"],
    }
    prepared = prepare_record("truthfulqa", source)
    assert prepared["choices"] == ["Yes"]
    assert prepared["ground_truth"] == ["No"]
    assert "(A)" in prepared["prompt"]


def test_full_counts_and_stable_shards() -> None:
    assert expected_count("mmlu", 0) == 14_042
    assert expected_count("mmlu", 200) == 200
    assert shard(["a", "b", "c", "d"], 0, 2) == ["a", "c"]
    assert shard(["a", "b", "c", "d"], 1, 2) == ["b", "d"]


def test_duplicate_source_ids_are_stably_disambiguated() -> None:
    source = [{"id": "same", "question": "one"}, {"id": "same", "question": "two"}]
    prepared, duplicate_records = _disambiguate_duplicate_ids("narrativeqa", source)
    assert duplicate_records == 2
    assert prepared[0]["id"] != prepared[1]["id"]
    assert prepared[0]["source_id"] == prepared[1]["source_id"] == "same"


def test_orchestrator_does_not_score_after_inference_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_tasks: list[str] = []
    scored_tasks: list[str] = []

    def fake_load_task_data(
        _data_dir: Path,
        task: str,
        *,
        max_samples: int,
    ) -> list[dict]:
        assert max_samples == 0
        loaded_tasks.append(task)
        return []

    def fail_inference(_options: InferenceOptions) -> None:
        raise RuntimeError("synthetic inference failure")

    def unexpected_score(*_args: object, **kwargs: object) -> dict:
        scored_tasks.append(str(kwargs["task"]))
        return {"total": 1, "acc": 100.0}

    monkeypatch.setattr(orchestrate, "load_task_data", fake_load_task_data)
    monkeypatch.setattr(
        orchestrate,
        "is_cell_complete",
        lambda *_args, **_kwargs: (False, "not complete"),
    )
    monkeypatch.setattr(orchestrate, "run_inference", fail_inference)
    monkeypatch.setattr(orchestrate, "score_cell", unexpected_score)
    monkeypatch.setattr(
        orchestrate,
        "validate_predictions",
        lambda *_args, **_kwargs: scored_tasks.append("validated"),
    )
    monkeypatch.setattr(
        orchestrate,
        "build_summary",
        lambda *_args, **_kwargs: {
            "completed_cells": 0,
            "expected_cells": 1,
        },
    )
    args = orchestrate.build_parser().parse_args(
        [
            "--models",
            "cola2",
            "--tasks",
            "lambada",
            "--work-dir",
            str(tmp_path),
            "--model-path",
            "cola2=checkpoint",
            "--skip-prepare",
        ]
    )

    result = orchestrate.run(args)

    assert loaded_tasks == ["lambada"]
    assert scored_tasks == []
    assert result["ok"] is False
    assert result["completed"] == []
    assert result["errors"][0]["error"] == "inference: synthetic inference failure"
