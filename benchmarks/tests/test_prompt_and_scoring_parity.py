# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.contracts import TASK_SIZES
from benchmarks.prompt_template import apply_prompt_template
from benchmarks.task_scoring import score_record


def _example_record(task: str) -> dict:
    record = {
        "context": "A short context.",
        "question": "What is the correct continuation?",
        "ground_truth": "correct",
        "choices": ["wrong", "correct", "other", "last"],
    }
    if task == "lambada":
        record["choices"] = None
    elif task == "story_cloze":
        record["choices"] = ["wrong ending", "correct ending"]
    elif task == "narrativeqa":
        record["choices"] = None
    elif task == "truthfulqa":
        record["ground_truth"] = ["correct"]
        record["choices"] = ["incorrect"]
    return record


def test_all_prompts_are_deterministic_and_nonempty() -> None:
    for task in TASK_SIZES:
        record = _example_record(task)
        gold = record.get("ground_truth", record.get("answer", ""))
        arguments = {
            "task": task,
            "context": record.get("context", "") or "",
            "question": record.get("question", "") or "",
            "answer": gold,
            "choices": record.get("choices"),
        }
        first = apply_prompt_template(**arguments)
        second = apply_prompt_template(**arguments)
        assert first == second
        assert isinstance(first, str) and first.strip()


def test_required_task_scoring_contract() -> None:
    cases = [
        ("race", "(C)", "third", ["first", "second", "third", "fourth"]),
        (
            "hellaswag",
            "the person opens the door",
            "the person opens the door.",
            ["sits down", "the person opens the door."],
        ),
        ("lambada", "Signs continue here", "signs", None),
        ("arc_easy", "Answer: B", "moon", ["sun", "moon", "mars", "venus"]),
        ("story_cloze", "B", "good ending", ["bad ending", "good ending"]),
        (
            "narrativeqa",
            "three days",
            ["3 days", "three days"],
            None,
        ),
        (
            "truthfulqa",
            "No, it cannot",
            ["No, it cannot", "That is impossible"],
            ["Yes, it can"],
        ),
        ("mmlu", "(A)", "alpha", ["alpha", "beta", "gamma", "delta"]),
    ]
    for task, generated, gold, choices in cases:
        result = score_record(task, generated, gold, choices)
        assert result["correct"] is True
