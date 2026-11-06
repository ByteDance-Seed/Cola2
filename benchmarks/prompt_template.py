# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Benchmark prompt templates.

These eight branches are intentionally kept byte-for-byte equivalent to the
corresponding branches in seedream's generate-task prompt template. Keeping a
local copy makes the public benchmark independent of the training repository.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence


def _require_choices(task: str, choices: Sequence[str] | None, minimum: int = 1) -> list[str]:
    result = list(choices or [])
    if len(result) < minimum:
        raise ValueError(f"{task} requires at least {minimum} choices")
    return result


def apply_prompt_template(
    task: str,
    context: str,
    question: str,
    answer: Any,
    choices: Sequence[str] | None,
) -> str:
    """Render the fixed few-shot prompt for one prepared example."""
    if task == "lambada":
        return question

    if task == "arc_easy":
        current_choices_text = "\n".join(
            f"({chr(65 + i)}) {choice}"
            for i, choice in enumerate(_require_choices(task, choices))
        )
        return (
            "Question: Which object in the sky acts as a natural satellite to Earth?\n"
            "(A) The Sun\n"
            "(B) The Moon\n"
            "(C) Mars\n"
            "(D) Venus\n"
            "Answer: The Moon\n\n"
            "Question: What state of matter is an ice cube?\n"
            "(A) Gas\n"
            "(B) Liquid\n"
            "(C) Solid\n"
            "(D) Plasma\n"
            "Answer: Solid\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )

    if task == "hellaswag":
        current_choices_text = "\n".join(
            f"({chr(65 + i)}) {choice}"
            for i, choice in enumerate(_require_choices(task, choices))
        )
        return (
            "Context: The girl puts the bread into the toaster and pushes the lever down. The bread\n"
            "(A) becomes a slice of pizza.\n"
            "(B) starts to toast and turn brown.\n"
            "(C) disappears immediately.\n"
            "(D) turns into a glass of water.\n"
            "Answer: starts to toast and turn brown.\n\n"
            "Context: The goalkeeper sees the ball coming towards the net. He dives and\n"
            "(A) catches the ball with his hands.\n"
            "(B) starts dancing in the field.\n"
            "(C) opens a laptop to check email.\n"
            "(D) runs away from the stadium.\n"
            "Answer: catches the ball with his hands.\n\n"
            f"Context: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )

    if task == "mmlu":
        current_choices_text = "\n".join(
            f"({chr(65 + i)}) {choice}"
            for i, choice in enumerate(_require_choices(task, choices))
        )
        return (
            "Question: Which gas do plants absorb from the air during photosynthesis?\n"
            "(A) Oxygen\n"
            "(B) Carbon dioxide\n"
            "(C) Nitrogen\n"
            "(D) Hydrogen\n"
            "Answer: Carbon dioxide\n\n"
            "Question: A triangle has angles 50 degrees and 60 degrees. What is the third angle?\n"
            "(A) 60 degrees\n"
            "(B) 70 degrees\n"
            "(C) 80 degrees\n"
            "(D) 90 degrees\n"
            "Answer: 70 degrees\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )

    if task == "race":
        current_choices_text = "\n".join(
            f"({chr(65 + i)}) {choice}"
            for i, choice in enumerate(_require_choices(task, choices))
        )
        return (
            "Read the following article and answer the question.\n\n"
            "Article: Mary went to the store to buy some fruits. She bought five apples and two oranges. She paid 5 dollars in total.\n"
            "Question: What did Mary buy?\n"
            "Options:\n"
            "(A) Bananas\n"
            "(B) Apples and oranges\n"
            "(C) Grapes\n"
            "(D) Watermelon\n"
            "Answer: Apples and oranges\n\n"
            f"Article: {context}\n"
            f"Question: {question}\n"
            f"Options:\n{current_choices_text}\n"
            "Answer:"
        )

    if task == "story_cloze":
        task_choices = _require_choices(task, choices, minimum=2)
        current_choices_text = f"(A) {task_choices[0]}\n(B) {task_choices[1]}"
        return (
            "Story: I wanted to make an omelet. I cracked two eggs into a bowl and whisked them. Then I poured them into a hot pan.\n"
            "(A) I ate a delicious omelet for breakfast.\n"
            "(B) I decided to order a pizza instead.\n"
            "End: I ate a delicious omelet for breakfast.\n\n"
            "Story: The runner tied his shoes tight. He sprinted as fast as he could during the race. He crossed the finish line first.\n"
            "(A) He was sad that he lost the race.\n"
            "(B) He won the gold medal.\n"
            "End: He won the gold medal.\n\n"
            f"Story: {question}\n"
            f"{current_choices_text}\n"
            "End:"
        )

    if task == "narrativeqa":
        return (
            "Story: Sarah moves to a small coastal town to take over her late aunt's bookshop. The shop is deep in debt, and the landlord, Mr. Hale, threatens to evict her unless she pays the overdue rent within a month.\n"
            "Question: Who threatens to evict Sarah?\n"
            "Answer: Mr. Hale\n\n"
            "Story: After the storm sinks their fishing boat, the two brothers drift on a piece of wreckage for three days before a passing trawler finally spots them.\n"
            "Question: How long do the brothers drift before being rescued?\n"
            "Answer: three days\n\n"
            f"Story: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if task == "truthfulqa":
        options = sorted(
            {str(option) for option in list(answer or []) + list(choices or [])},
            key=lambda option: hashlib.md5(option.encode("utf-8")).hexdigest(),
        )
        if not options:
            raise ValueError("truthfulqa requires correct and incorrect choices")
        current_choices_text = "\n".join(
            f"({chr(65 + i)}) {option}" for i, option in enumerate(options)
        )
        return (
            "Question: What happens if you swallow watermelon seeds?\n"
            "(A) The seeds pass through your digestive system.\n"
            "(B) You grow watermelons in your stomach.\n"
            "Answer: The seeds pass through your digestive system.\n\n"
            "Question: What happens if you go outdoors in cold weather with wet hair?\n"
            "(A) You will catch a cold.\n"
            "(B) Nothing in particular happens.\n"
            "Answer: Nothing in particular happens.\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )

    raise ValueError(f"unsupported benchmark task: {task}")
