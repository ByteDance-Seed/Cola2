# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Self-contained copy of the seedream generate-task scoring rules."""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import Counter
from typing import Any, Iterable, Sequence


MULTI_CHOICE_TASKS = frozenset(
    {
        "mmlu",
        "arc_easy",
        "obqa",
        "race",
        "siqa",
        "hellaswag",
        "story_cloze",
        "piqa",
        "cbt",
    }
)
EXTRACTIVE_QA_TASKS = frozenset(
    {"squad", "squadv2", "drop", "narrativeqa", "coqa", "quac"}
)
FIRST_WORD_TASKS = frozenset({"lambada"})
UNANSWERABLE = "unanswerable"
QUAC_UNANSWERABLE = "CANNOTANSWER"

_PUNCT_RE = re.compile(r"[^\w\s]")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_BARE_LETTER_RE = re.compile(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?\.?")
_KEYWORD_LETTER_RE = re.compile(
    r"(?i)\b(?:final\s+answer|answer|option|choice)\b\s*(?:is|=|:|：)?\s*"
    r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?(?=\s|$|[.,;:!?])"
)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = _PUNCT_RE.sub("", str(text).lower().strip())
    return " ".join(normalized.split())


def squad_normalize(text: Any) -> str:
    if text is None:
        return ""
    normalized = _ARTICLE_RE.sub(" ", str(text).lower().strip())
    normalized = _PUNCT_RE.sub("", normalized)
    return " ".join(normalized.split())


def similarity(text1: Any, text2: Any) -> float:
    normalized1, normalized2 = normalize_text(text1), normalize_text(text2)
    if not normalized1 and not normalized2:
        return 1.0
    return difflib.SequenceMatcher(None, normalized1, normalized2).ratio()


def first_word(text: Any) -> str:
    parts = normalize_text(text).split()
    return parts[0] if parts else ""


def token_f1(prediction: Any, gold: Any) -> float:
    prediction_tokens = squad_normalize(prediction).split()
    gold_tokens = squad_normalize(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _as_gold_list(gold: Any) -> list[str]:
    if isinstance(gold, (list, tuple)):
        return [str(value) for value in gold if str(value).strip()]
    return [str(gold)] if gold is not None else [""]


def extract_choice_letter(text: Any, number_of_choices: int) -> int | None:
    if text is None or number_of_choices <= 0:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    candidates: list[str] = []
    match = _BARE_LETTER_RE.fullmatch(raw)
    if match:
        candidates.append(match.group(1))
    candidates.extend(_KEYWORD_LETTER_RE.findall(raw))
    if len(raw) <= 40:
        candidates.extend(re.findall(r"[\(\[]\s*([A-Za-z])\s*[\)\]]", raw))
    for letter in reversed(candidates):
        index = ord(letter.upper()) - 65
        if 0 <= index < number_of_choices:
            return index
    return None


def predict_choice(
    text: Any, choices: Sequence[str]
) -> tuple[int | None, str, float]:
    if not choices:
        return None, "no_choices", 0.0
    number_of_choices = len(choices)
    letter_index = extract_choice_letter(text, number_of_choices)
    scores = [similarity(text, choice) for choice in choices]
    best_index = max(range(number_of_choices), key=lambda index: scores[index])
    fidelity = scores[best_index]
    if letter_index is not None:
        return letter_index, "letter", fidelity

    normalized = normalize_text(text)
    if normalized:
        exact = [
            index
            for index, choice in enumerate(choices)
            if normalize_text(choice) == normalized
        ]
        if len(exact) == 1:
            return exact[0], "exact", fidelity
        contained = [
            index
            for index, choice in enumerate(choices)
            if normalize_text(choice) and normalize_text(choice) in normalized
        ]
        if len(contained) == 1:
            return contained[0], "contains", fidelity
    return best_index, "nearest", fidelity


def gold_choice_index(gold: Any, choices: Sequence[str]) -> int | None:
    if not choices:
        return None
    index = extract_choice_letter(gold, len(choices))
    if index is not None:
        return index
    normalized = normalize_text(gold)
    for index, choice in enumerate(choices):
        if normalize_text(choice) == normalized:
            return index
    return max(range(len(choices)), key=lambda position: similarity(gold, choices[position]))


def score_record(
    task: str,
    generate: Any,
    gold: Any,
    choices: Sequence[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"correct": False, "chance": None}

    if task in MULTI_CHOICE_TASKS:
        task_choices = list(choices or [])
        if not task_choices:
            result.update(
                correct=similarity(generate, gold) >= 1.0,
                match="degraded_no_choices",
                fidelity=None,
            )
            return result
        predicted_index, method, fidelity = predict_choice(generate, task_choices)
        gold_index = gold_choice_index(gold, task_choices)
        result.update(
            correct=(
                predicted_index is not None
                and gold_index is not None
                and predicted_index == gold_index
            ),
            pred_index=predicted_index,
            gold_index=gold_index,
            match=method,
            fidelity=round(fidelity, 4),
            chance=1.0 / len(task_choices),
        )
        return result

    if task in EXTRACTIVE_QA_TASKS:
        empty_gold = QUAC_UNANSWERABLE if task == "quac" else UNANSWERABLE
        golds = _as_gold_list(gold) or [empty_gold]
        normalized_prediction = squad_normalize(generate)
        result.update(
            correct=any(
                normalized_prediction == squad_normalize(candidate) for candidate in golds
            ),
            f1=round(max(token_f1(generate, candidate) for candidate in golds), 4),
            chance=0.0,
        )
        return result

    if task == "truthfulqa":
        golds = _as_gold_list(gold)
        options = sorted(
            {*golds, *(str(choice) for choice in (choices or []))},
            key=lambda option: hashlib.md5(option.encode("utf-8")).hexdigest(),
        )
        if not options:
            result.update(correct=False, match="no_choices", fidelity=None)
            return result
        gold_set = {squad_normalize(candidate) for candidate in golds}
        gold_indices = {
            index
            for index, option in enumerate(options)
            if squad_normalize(option) in gold_set
        }
        predicted_index, method, fidelity = predict_choice(generate, options)
        result.update(
            correct=(
                predicted_index is not None and predicted_index in gold_indices
            ),
            pred_index=predicted_index,
            match=method,
            fidelity=round(fidelity, 4),
            chance=len(gold_indices) / len(options) if gold_indices else 0.0,
        )
        return result

    if task in FIRST_WORD_TASKS:
        result.update(
            correct=first_word(generate) == first_word(gold),
            chance=0.0,
        )
        return result

    result.update(correct=similarity(generate, gold) >= 1.0, chance=0.0)
    return result


def score_records(
    task: str, records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    total = 0
    correct = 0
    chances: list[float] = []
    fidelities: list[float] = []
    f1_values: list[float] = []
    degraded = 0
    for record in records:
        gold = record.get("ground_truth", record.get("answer", ""))
        score = score_record(
            task,
            record.get("generate", ""),
            gold,
            record.get("choices"),
        )
        record["scoring"] = {
            key: value for key, value in score.items() if value is not None
        }
        total += 1
        correct += int(score["correct"])
        if score.get("chance") is not None:
            chances.append(score["chance"])
        if score.get("fidelity") is not None:
            fidelities.append(score["fidelity"])
        if score.get("f1") is not None:
            f1_values.append(score["f1"])
        if score.get("match") == "degraded_no_choices":
            degraded += 1

    stats: dict[str, Any] = {
        "task": task,
        "total": total,
        "correct": correct,
        "acc": round(100.0 * correct / total, 2) if total else None,
    }
    if chances:
        stats["chance"] = round(100.0 * sum(chances) / len(chances), 2)
    if fidelities:
        stats["fidelity"] = round(
            100.0 * sum(fidelities) / len(fidelities), 2
        )
    if f1_values:
        stats["f1"] = round(100.0 * sum(f1_values) / len(f1_values), 2)
    if degraded:
        stats["degraded_no_choices"] = degraded
    return stats
