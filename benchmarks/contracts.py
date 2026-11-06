# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Fixed model, task, and decoding contracts for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar


TASK_SIZES: dict[str, int] = {
    "race": 4_887,
    "narrativeqa": 1_000,
    "hellaswag": 10_042,
    "story_cloze": 468,
    "lambada": 5_153,
    "arc_easy": 2_375,
    "truthfulqa": 817,
    "mmlu": 14_042,
}


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    backend: str
    default_source: str | None
    decoder_cfg: float | None = None
    dit_cfg: float | None = None
    dit_timestep_schedule: str | None = None
    default_batch_size: int = 8


MODEL_SPECS: dict[str, ModelSpec] = {
    "meta-llama-3-8b": ModelSpec(
        slug="meta-llama-3-8b",
        label="Meta-Llama-3-8B",
        backend="causal_lm",
        default_source="meta-llama/Meta-Llama-3-8B",
    ),
    "qwen2.5-7b": ModelSpec(
        slug="qwen2.5-7b",
        label="Qwen2.5-7B",
        backend="causal_lm",
        default_source="Qwen/Qwen2.5-7B",
    ),
    "qwen3-8b-base": ModelSpec(
        slug="qwen3-8b-base",
        label="Qwen3-8B-Base",
        backend="causal_lm",
        default_source="Qwen/Qwen3-8B-Base",
    ),
    "llada-8b-base": ModelSpec(
        slug="llada-8b-base",
        label="LLaDA-8B-Base",
        backend="llada",
        default_source="GSAI-ML/LLaDA-8B-Base",
        default_batch_size=4,
    ),
    "cola2": ModelSpec(
        slug="cola2",
        label="Cola2",
        backend="cola2",
        default_source=None,
        decoder_cfg=-0.9,
        dit_cfg=10.0,
        dit_timestep_schedule="low_noise",
        default_batch_size=1,
    ),
}

MAX_NEW_TOKENS = 32
DECODING = "greedy"
DEFAULT_MAX_PROMPT_LENGTH = 2_048
COLA2_DIT_STEPS = 8
COLA2_CHUNK_SPANS = 1
DEFAULT_SAMPLE_SEED = 42
SAMPLE_SEED_SCHEME = "sha256-v1"
CONTRACT_VERSION = 3

T = TypeVar("T")


def parse_names(raw: str | Iterable[str] | None, allowed: Sequence[str]) -> list[str]:
    """Parse a comma/space separated list while preserving contract order."""
    if raw is None:
        return list(allowed)
    if isinstance(raw, str):
        requested = [part for part in raw.replace(",", " ").split() if part]
    else:
        requested = list(raw)
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise ValueError(f"unknown values: {', '.join(unknown)}")
    requested_set = set(requested)
    return [name for name in allowed if name in requested_set]


def shard(items: Sequence[T], index: int, count: int) -> list[T]:
    if count < 1:
        raise ValueError("shard count must be at least 1")
    if index < 0 or index >= count:
        raise ValueError(f"shard index must be in [0, {count}), got {index}")
    return [item for position, item in enumerate(items) if position % count == index]


def expected_count(task: str, max_samples: int) -> int:
    if task not in TASK_SIZES:
        raise ValueError(f"unknown task: {task}")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    return TASK_SIZES[task] if max_samples == 0 else min(TASK_SIZES[task], max_samples)
