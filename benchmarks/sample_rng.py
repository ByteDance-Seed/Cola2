# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Stable per-record random seeds for stochastic Cola2 latent generation."""

from __future__ import annotations

import hashlib
from typing import Any

from .contracts import SAMPLE_SEED_SCHEME
from .io_utils import canonical_id


MAX_SAMPLE_SEED = 2**63 - 1


def validate_base_seed(value: int) -> int:
    seed = int(value)
    if seed < 0 or seed >= MAX_SAMPLE_SEED:
        raise ValueError(
            f"sample seed must be in [0, {MAX_SAMPLE_SEED}), got {seed}"
        )
    return seed


def derive_sample_seed(*, base_seed: int, task: str, sample_id: Any) -> int:
    """Derive a stable torch seed independent of execution order and sharding."""
    seed = validate_base_seed(base_seed)
    if not task:
        raise ValueError("task must be non-empty when deriving a sample seed")
    payload = (
        f"{SAMPLE_SEED_SCHEME}\n{seed}\n{task}\n{canonical_id(sample_id)}"
    ).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return derived % MAX_SAMPLE_SEED
