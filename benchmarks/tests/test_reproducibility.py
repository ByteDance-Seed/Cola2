# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from benchmarks.aggregate_runs import aggregate_summaries
from benchmarks.contracts import CONTRACT_VERSION
from benchmarks.data import prepare_eval_data
from benchmarks.inference import Cola2Backend, InferenceOptions
from benchmarks.sample_rng import derive_sample_seed


def test_derived_sample_seed_is_stable_and_distinct() -> None:
    first = derive_sample_seed(
        base_seed=42,
        task="lambada",
        sample_id="example",
    )
    assert first == derive_sample_seed(
        base_seed=42,
        task="lambada",
        sample_id="example",
    )
    assert first != derive_sample_seed(
        base_seed=43,
        task="lambada",
        sample_id="example",
    )
    assert first != derive_sample_seed(
        base_seed=42,
        task="arc_easy",
        sample_id="example",
    )
    assert first != derive_sample_seed(
        base_seed=42,
        task="lambada",
        sample_id=1,
    )


def test_missing_external_data_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Hugging Face"):
        prepare_eval_data(
            tmp_path / "missing",
            tmp_path / "prepared",
            tasks=["lambada"],
            max_samples=1,
        )


class _Encoded(dict):
    def to(self, _device: str) -> "_Encoded":
        return self


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, _prompt: str, **_kwargs: object) -> _Encoded:
        return _Encoded(
            input_ids=torch.tensor([[11, 12]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        )

    def batch_decode(
        self, tokens: torch.Tensor, *, skip_special_tokens: bool
    ) -> list[str]:
        assert skip_special_tokens is True
        return [str(int(tokens[0, 0]))]


class _Model:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        generator: torch.Generator,
        **_kwargs: object,
    ) -> torch.Tensor:
        self.seeds.append(generator.initial_seed())
        suffix = torch.tensor([[99]], dtype=input_ids.dtype)
        return torch.cat([input_ids, suffix], dim=1)


def test_cola2_backend_uses_per_record_generator(tmp_path: Path) -> None:
    options = InferenceOptions(
        model="cola2",
        model_source="checkpoint",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        tasks=("lambada",),
        device="cpu",
        sample_seed=42,
    )
    backend = object.__new__(Cola2Backend)
    backend.options = options
    backend.torch = torch
    backend.tokenizer = _Tokenizer()
    backend.model = _Model()
    records = [
        {"id": "first", "task": "lambada", "prompt": "one"},
        {"id": "second", "task": "lambada", "prompt": "two"},
    ]

    assert backend.generate(records) == ["99", "99"]
    assert backend.model.seeds == [
        derive_sample_seed(
            base_seed=42,
            task="lambada",
            sample_id=record["id"],
        )
        for record in records
    ]


def _write_summary(
    path: Path,
    *,
    seed: int,
    lambada: float,
    arc_easy: float,
) -> None:
    value = {
        "contract_version": CONTRACT_VERSION,
        "complete": True,
        "settings": {
            "max_samples": 2,
            "max_new_tokens": 32,
            "decoding": "greedy",
            "sample_seeds": {"cola2": seed},
            "sample_seed_scheme": "sha256-v1",
        },
        "models": [{"slug": "cola2", "label": "Cola2"}],
        "tasks": ["lambada", "arc_easy"],
        "rows": [
            {
                "task": "lambada",
                "N": 2,
                "models": {"cola2": {"acc": lambada}},
            },
            {
                "task": "arc_easy",
                "N": 2,
                "models": {"cola2": {"acc": arc_easy}},
            },
        ],
        "mean": {"cola2": round((lambada + arc_easy) / 2, 2)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_three_run_aggregate_writes_all_formats(tmp_path: Path) -> None:
    summaries = []
    for seed, lambada, arc_easy in (
        (42, 50.0, 80.0),
        (43, 60.0, 70.0),
        (44, 70.0, 90.0),
    ):
        path = tmp_path / f"seed-{seed}" / "summary.json"
        _write_summary(
            path,
            seed=seed,
            lambada=lambada,
            arc_easy=arc_easy,
        )
        summaries.append(path)

    output = tmp_path / "aggregate"
    aggregate = aggregate_summaries(summaries, output)

    assert aggregate["run_count"] == 3
    assert aggregate["sample_seeds"] == [42, 43, 44]
    assert aggregate["rows"][0]["mean"] == 60.0
    assert aggregate["rows"][1]["mean"] == 80.0
    assert aggregate["mean"]["mean"] == 70.0
    assert (output / "summary.json").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "summary.md").is_file()


def test_aggregate_rejects_incomplete_summary(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, seed=42, lambada=50.0, arc_easy=80.0)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["complete"] = False
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="summary is not complete"):
        aggregate_summaries([path], tmp_path / "aggregate")


def test_aggregate_mean_uses_unrounded_task_means(tmp_path: Path) -> None:
    summaries = []
    for seed, lambada in ((42, 0.0), (43, 0.0), (44, 0.01)):
        path = tmp_path / f"seed-{seed}" / "summary.json"
        _write_summary(
            path,
            seed=seed,
            lambada=lambada,
            arc_easy=0.03,
        )
        summaries.append(path)

    aggregate = aggregate_summaries(summaries, tmp_path / "aggregate")

    assert aggregate["rows"][0]["mean"] == 0.0
    assert aggregate["rows"][1]["mean"] == 0.03
    assert aggregate["mean"]["mean"] == 0.02
