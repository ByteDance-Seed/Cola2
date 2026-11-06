#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CLI for one model shard of the resumable benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .contracts import (
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_SAMPLE_SEED,
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    TASK_SIZES,
    parse_names,
)
from .inference import InferenceOptions, run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_SPECS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tasks", default=" ".join(TASK_SIZES))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS, choices=[MAX_NEW_TOKENS]
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--llada-steps", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--llada-block-length", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--llada-mask-id", type=int, default=126_336)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=int(os.environ.get("COLA2_BENCHMARK_SEED", DEFAULT_SAMPLE_SEED)),
        help=(
            "Base seed for deterministic per-record Cola2 latent noise. "
            "Ignored by other backends."
        ),
    )
    parser.add_argument(
        "--external-command",
        default=os.environ.get("COLA2_BENCHMARK_COMMAND"),
        help=(
            "Optional Cola2 CLI adapter instead of the installed package. "
            "Required placeholders: {model_path}, {task}, {input}, {output}, "
            "{max_new_tokens}, {decoding}, {decoder_cfg}, {dit_cfg}, "
            "{dit_timestep_schedule}, {sample_seed}, {sample_seed_scheme}. "
            "The adapter must derive each record seed with "
            "benchmarks.sample_rng.derive_sample_seed."
        ),
    )
    parser.add_argument("--external-cwd", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = MODEL_SPECS[args.model]
    model_source = args.model_path or spec.default_source
    if not model_source:
        raise SystemExit(f"--model-path is required for {args.model}")
    tasks = tuple(parse_names(args.tasks, list(TASK_SIZES)))
    options = InferenceOptions(
        model=args.model,
        model_source=model_source,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tasks=tasks,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size or spec.default_batch_size,
        max_prompt_length=args.max_prompt_length,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        resume=args.resume,
        force=args.force,
        llada_steps=args.llada_steps,
        llada_block_length=args.llada_block_length,
        llada_mask_id=args.llada_mask_id,
        sample_seed=args.sample_seed,
        external_command=args.external_command,
        external_cwd=args.external_cwd,
    )
    result = run_inference(options)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
