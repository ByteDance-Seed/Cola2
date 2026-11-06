# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Interactive command-line generation for Cola2."""

from __future__ import annotations

import argparse
import copy
import os

import torch
from transformers import AutoModel, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a Cola2 checkpoint.")
    parser.add_argument(
        "--model",
        default=os.environ.get("COLA2_MODEL_PATH"),
        help="Local model directory or Hugging Face model ID.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum response length.",
    )
    parser.add_argument("--dit-steps", type=int, default=8)
    parser.add_argument(
        "--dit-timestep-schedule",
        choices=("uniform", "high_noise", "low_noise", "power"),
        default="low_noise",
    )
    parser.add_argument("--dit-cfg", type=float, default=10.0)
    parser.add_argument("--decoder-cfg", type=float, default=-0.9)
    parser.add_argument("--decoder-chunk-spans", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system", default="")
    parser.add_argument("--prompt", default=None, help="Generate once and exit.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _render_history(system: str, history: list[tuple[str, str]], user: str) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"System: {system.strip()}")
    for previous_user, previous_assistant in history:
        parts.append(f"User: {previous_user.strip()}")
        parts.append(f"Assistant: {previous_assistant.strip()}")
    parts.append(f"User: {user.strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _load(args: argparse.Namespace):
    if not args.model:
        raise SystemExit("Set --model or COLA2_MODEL_PATH.")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device).eval()
    return model, tokenizer


def _generate(
    model,
    tokenizer,
    *,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_context_tokens,
    )
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.dit_num_steps = args.dit_steps
    generation_config.dit_timestep_schedule = args.dit_timestep_schedule
    generation_config.dit_guidance_scale = args.dit_cfg
    generation_config.decoder_guidance_scale = args.decoder_cfg
    generation_config.span_causal_chunk_size = args.decoder_chunk_spans
    generation_config.repetition_penalty = args.repetition_penalty
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            generation_config=generation_config,
            generator=generator,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.max_new_tokens <= 0
        or args.max_context_tokens <= 0
        or args.dit_steps <= 0
        or args.decoder_chunk_spans <= 0
        or args.repetition_penalty <= 0
    ):
        raise SystemExit(
            "Token limits, DiT steps, chunk spans, and repetition penalty "
            "must be positive."
        )
    model, tokenizer = _load(args)
    history: list[tuple[str, str]] = []

    if args.prompt is not None:
        text = _render_history(args.system, history, args.prompt)
        print(_generate(model, tokenizer, prompt=text, args=args))
        return 0

    print("Cola2 interactive session. Commands: /reset, /quit")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"/quit", "/exit", "/q"}:
            break
        if user.lower() == "/reset":
            history.clear()
            print("History cleared.")
            continue
        prompt = _render_history(args.system, history, user)
        response = _generate(model, tokenizer, prompt=prompt, args=args)
        print(f"Cola2> {response}")
        history.append((user, response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
