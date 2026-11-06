# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Resumable inference adapters for the five benchmark columns."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO

from .contracts import (
    COLA2_CHUNK_SPANS,
    COLA2_DIT_STEPS,
    CONTRACT_VERSION,
    DECODING,
    DEFAULT_SAMPLE_SEED,
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    SAMPLE_SEED_SCHEME,
)
from .data import load_task_data
from .io_utils import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_id,
    iter_batches,
    read_jsonl,
    read_jsonl_recover_tail,
    unique_by_id,
)
from .sample_rng import derive_sample_seed, validate_base_seed
from .validation import (
    COMPLETE_NAME,
    METRICS_NAME,
    PREDICTIONS_NAME,
    RUN_CONFIG_NAME,
    SCORED_NAME,
    cell_dir,
)


@dataclass(frozen=True)
class InferenceOptions:
    model: str
    model_source: str
    data_dir: Path
    output_dir: Path
    tasks: tuple[str, ...]
    max_samples: int = 0
    max_new_tokens: int = MAX_NEW_TOKENS
    batch_size: int = 8
    max_prompt_length: int = 2_048
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = True
    resume: bool = True
    force: bool = False
    llada_steps: int = MAX_NEW_TOKENS
    llada_block_length: int = MAX_NEW_TOKENS
    llada_mask_id: int = 126_336
    sample_seed: int = DEFAULT_SAMPLE_SEED
    external_command: str | None = None
    external_cwd: Path | None = None


@dataclass
class TaskState:
    task: str
    inputs: list[dict[str, Any]]
    existing: dict[str, dict[str, Any]]
    pending: list[dict[str, Any]]
    directory: Path
    predictions_path: Path
    lock: IO[str]


def _base_prediction(
    model: str, source: dict[str, Any], generated: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": source["id"],
        "model": model,
        "task": source["task"],
        "prompt": source["prompt"],
        "generate": generated,
        "ground_truth": source["ground_truth"],
    }
    if "choices" in source:
        result["choices"] = source["choices"]
    if "source_id" in source:
        result["source_id"] = source["source_id"]
    return result


def _run_config(options: InferenceOptions, task: str) -> dict[str, Any]:
    spec = MODEL_SPECS[options.model]
    effective_backend = (
        "external"
        if spec.backend == "cola2" and options.external_command
        else spec.backend
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "model": options.model,
        "model_label": spec.label,
        "backend": effective_backend,
        "model_source": options.model_source,
        "task": task,
        "decoding": DECODING,
        "max_new_tokens": options.max_new_tokens,
        "max_prompt_length": options.max_prompt_length,
        "dtype": options.dtype,
        "device": options.device,
        "trust_remote_code": options.trust_remote_code,
        "llada_steps": options.llada_steps if spec.backend == "llada" else None,
        "llada_block_length": (
            options.llada_block_length if spec.backend == "llada" else None
        ),
        "llada_mask_id": options.llada_mask_id if spec.backend == "llada" else None,
        "decoder_cfg": spec.decoder_cfg,
        "dit_cfg": spec.dit_cfg,
        "dit_steps": COLA2_DIT_STEPS if spec.backend == "cola2" else None,
        "dit_timestep_schedule": (
            spec.dit_timestep_schedule if spec.backend == "cola2" else None
        ),
        "span_causal_chunk_size": (
            COLA2_CHUNK_SPANS if spec.backend == "cola2" else None
        ),
        "dynamic_duration": True if spec.backend == "cola2" else None,
        "latent_feedback": True if spec.backend == "cola2" else None,
        "sample_seed": (
            validate_base_seed(options.sample_seed)
            if spec.backend == "cola2"
            else None
        ),
        "sample_seed_scheme": (
            SAMPLE_SEED_SCHEME if spec.backend == "cola2" else None
        ),
        "external_command": (
            options.external_command
            if spec.backend == "cola2" and options.external_command
            else None
        ),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate_existing(
    options: InferenceOptions,
    task: str,
    inputs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    predictions_path: Path,
) -> dict[str, dict[str, Any]]:
    input_index = unique_by_id(inputs, source=f"input:{task}")
    existing = unique_by_id(predictions, source=str(predictions_path))
    extra = [key for key in existing if key not in input_index]
    if extra:
        raise ValueError(
            f"{predictions_path}: {len(extra)} existing IDs are outside the "
            "current sample selection; use --force"
        )
    for key, prediction in existing.items():
        source = input_index[key]
        if prediction.get("model") != options.model:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} has a different model"
            )
        if prediction.get("task") != task:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} has a different task"
            )
        if not isinstance(prediction.get("generate"), str):
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} has no string generate"
            )
        for field in ("prompt", "ground_truth"):
            if prediction.get(field) != source.get(field):
                raise ValueError(
                    f"{predictions_path}: id={source['id']!r} changed {field}; "
                    "use --force"
                )
        if "choices" in source and prediction.get("choices") != source["choices"]:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} changed choices; "
                "use --force"
            )
        if "source_id" in source and prediction.get("source_id") != source["source_id"]:
            raise ValueError(
                f"{predictions_path}: id={source['id']!r} changed source_id; "
                "use --force"
            )
    return existing


def _prepare_task(options: InferenceOptions, task: str) -> TaskState:
    import fcntl

    model_directory = Path(options.output_dir) / options.model
    model_directory.mkdir(parents=True, exist_ok=True)
    lock = (model_directory / f".{task}.lock").open("a+", encoding="utf-8")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    directory = cell_dir(options.output_dir, options.model, task)
    try:
        if options.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        predictions_path = directory / PREDICTIONS_NAME
        config_path = directory / RUN_CONFIG_NAME
        expected_config = _run_config(options, task)

        if config_path.is_file():
            actual_config = _load_json_object(config_path)
            if actual_config != expected_config and predictions_path.exists():
                raise ValueError(
                    f"{config_path}: inference configuration changed; use --force"
                )
        if predictions_path.exists() and not options.resume and not options.force:
            raise ValueError(
                f"{predictions_path} already exists; enable resume or use --force"
            )
        atomic_write_json(config_path, expected_config)

        inputs = load_task_data(
            options.data_dir, task, max_samples=options.max_samples
        )
        predictions = read_jsonl_recover_tail(predictions_path)
        existing = _validate_existing(
            options, task, inputs, predictions, predictions_path
        )
        pending = [
            record for record in inputs if canonical_id(record["id"]) not in existing
        ]
        if pending:
            for stale_name in (SCORED_NAME, METRICS_NAME, COMPLETE_NAME):
                (directory / stale_name).unlink(missing_ok=True)
        return TaskState(
            task=task,
            inputs=inputs,
            existing=existing,
            pending=pending,
            directory=directory,
            predictions_path=predictions_path,
            lock=lock,
        )
    except BaseException:
        lock.close()
        raise


def _finalize_task(state: TaskState) -> None:
    predictions = read_jsonl(state.predictions_path)
    indexed = unique_by_id(predictions, source=str(state.predictions_path))
    ordered = [indexed[canonical_id(source["id"])] for source in state.inputs]
    atomic_write_jsonl(state.predictions_path, ordered)


class CausalLMBackend:
    def __init__(self, options: InferenceOptions):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.options = options
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[options.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            options.model_source,
            trust_remote_code=options.trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither a pad token nor an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            options.model_source,
            trust_remote_code=options.trust_remote_code,
            torch_dtype=dtype,
        )
        self.model.to(options.device)
        self.model.eval()

    def generate(self, records: list[dict[str, Any]]) -> list[str]:
        encoded = self.tokenizer(
            [record["prompt"] for record in records],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.options.max_prompt_length,
        ).to(self.options.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output[:, encoded["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


class Cola2Backend(CausalLMBackend):
    def __init__(self, options: InferenceOptions):
        try:
            import cola2  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "the Cola2 package is unavailable; install Cola2 or pass "
                "--external-command"
            ) from error
        super().__init__(options)

    def _generate_record(self, record: dict[str, Any]) -> str:
        spec = MODEL_SPECS[self.options.model]
        encoded = self.tokenizer(
            record["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=self.options.max_prompt_length,
        ).to(self.options.device)
        sample_seed = derive_sample_seed(
            base_seed=self.options.sample_seed,
            task=str(record["task"]),
            sample_id=record["id"],
        )
        generator = self.torch.Generator(
            device=self.options.device
        ).manual_seed(sample_seed)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
                dit_num_steps=COLA2_DIT_STEPS,
                dit_timestep_schedule=spec.dit_timestep_schedule,
                dit_guidance_scale=spec.dit_cfg,
                decoder_guidance_scale=spec.decoder_cfg,
                span_causal_chunk_size=COLA2_CHUNK_SPANS,
                dynamic_duration=True,
                latent_feedback=True,
                pad_token_id=self.tokenizer.pad_token_id,
                generator=generator,
            )
        new_tokens = output[:, encoded["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

    def generate(self, records: list[dict[str, Any]]) -> list[str]:
        return [self._generate_record(record) for record in records]


class LLaDABackend:
    def __init__(self, options: InferenceOptions):
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.functional = functional
        self.options = options
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[options.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            options.model_source,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither a pad token nor an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            options.model_source,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(options.device)
        self.model.eval()

    def _transfer_counts(self, mask_index: Any, steps: int) -> Any:
        mask_number = mask_index.sum(dim=1, keepdim=True)
        base = mask_number // steps
        remainder = mask_number % steps
        counts = (
            self.torch.zeros(
                mask_number.size(0),
                steps,
                device=mask_index.device,
                dtype=self.torch.int64,
            )
            + base
        )
        for row in range(mask_number.size(0)):
            counts[row, : remainder[row]] += 1
        return counts

    def _generate_tokens(self, prompt: Any, attention_mask: Any) -> Any:
        steps = self.options.llada_steps
        generation_length = self.options.max_new_tokens
        block_length = self.options.llada_block_length
        if generation_length % block_length != 0:
            raise ValueError("LLaDA generation length must be divisible by block length")
        number_of_blocks = generation_length // block_length
        if steps % number_of_blocks != 0:
            raise ValueError("LLaDA steps must be divisible by number of blocks")

        batch_size, prompt_length = prompt.shape
        tokens = self.torch.full(
            (batch_size, prompt_length + generation_length),
            self.options.llada_mask_id,
            dtype=self.torch.long,
            device=prompt.device,
        )
        tokens[:, :prompt_length] = prompt
        attention_mask = self.torch.cat(
            [
                attention_mask,
                self.torch.ones(
                    (batch_size, generation_length),
                    dtype=attention_mask.dtype,
                    device=prompt.device,
                ),
            ],
            dim=-1,
        )
        steps_per_block = steps // number_of_blocks
        for block in range(number_of_blocks):
            low = prompt_length + block * block_length
            high = prompt_length + (block + 1) * block_length
            transfer_counts = self._transfer_counts(
                tokens[:, low:high] == self.options.llada_mask_id,
                steps_per_block,
            )
            for step in range(steps_per_block):
                mask_index = tokens == self.options.llada_mask_id
                logits = self.model(tokens, attention_mask=attention_mask).logits
                predicted = self.torch.argmax(logits, dim=-1)
                probabilities = self.functional.softmax(logits.float(), dim=-1)
                confidence = self.torch.gather(
                    probabilities, dim=-1, index=predicted.unsqueeze(-1)
                ).squeeze(-1)
                confidence[:, high:] = -self.torch.inf
                predicted = self.torch.where(mask_index, predicted, tokens)
                confidence = self.torch.where(
                    mask_index,
                    confidence,
                    self.torch.full_like(confidence, -self.torch.inf),
                )
                transfer = self.torch.zeros_like(predicted, dtype=self.torch.bool)
                for row in range(batch_size):
                    count = int(transfer_counts[row, step])
                    if count > 0:
                        _, indices = self.torch.topk(confidence[row], k=count)
                        transfer[row, indices] = True
                tokens[transfer] = predicted[transfer]
        return tokens[:, prompt_length:]

    def generate(self, records: list[dict[str, Any]]) -> list[str]:
        encoded = self.tokenizer(
            [record["prompt"] for record in records],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.options.max_prompt_length,
        ).to(self.options.device)
        with self.torch.no_grad():
            tokens = self._generate_tokens(
                encoded["input_ids"], encoded["attention_mask"]
            )
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)


def _run_internal_task(
    options: InferenceOptions,
    state: TaskState,
    generate: Callable[[list[dict[str, Any]]], list[str]],
) -> None:
    completed = len(state.existing)
    total = len(state.inputs)
    for batch in iter_batches(state.pending, options.batch_size):
        generated = generate(batch)
        if len(generated) != len(batch):
            raise ValueError(
                f"{state.task}: backend returned {len(generated)} outputs for "
                f"{len(batch)} inputs"
            )
        output = [
            _base_prediction(options.model, source, str(text))
            for source, text in zip(batch, generated)
        ]
        append_jsonl(state.predictions_path, output)
        completed += len(output)
        print(f"[{options.model}/{state.task}] {completed}/{total}", flush=True)
    _finalize_task(state)


_REQUIRED_EXTERNAL_PLACEHOLDERS = {
    "model_path",
    "task",
    "input",
    "output",
    "max_new_tokens",
    "decoding",
    "decoder_cfg",
    "dit_cfg",
    "dit_timestep_schedule",
    "sample_seed",
    "sample_seed_scheme",
}


def _external_argv(options: InferenceOptions, state: TaskState, input_path: Path, output_path: Path) -> list[str]:
    if not options.external_command:
        raise ValueError(
            "Cola2 models require --external-command or COLA2_BENCHMARK_COMMAND"
        )
    missing = sorted(
        placeholder
        for placeholder in _REQUIRED_EXTERNAL_PLACEHOLDERS
        if "{" + placeholder + "}" not in options.external_command
    )
    if missing:
        raise ValueError(
            "external command must contain placeholders: " + ", ".join(missing)
        )
    spec = MODEL_SPECS[options.model]
    values = {
        "python": sys.executable,
        "model": options.model,
        "model_path": options.model_source,
        "task": state.task,
        "input": str(input_path),
        "output": str(output_path),
        "max_new_tokens": str(options.max_new_tokens),
        "max_prompt_length": str(options.max_prompt_length),
        "decoding": DECODING,
        "decoder_cfg": str(spec.decoder_cfg),
        "dit_cfg": str(spec.dit_cfg),
        "dit_timestep_schedule": str(spec.dit_timestep_schedule),
        "sample_seed": str(validate_base_seed(options.sample_seed)),
        "sample_seed_scheme": SAMPLE_SEED_SCHEME,
    }
    return [
        token.format_map(values) for token in shlex.split(options.external_command)
    ]


def _generated_text(record: dict[str, Any], path: Path) -> str:
    for field in ("generate", "generated_text", "output", "text"):
        if field in record:
            value = record[field]
            if not isinstance(value, str):
                raise ValueError(f"{path}: field {field} must be a string")
            return value
    raise ValueError(f"{path}: output record has no generated text field")


def _run_external_task(options: InferenceOptions, state: TaskState) -> None:
    if not state.pending:
        _finalize_task(state)
        return
    with tempfile.TemporaryDirectory(prefix=".external.", dir=state.directory) as temporary:
        temporary_dir = Path(temporary)
        input_path = temporary_dir / "pending.jsonl"
        output_path = temporary_dir / "generated.jsonl"
        atomic_write_jsonl(input_path, state.pending)
        argv = _external_argv(options, state, input_path, output_path)
        subprocess.run(
            argv,
            cwd=options.external_cwd,
            check=True,
        )
        if not output_path.is_file():
            raise FileNotFoundError(
                f"external command did not create {output_path}"
            )
        generated_records = read_jsonl(output_path)
        generated_index = unique_by_id(
            generated_records, source=str(output_path)
        )
        pending_index = unique_by_id(
            state.pending, source=f"pending:{state.task}"
        )
        if set(generated_index) != set(pending_index):
            raise ValueError(
                f"{output_path}: output IDs do not exactly match pending input IDs"
            )
        merged = [
            _base_prediction(
                options.model,
                source,
                _generated_text(
                    generated_index[canonical_id(source["id"])], output_path
                ),
            )
            for source in state.pending
        ]
        append_jsonl(state.predictions_path, merged)
        print(
            f"[{options.model}/{state.task}] "
            f"{len(state.existing) + len(merged)}/{len(state.inputs)}",
            flush=True,
        )
    _finalize_task(state)


def run_inference(options: InferenceOptions) -> dict[str, Any]:
    if options.model not in MODEL_SPECS:
        raise ValueError(f"unknown model: {options.model}")
    if options.max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError(
            f"benchmark max_new_tokens is fixed at {MAX_NEW_TOKENS}"
        )
    if options.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not options.model_source:
        raise ValueError(f"{options.model}: model source is required")
    if MODEL_SPECS[options.model].backend == "cola2":
        validate_base_seed(options.sample_seed)

    states: list[TaskState] = []
    try:
        for task in options.tasks:
            states.append(_prepare_task(options, task))
        pending_states = [state for state in states if state.pending]
        spec = MODEL_SPECS[options.model]
        if spec.backend == "cola2" and options.external_command:
            for state in states:
                _run_external_task(options, state)
        elif pending_states:
            backend: CausalLMBackend | Cola2Backend | LLaDABackend
            if spec.backend == "causal_lm":
                backend = CausalLMBackend(options)
            elif spec.backend == "cola2":
                backend = Cola2Backend(options)
            elif spec.backend == "llada":
                backend = LLaDABackend(options)
            else:
                raise ValueError(f"unsupported backend: {spec.backend}")
            for state in states:
                if state.pending:
                    _run_internal_task(options, state, backend.generate)
                else:
                    _finalize_task(state)
        else:
            for state in states:
                _finalize_task(state)

        return {
            "model": options.model,
            "sample_seed": (
                options.sample_seed
                if MODEL_SPECS[options.model].backend == "cola2"
                else None
            ),
            "tasks": [
                {"task": state.task, "count": len(state.inputs), "pending": 0}
                for state in states
            ],
        }
    finally:
        for state in reversed(states):
            with suppress(OSError):
                state.lock.close()
