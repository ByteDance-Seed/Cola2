# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepare and verify the fixed benchmark inputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .contracts import CONTRACT_VERSION, TASK_SIZES, expected_count
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_id,
    ids_sha256,
    json_dumps,
    read_jsonl,
    sha256_file,
    unique_by_id,
)
from .prompt_template import apply_prompt_template


MANIFEST_NAME = "manifest.json"
SOURCE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "eval_data" / MANIFEST_NAME
)


def _source_contract() -> dict[str, Any]:
    with SOURCE_CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), dict):
        raise ValueError(f"{SOURCE_CONTRACT_PATH}: invalid source-data contract")
    return value


def _disambiguate_duplicate_ids(
    task: str, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Preserve repeated source rows while assigning unambiguous benchmark IDs."""
    for position, record in enumerate(records, start=1):
        if record.get("id") is None:
            raise ValueError(f"{task}:{position}: missing non-null source id")
    keys = [canonical_id(record.get("id")) for record in records]
    counts = Counter(keys)
    occurrences: Counter[str] = Counter()
    used = set(keys)
    duplicate_records = sum(count for count in counts.values() if count > 1)
    if not duplicate_records:
        return records, 0

    result: list[dict[str, Any]] = []
    for record, key in zip(records, keys):
        if counts[key] == 1:
            result.append(record)
            continue
        occurrences[key] += 1
        source_id = record.get("id")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        candidate = (
            f"__cola2_benchmark__:{task}:{digest}:{occurrences[key]}"
        )
        candidate_key = canonical_id(candidate)
        if candidate_key in used:
            raise ValueError(
                f"{task}: generated duplicate-safe ID collides for {json_dumps(source_id)}"
            )
        used.add(candidate_key)
        updated = dict(record)
        updated["source_id"] = source_id
        updated["id"] = candidate
        result.append(updated)
    return result, duplicate_records


def prepare_record(task: str, source: dict[str, Any]) -> dict[str, Any]:
    if source.get("id") is None:
        raise ValueError(f"{task}: every record must have a non-null id")
    if "question" not in source:
        raise ValueError(f"{task} id={source['id']!r}: missing question")
    if "ground_truth" in source:
        gold = source["ground_truth"]
    elif "answer" in source:
        gold = source["answer"]
    else:
        raise ValueError(f"{task} id={source['id']!r}: missing ground_truth/answer")

    choices = source.get("choices")
    if task == "truthfulqa" and (
        not isinstance(choices, list) or not choices
    ):
        raise ValueError(
            f"truthfulqa id={source['id']!r}: choices must be retained and non-empty"
        )

    prepared = dict(source)
    prepared["task"] = task
    prepared["ground_truth"] = gold
    prepared["prompt"] = apply_prompt_template(
        task=task,
        context=source.get("context", "") or "",
        question=source.get("question", "") or "",
        answer=gold,
        choices=choices,
    )
    return prepared


def prepare_eval_data(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    tasks: Iterable[str] | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    source_root = Path(source_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    selected_tasks = list(tasks or TASK_SIZES)
    unknown = sorted(set(selected_tasks) - set(TASK_SIZES))
    if unknown:
        raise ValueError(f"unknown tasks: {', '.join(unknown)}")
    missing = [
        str(source_root / f"{task}.jsonl")
        for task in selected_tasks
        if not (source_root / f"{task}.jsonl").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing external benchmark source files: "
            + ", ".join(missing)
            + ". Download the normalized files from the Cola2 Hugging Face "
            "release (link coming soon), then set GENERATE_TASK_DATA_DIR or "
            "--source-data-dir."
        )
    source_contract = _source_contract()
    contract_tasks = source_contract["tasks"]

    prompt_path = Path(__file__).with_name("prompt_template.py")
    manifest: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "max_samples": max_samples,
        "source_dir": str(source_root),
        "prompt_template_sha256": sha256_file(prompt_path),
        "tasks": {},
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for task in selected_tasks:
        source_path = source_root / f"{task}.jsonl"
        expected_source = contract_tasks.get(task)
        if not isinstance(expected_source, dict):
            raise ValueError(
                f"{SOURCE_CONTRACT_PATH}: missing task contract for {task}"
            )
        source_records = read_jsonl(source_path)
        actual_source_count = len(source_records)
        required_source_count = TASK_SIZES[task]
        if actual_source_count != required_source_count:
            raise ValueError(
                f"{source_path}: expected {required_source_count} records, "
                f"found {actual_source_count}"
            )
        observed_source_sha256 = sha256_file(source_path)
        expected_source_sha256 = expected_source.get("sha256")
        if observed_source_sha256 != expected_source_sha256:
            raise ValueError(
                f"{source_path}: SHA-256 mismatch; expected "
                f"{expected_source_sha256}, found {observed_source_sha256}"
            )
        source_records, duplicate_source_id_records = _disambiguate_duplicate_ids(
            task, source_records
        )

        limit = expected_count(task, max_samples)
        selected_source = source_records[:limit]
        prepared_records = [prepare_record(task, record) for record in selected_source]
        unique_by_id(prepared_records, source=f"prepared:{task}")

        output_path = output_root / f"{task}.jsonl"
        atomic_write_jsonl(output_path, prepared_records)
        manifest["tasks"][task] = {
            "source_count": actual_source_count,
            "prepared_count": len(prepared_records),
            "source_sha256": observed_source_sha256,
            "prepared_sha256": sha256_file(output_path),
            "ids_sha256": ids_sha256(prepared_records),
            "duplicate_source_id_records": duplicate_source_id_records,
        }

    atomic_write_json(output_root / MANIFEST_NAME, manifest)
    return manifest


def prepared_data_matches(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    tasks: Iterable[str] | None = None,
    max_samples: int = 0,
) -> bool:
    selected_tasks = list(tasks or TASK_SIZES)
    source_root = Path(source_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    try:
        manifest = load_manifest(output_root)
        if manifest.get("contract_version") != CONTRACT_VERSION:
            return False
        if manifest.get("max_samples") != max_samples:
            return False
        if manifest.get("prompt_template_sha256") != sha256_file(
            Path(__file__).with_name("prompt_template.py")
        ):
            return False
        task_manifest = manifest.get("tasks")
        if not isinstance(task_manifest, dict):
            return False
        for task in selected_tasks:
            details = task_manifest.get(task)
            if not isinstance(details, dict):
                return False
            source_path = source_root / f"{task}.jsonl"
            output_path = output_root / f"{task}.jsonl"
            if not source_path.is_file() or not output_path.is_file():
                return False
            if details.get("source_sha256") != sha256_file(source_path):
                return False
            if details.get("prepared_sha256") != sha256_file(output_path):
                return False
            if details.get("prepared_count") != expected_count(task, max_samples):
                return False
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return False
    return True


def ensure_eval_data(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    tasks: Iterable[str] | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    """Prepare once, with an inter-process lock for parallel shards."""
    import fcntl

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_tasks = list(tasks or TASK_SIZES)
    lock_path = output_root / ".prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if prepared_data_matches(
            source_dir,
            output_root,
            tasks=selected_tasks,
            max_samples=max_samples,
        ):
            return load_manifest(output_root)
        return prepare_eval_data(
            source_dir,
            output_root,
            tasks=selected_tasks,
            max_samples=max_samples,
        )


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared-data manifest: {path}")
    import json

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_task_data(
    data_dir: str | Path, task: str, *, max_samples: int = 0
) -> list[dict[str, Any]]:
    if task not in TASK_SIZES:
        raise ValueError(f"unknown task: {task}")
    records = read_jsonl(Path(data_dir) / f"{task}.jsonl")
    if max_samples > 0:
        records = records[:max_samples]
    expected = expected_count(task, max_samples)
    if len(records) != expected:
        raise ValueError(
            f"prepared {task}: expected {expected} records for max_samples="
            f"{max_samples}, found {len(records)}"
        )
    unique_by_id(records, source=f"prepared:{task}")
    for position, record in enumerate(records, start=1):
        if record.get("task") != task:
            raise ValueError(
                f"prepared {task}:{position}: task field is {record.get('task')!r}"
            )
        if not isinstance(record.get("prompt"), str):
            raise ValueError(f"prepared {task}:{position}: missing prompt")
        if task == "truthfulqa" and not record.get("choices"):
            raise ValueError(
                f"prepared truthfulqa:{position}: choices were not retained"
            )
    return records
