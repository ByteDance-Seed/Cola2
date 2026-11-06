#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COLA2_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK_ROOT="${COLA2_BENCHMARK_ROOT:-${COLA2_ROOT}/benchmark_artifacts}"
PAPER_SEEDS="${COLA2_PAPER_SEEDS:-42 43 44}"

read -r -a SEEDS <<< "${PAPER_SEEDS}"
if [[ "${#SEEDS[@]}" -ne 3 ]]; then
    echo "COLA2_PAPER_SEEDS must contain exactly three integer seeds." >&2
    exit 2
fi

declare -A SEEN=()
for seed in "${SEEDS[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed in COLA2_PAPER_SEEDS: ${seed}" >&2
        exit 2
    fi
    if [[ -n "${SEEN[${seed}]:-}" ]]; then
        echo "COLA2_PAPER_SEEDS must contain three distinct seeds." >&2
        exit 2
    fi
    SEEN["${seed}"]=1
done

export PYTHONPATH="${COLA2_ROOT}:${COLA2_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SUMMARY_ARGS=()
for seed in "${SEEDS[@]}"; do
    RUN_ROOT="${BENCHMARK_ROOT}/paper_runs/seed-${seed}"
    "${PYTHON_BIN}" -m benchmarks.orchestrate \
        "$@" \
        --models cola2 \
        --sample-seed "${seed}" \
        --work-dir "${RUN_ROOT}" \
        --data-dir "${BENCHMARK_ROOT}/paper_data" \
        --output-dir "${RUN_ROOT}/outputs" \
        --summary-dir "${RUN_ROOT}/summary"
    SUMMARY_ARGS+=(--summary "${RUN_ROOT}/summary/summary.json")
done

"${PYTHON_BIN}" -m benchmarks.aggregate_runs \
    "${SUMMARY_ARGS[@]}" \
    --model cola2 \
    --output-dir "${BENCHMARK_ROOT}/paper_summary"
