#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COLA2_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAMPLE_SEED="${COLA2_BENCHMARK_SEED:-42}"

export PYTHONPATH="${COLA2_ROOT}:${COLA2_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m benchmarks.orchestrate \
    --sample-seed "${SAMPLE_SEED}" \
    "$@"
