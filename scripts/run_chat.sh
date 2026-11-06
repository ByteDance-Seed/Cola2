#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COLA2_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_MODEL="${COLA2_ROOT}/hf_models/cola2"
MODEL_PATH="${COLA2_MODEL_PATH:-${DEFAULT_MODEL}}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Cola2 model not found at ${MODEL_PATH}." >&2
    echo "Set COLA2_MODEL_PATH to a downloaded model directory." >&2
    exit 2
fi

export PYTHONPATH="${COLA2_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m cola2.cli_chat --model "${MODEL_PATH}" "$@"
