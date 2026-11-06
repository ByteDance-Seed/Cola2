#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES="${COLA2_QUICK_SAMPLES:-200}"

if ! [[ "${SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "COLA2_QUICK_SAMPLES must be a positive integer." >&2
    exit 2
fi

exec "${SCRIPT_DIR}/run_benchmark_all.sh" --max-samples "${SAMPLES}" "$@"
