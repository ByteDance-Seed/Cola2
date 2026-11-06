# Reproduction protocol

[中文](reproduction_zh.md) · [Documentation](README.md) ·
[Benchmark reference](benchmark.md)

This guide separates three goals:

1. a fast smoke test of the public pipeline;
2. one complete, deterministic benchmark run; and
3. the paper-style mean of three independently sampled Cola2 runs.

The model checkpoint and normalized source JSONL files are external artifacts.
They are hosted at
[ByteDance-Seed/Cola2](https://huggingface.co/ByteDance-Seed/Cola2).

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

export COLA2_MODEL_PATH=/path/to/cola2
export GENERATE_TASK_DATA_DIR=/path/to/eval_data
```

Verify the checkpoint's `SHA256SUMS` and confirm the eight source files match
`eval_data/manifest.json`. The runner checks row counts and records hashes in
the prepared-data manifest.

CUDA and bfloat16 are the public defaults. Record GPU, driver, CUDA, PyTorch,
Transformers, and checkpoint revision when publishing numbers.

## 2. Smoke test

Run a small Cola2-only subset:

```bash
COLA2_QUICK_SAMPLES=20 bash scripts/run_benchmark_quick.sh \
  --models cola2 \
  --tasks "lambada arc_easy"
```

This validates data preparation, model loading, prediction, scoring, and
summary generation. It is not comparable to the complete paper table.

## 3. One deterministic complete run

```bash
export COLA2_BENCHMARK_SEED=42
bash scripts/run_benchmark_all.sh \
  --models cola2 \
  --max-samples 0
```

For each record, the runner calls
`benchmarks.sample_rng.derive_sample_seed(base_seed, task, sample_id)`. The
`sha256-v1` scheme hashes the UTF-8 payload
`"sha256-v1\n{base_seed}\n{task}\n{canonical_sample_id}"`, interprets the first
eight digest bytes as an unsigned big-endian integer, and reduces it modulo
`2**63 - 1`.

This makes an individual output invariant to task ordering, resume, and
model/task sharding. It also means changing the base seed creates a distinct
run and invalidates the previous run configuration.

## 4. Paper-style three-run mean

The manuscript reports each Cola2 task score as the mean of three evaluations
with independently sampled latent noise. Token selection remains greedy.

```bash
COLA2_PAPER_SEEDS="42 43 44" \
  bash scripts/run_benchmark_paper.sh \
  --max-samples 0
```

The script creates independent artifacts for all three seeds and then writes:

```text
benchmark_artifacts/
├── paper_runs/seed-42/
├── paper_runs/seed-43/
├── paper_runs/seed-44/
└── paper_summary/
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

`paper_summary` contains per-task run values, arithmetic mean, and population
standard deviation. The eight-task result is the unweighted mean of the eight
aggregated task means.

The paper's original seeds are not available. Seeds 42, 43, and 44 define a
public reproducible protocol, not a guarantee of byte-identical historical
outputs or the exact reported 60.10.

## 5. Baseline grid

To run the full comparison, configure all model sources:

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export LLAMA3_MODEL_PATH=meta-llama/Meta-Llama-3-8B
export QWEN25_MODEL_PATH=Qwen/Qwen2.5-7B
export QWEN3_MODEL_PATH=Qwen/Qwen3-8B-Base
export LLADA_MODEL_PATH=GSAI-ML/LLaDA-8B-Base

bash scripts/run_benchmark_all.sh --max-samples 0
```

Meta Llama access is gated. The paper's baseline columns are reported values;
dependency revisions and model snapshots should be pinned when independently
rerunning them.

## 6. Resume, force, and shards

- Resume is enabled by default and skips valid completion markers.
- Use `--force` to replace artifacts after changing a model snapshot, seed, or
  inference setting.
- Use `--no-resume` to fail if output already exists.
- `--model-shard-index/count` and `--task-shard-index/count` divide independent
  cells across jobs.
- Use `--dry-run` to inspect pending cells without loading a model.

Never merge prediction files from different `run_config.json` values.

## 7. Independent validation

The pipeline validates:

- source and prepared sample counts;
- unique benchmark IDs after duplicate-source-ID disambiguation;
- prompts, ground truth, choices, and task/model labels;
- scored record counts and accuracy arithmetic;
- hashes of predictions, scored output, metrics, and run configuration.

Machine-readable output is the authority for downstream analysis:

- `summary.json` for one run;
- `paper_summary/summary.json` for a three-run aggregate; and
- [eval_output/full_test_results.json](../eval_output/full_test_results.json)
  for the reported paper table.

## 8. What is and is not reproduced

The public repository reproduces the inference preset, prompt/scoring contract,
and three-run aggregation rule. It does not reproduce training and cannot
guarantee the historical random draws. If local values differ, publish the
generated run configuration and environment rather than replacing the reported
reference table.
