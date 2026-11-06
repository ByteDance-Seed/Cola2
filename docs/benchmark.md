# Benchmark

[中文](benchmark_zh.md) · [Documentation](README.md) ·
[Reproduction protocol](reproduction.md)

The public benchmark fixes the data normalization, prompt templates, generation
length, output schema, and scorer used for the reported eight-task comparison.
It is designed for auditable reproduction rather than as a general evaluation
framework.

## External data

Normalized source JSONL files are not committed to this repository. They are
published in the same
[Hugging Face repository](https://huggingface.co/ByteDance-Seed/Cola2) as the
Cola2 checkpoint.

After download, either place all eight files in `eval_data/` or set:

```bash
export GENERATE_TASK_DATA_DIR=/path/to/eval_data
```

The files must match [eval_data/manifest.json](../eval_data/manifest.json).
Preparation rejects incomplete datasets and records source and prepared
SHA-256 values. Do not reorder or deduplicate rows: NarrativeQA intentionally
contains repeated source IDs, which are disambiguated by input position.

## Tasks

| Task | Source rows | Evaluation focus |
|---|---:|---|
| RACE | 4,887 | Reading comprehension |
| NarrativeQA | 1,000 | Extractive/open answer |
| HellaSwag | 10,042 | Commonsense completion |
| Story Cloze | 468 | Story ending |
| LAMBADA | 5,153 | Last-word completion |
| ARC-Easy | 2,375 | Science multiple choice |
| TruthfulQA | 817 | Truthfulness multiple choice |
| MMLU | 14,042 | Multidomain knowledge |

All tasks except LAMBADA use fixed few-shot prompt templates. Prompt construction
is implemented in `benchmarks/prompt_template.py`; answer extraction and
scoring are in `benchmarks/task_scoring.py`.

The reported aggregate is the unweighted arithmetic mean of the eight task
accuracies. It is not weighted by sample count.

## Models and decoding contract

The paper table contains Meta-Llama-3-8B, Qwen2.5-7B, Qwen3-8B-Base,
LLaDA-8B-Base, and Cola2. All columns use:

- the same normalized rows and fixed prompts;
- greedy token selection;
- `max_new_tokens=32`;
- maximum prompt length 2048 in the public runner;
- the same prediction schema and task scorer.

Cola2 additionally uses low-noise 8-step DiT, DiT CFG 10, decoder CFG -0.9,
one latent span per decoder chunk, dynamic duration, and latent feedback.
Although token decoding is greedy, the DiT starts from sampled Gaussian noise.
A base seed is therefore part of the public run configuration.

The Meta Llama checkpoint is gated and requires an accepted license and an
authenticated Hugging Face account. Hardware and memory requirements vary by
model; the runner defaults to CUDA and bfloat16 and does not perform automatic
offloading.

## Commands

Quick Cola2 smoke run, 200 rows per selected task:

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export GENERATE_TASK_DATA_DIR=/path/to/eval_data

bash scripts/run_benchmark_quick.sh \
  --models cola2 \
  --tasks "lambada arc_easy"
```

Override the quick count with `COLA2_QUICK_SAMPLES`:

```bash
COLA2_QUICK_SAMPLES=100 bash scripts/run_benchmark_quick.sh --models cola2
```

Complete five-model grid:

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export LLAMA3_MODEL_PATH=meta-llama/Meta-Llama-3-8B
export QWEN25_MODEL_PATH=Qwen/Qwen2.5-7B
export QWEN3_MODEL_PATH=Qwen/Qwen3-8B-Base
export LLADA_MODEL_PATH=GSAI-ML/LLaDA-8B-Base

bash scripts/run_benchmark_all.sh --max-samples 0
```

Paper-style three-seed Cola2 run:

```bash
COLA2_PAPER_SEEDS="42 43 44" \
  bash scripts/run_benchmark_paper.sh --max-samples 0
```

The manuscript does not publish its original random seeds. The public seeds
above define a repeatable protocol with the same three-evaluation averaging
rule, but exact equality with the historical table is not guaranteed.

## Prepared data and artifacts

With the default `COLA2_BENCHMARK_ROOT=benchmark_artifacts`, a single run writes:

```text
benchmark_artifacts/
├── data/
│   ├── manifest.json
│   └── {task}.jsonl
├── outputs/{model}/{task}/
│   ├── run_config.json
│   ├── predictions.jsonl
│   ├── scored.jsonl
│   ├── metrics.json
│   └── complete.json
└── summary/
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

The paper script keeps each seed in an independent run directory and writes an
aggregate mean/standard-deviation summary under `paper_summary/`.

`run_config.json` records model source, dtype, device, decoding settings, and
the base seed. Completion markers include hashes of predictions, scores,
metrics, and configuration so stale artifacts are not silently reused.

## Resume and sharding

Inference appends predictions and can recover an incomplete trailing JSONL
record. Resume is enabled by default. If any inference setting changes,
including `sample_seed`, the runner requires `--force` instead of mixing
artifacts.

Independent jobs can split the grid with:

```bash
python -m benchmarks.orchestrate \
  --model-shard-index 0 --model-shard-count 2 \
  --task-shard-index 0 --task-shard-count 4
```

The deterministic seed for Cola2 is derived from the base seed, task, and
canonical sample ID, so task order, resume, and sharding do not change an
individual sample.

## External Cola2 adapter

`COLA2_BENCHMARK_COMMAND` can replace the installed Cola2 backend. The command
template must consume the documented placeholders, including `{input}`,
`{output}`, model/settings fields, `{sample_seed}`, and
`{sample_seed_scheme}`. The adapter must derive each record's generator seed
with the same `sha256-v1` algorithm implemented by
`benchmarks.sample_rng.derive_sample_seed(base_seed, task, sample_id)`. Run
`python -m benchmarks.run_inference --help` for the complete contract.

## Interpreting results

[eval_output/full_test_results.md](../eval_output/full_test_results.md) contains
reported paper values. Local outputs may differ because the original Cola2
seeds are not published and because devices or dependency versions can affect
numerics. Compare protocol and per-task trends, and preserve generated
`run_config.json` files when reporting results.
