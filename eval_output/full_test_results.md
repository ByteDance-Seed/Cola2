# Post-pretraining evaluation

Accuracy (%) on eight downstream tasks, as reported in the
[Cola2 paper](https://arxiv.org/abs/2610.31305). The models are not
compute-matched and use different training data, tokenizers, objectives, and
token-counting conventions.

Cola2 uses 346.03B **source tokens** and the official low-noise 8-step DiT,
DiT CFG 10, decoder CFG -0.9, one latent span per decoder chunk, dynamic
duration, latent feedback, and greedy token selection. Each Cola2 task value is
the mean of three evaluations with independently sampled latent noise. The
original random seeds are not published.

| Evaluation focus | Task | Meta-Llama-3-8B | Qwen2.5-7B | Qwen3-8B-Base | LLaDA-8B-Base | Cola2 |
|---|---|---:|---:|---:|---:|---:|
| — | Pretraining tokens | 15T+ | 18T | ~36T | 2.3T | 346.03B source tokens |
| Reading comprehension | RACE | 75.77 | 89.95 | 90.65 | 87.40 | 69.91 |
| Reading comprehension | NarrativeQA | 42.70 | 32.90 | 37.30 | 36.00 | 32.70 |
| Commonsense reasoning | HellaSwag | 39.21 | 75.67 | 68.13 | 69.65 | 41.75 |
| Commonsense reasoning | StoryCloze | 79.49 | 95.09 | 97.44 | 95.94 | 80.42 |
| Language modeling | LAMBADA | 75.70 | 71.82 | 71.20 | 67.90 | 63.00 |
| Knowledge-oriented QA | ARC-Easy | 90.86 | 96.00 | 97.35 | 93.85 | 87.62 |
| Knowledge-oriented QA | TruthfulQA | 63.04 | 69.03 | 72.58 | 64.87 | 54.10 |
| Knowledge-oriented QA | MMLU | 63.19 | 70.16 | 72.31 | 61.27 | 51.32 |
| **All tasks** | **8-task average** | **66.25** | **75.08** | **75.87** | **72.11** | **60.10** |

The normalized JSONL files are distributed externally with the Hugging Face
model release (link coming soon). This repository retains the task contract,
expected hashes, prompt templates, scorer, and machine-readable copies of this
table. See [the reproduction protocol](../docs/reproduction.md) before
comparing local runs with these reference values.
