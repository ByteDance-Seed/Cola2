# Evaluation data

[中文](README_zh.md) · [Benchmark documentation](../docs/benchmark.md)

Normalized benchmark JSONL files are distributed externally with the Cola2
[Hugging Face model repository](https://huggingface.co/ByteDance-Seed/Cola2);
large evaluation files are intentionally not committed to this source
repository.

## Expected files

Download these files into this directory, or place them elsewhere and set
`GENERATE_TASK_DATA_DIR`:

```text
arc_easy.jsonl
hellaswag.jsonl
lambada.jsonl
mmlu.jsonl
narrativeqa.jsonl
race.jsonl
story_cloze.jsonl
truthfulqa.jsonl
```

`manifest.json` is versioned with the code and records each expected filename,
row count, and SHA-256 digest. The benchmark rejects missing files and incorrect
row counts; the prepared-data manifest records observed hashes for auditability.

## Record contract

Source rows contain a non-null `id`, `question`, an answer in
`ground_truth` or `answer`, and task-specific fields such as `context`,
`choices`, and `type`. Preparation adds the fixed prompt and canonical `task`
field without altering ground truth or choices.

Do not reorder or deduplicate files. NarrativeQA intentionally contains
repeated source IDs; preparation preserves every row and assigns an unambiguous
benchmark ID based on source position.

## Provenance

Records are derived from public evaluation splits of:

- RACE
- HellaSwag
- LAMBADA
- ARC-Easy
- Story Cloze
- NarrativeQA
- TruthfulQA
- MMLU

Each upstream dataset remains subject to its original license, terms, and
citation requirements. The normalized files are evaluation artifacts and must
not be treated as model-training data. Consult the Hugging Face release and
upstream dataset cards before redistribution.
