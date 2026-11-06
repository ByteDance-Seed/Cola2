# Cola2 documentation

[中文](README_zh.md) · [Repository home](../README.md)

This documentation describes the released inference implementation and the
public evaluation contract. Model weights, tokenizer assets, and normalized
benchmark JSONL files are distributed through the same Hugging Face
[repository](https://huggingface.co/ByteDance-Seed/Cola2).

## Start here

- [Paper overview](paper.md) and
  [arXiv:2610.31305](https://arxiv.org/abs/2610.31305): motivation, method,
  training stages, and headline findings.
- [Architecture](architecture.md): released modules and the block generation
  loop.
- [Inference](inference.md): checkpoint loading, seeded replay, CLI, and
  troubleshooting.
- [Configuration](configuration.md): model fields, generation controls, aliases,
  and environment variables.

## Evaluation

- [Benchmark](benchmark.md): tasks, prompting, scoring, data contract, and
  artifact layout.
- [Reproduction protocol](reproduction.md): quick, complete, and three-seed
  paper-style runs.
- [Evaluation data](../eval_data/README.md): external distribution and manifest
  verification.
- [Reported results](../eval_output/full_test_results.md): the table published
  with the paper.

## Release and governance

- [Model card](model_card.md)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Citation](../CITATION.md)
- [Apache License 2.0](../LICENSE)
- [Third-party notices](../NOTICE)

## Repository boundary

This is an inference and evaluation repository. It intentionally excludes
training code, checkpoints, tokenizer assets, benchmark JSONL files, and the
paper's LaTeX or figure sources. The paper overview summarizes relevant
training facts but does not imply that training is reproducible from this
repository.
