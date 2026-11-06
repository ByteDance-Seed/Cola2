# Changelog

All notable changes to Cola2 are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Linked the public [arXiv paper](https://arxiv.org/abs/2610.31305).
- Added the overview video, controlled-scaling figures, and final eight-task
  paper table to the bilingual repository home pages.

Checkpoint, tokenizer, and normalized benchmark download links will be added
when available.

## [0.1.0] - 2026-10-22

### Added

- Hugging Face-compatible Cola2 configuration and inference implementation.
- Dynamic-compression VAE, block-causal latent DiT, span-causal decoder, and
  latent-feedback generation.
- Interactive CLI and Python quick-start example.
- Resumable eight-task benchmark, fixed prompts, scorer, validation, and
  JSON/CSV/Markdown summaries.
- Stable per-record latent-noise seeds and three-run paper-style aggregation.
- English and Chinese architecture, inference, configuration, benchmark,
  reproduction, paper-overview, and model-card documentation.
- Apache-2.0 licensing, source-file SPDX headers, security policy, citation,
  contribution guidelines, and CPU CI.

### Changed

- Normalized benchmark JSONL files are distributed externally with the model
  release instead of being committed to the source repository.
