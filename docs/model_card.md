# Model card: Cola2

[中文](model_card_zh.md) · [Documentation](README.md)

## Model summary

Cola2 is a base continuous-latent language model for hierarchical text
generation. It combines:

- a dynamic-compression VAE that maps 1–8 token semantic spans to
  64-dimensional latents;
- an approximately 4B-parameter, 34-layer block-causal latent DiT; and
- a 36-layer Qwen3-4B-shaped span-causal token decoder.

The model uses dynamic duration prediction and re-encodes committed text as
latent feedback. The checkpoint and tokenizer are hosted in the
[Hugging Face model repository](https://huggingface.co/ByteDance-Seed/Cola2).
This repository contains the inference implementation and evaluation contract.

## Paper and project

- **Paper:** *Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous
  Semantic Compression and Balanced Discrete Generation*
- **Project page:** <https://hongcanguo.github.io/Cola2/>
- **Paper/arXiv:** <https://arxiv.org/abs/2610.31305>
- **Model repository:** <https://huggingface.co/ByteDance-Seed/Cola2>
- **Architecture:** [docs/architecture.md](architecture.md)

## Training summary

The reported checkpoint uses OLMo 2 pretraining text retokenized with the
Qwen3-4B tokenizer. The decoder and VAE encoder embedding are initialized from
Qwen3-4B-Base.

| Stage | Trainable components | Additional source tokens |
|---|---|---:|
| I | Dynamic-compression VAE | 52.43B |
| II | VAE + approximately 4B latent DiT, detached DiT target | 125.83B |
| III | DiT with frozen Stage-II EMA VAE | 167.77B |
| **Total** |  | **346.03B** |

This public repository does not include training code, the training corpus,
optimizer state, or enough artifacts to reproduce training.

## Inputs and outputs

Input is tokenized text accepted by the released Qwen3-compatible tokenizer.
The model returns the original prompt followed by generated token IDs, matching
the decoder-only Transformers `generate()` convention.

The practical context and output limits depend on memory and checkpoint
configuration. The public benchmark truncates prompts to 2048 tokens and
generates at most 32 new tokens; those are evaluation settings, not a general
quality guarantee.

## Intended use

- research on continuous-latent and block-parallel language generation;
- analysis of semantic compression and latent/token co-adaptation;
- short completion, cloze, multiple-choice, and extractive-answer experiments;
- reproduction of the published inference and scoring protocol.

Users should independently validate model behavior for any downstream
application.

## Out-of-scope use

- treating outputs as verified facts or professional advice;
- deploying as an instruction-following or safety-aligned assistant;
- autonomous high-impact decisions about people;
- unlawful, deceptive, abusive, privacy-invasive, or rights-violating use.

Cola2 is not instruction-tuned, preference-tuned, or documented as having
undergone deployment-grade safety alignment.

## Official inference setting

- low-noise 8-step Euler DiT,
  \(t_i=T(1-i/8)^2\);
- DiT CFG 10;
- decoder CFG -0.9;
- one latent span per decoder chunk;
- dynamic duration and latent feedback enabled;
- greedy token selection with `max_new_tokens=32`.

Latent generation remains stochastic under greedy token selection. Supply a
seeded `torch.Generator` for replay. See [Inference](inference.md).

## Evaluation

The reported results are in
[eval_output/full_test_results.md](../eval_output/full_test_results.md).
Cola2 reports the following accuracies:

| Task | Accuracy (%) |
|---|---:|
| RACE | 69.91 |
| NarrativeQA | 32.70 |
| HellaSwag | 41.75 |
| Story Cloze | 80.42 |
| LAMBADA | 63.00 |
| ARC-Easy | 87.62 |
| TruthfulQA | 54.10 |
| MMLU | 51.32 |
| **Unweighted mean** | **60.10** |

Each Cola2 task value is the mean of three evaluations with independently
sampled latent noise. Baseline and Cola2 training budgets are not
compute-matched, and models use different tokenizers and data conventions.

Normalized benchmark JSONL files will be distributed externally with the model.
Prompt templates, scorer, expected counts, and hashes remain in this
repository. See [Benchmark](benchmark.md) and
[Reproduction](reproduction.md).

## Limitations and risks

- Overall benchmark quality remains below the stronger listed baselines.
- Long-form output may repeat, become ungrammatical, lose coherence, or drift
  off topic.
- The published evaluation is English-only and does not establish multilingual,
  coding, mathematical, long-context, factual, robustness, or safety quality.
- Exact outputs depend on sampled latent noise and may vary across software and
  hardware environments.
- Generated text may reproduce factual errors, stereotypes, toxic content,
  copyrighted text, personal information, or other undesirable patterns from
  training data.
- Benchmark coverage is narrow and may contain contamination or annotation
  limitations inherited from upstream datasets.

Apply application-specific testing, output validation, access controls, and
safety review before use.

## Privacy and security

Do not provide secrets or sensitive personal data unless the deployment has
been explicitly designed and reviewed for that purpose. Report vulnerabilities
through the private channels in [SECURITY.md](../SECURITY.md), not a public
issue.

## License

Repository code is licensed under
[Apache License 2.0](../LICENSE). Model weights, tokenizer assets, normalized
benchmark files, and upstream datasets remain subject to the terms published
with those artifacts. See [NOTICE](../NOTICE).
