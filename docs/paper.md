# Paper overview

[中文](paper_zh.md) · [Documentation](README.md)

## Metadata

**Title:** *Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous
Semantic Compression and Balanced Discrete Generation*

**Authors:** Hongcan Guo, Yian Zhao, Feng Wang, Shen Nie, Guoqiang Wei,
Yan Zeng, and Hengshuang Zhao.

**Affiliations:** The University of Hong Kong, ByteDance Seed, Peking
University, and Renmin University of China.

[Project page](https://hongcanguo.github.io/Cola2/) ·
[arXiv:2610.31305](https://arxiv.org/abs/2610.31305)

## Motivation

Hierarchical language models separate semantic planning from token
realization. Their performance depends on two coupled questions: what
information should be retained in a compressed continuous representation, and
how should a discrete decoder combine that representation with its own
language-modeling ability? Cola2 studies both questions jointly rather than
treating the latent encoder and token decoder as independent modules.

## Method

Cola2 contains three co-adapted parts:

1. **Dynamic semantic compression.** Text is segmented into variable-length
   spans. An auxiliary-loss-free negative-feedback mechanism controls the mean
   compression rate, and each span is mapped to a compact continuous latent.
2. **Latent generation.** A block-causal DiT learns a flow-matching prior over
   latent blocks. The flow-matching gradient is stopped at the latent target so
   the generator cannot make the representation easier to predict at the cost
   of token semantics.
3. **Balanced discrete generation.** The token decoder is trained with clean,
   noisy, and dropped latent conditions. Decoder classifier-free guidance
   adjusts the balance between following the latent and generating residual
   token-level information.

At inference time, generated text is re-encoded under its committed span
layout. This latent-feedback step keeps subsequent latent history aligned with
the text that was actually emitted.

## Final checkpoint training

The reported release checkpoint uses OLMo 2 pretraining text retokenized with
the Qwen3-4B tokenizer. The decoder and VAE encoder embedding are initialized
from Qwen3-4B-Base.

- **Stage I:** train the dynamic-compression VAE on 52.43B source tokens.
- **Stage II:** jointly update the VAE and an approximately 4B-parameter latent
  DiT, with detached VAE targets for the DiT objective, on 125.83B additional
  source tokens.
- **Stage III:** freeze the Stage-II EMA VAE and continue training the DiT on
  167.77B additional source tokens.

The total is **346.03B source tokens**. This repository does not contain the
training implementation, optimizer state, or training corpus.

## Reported findings

Component studies support three conclusions:

- semantic compression benefits from dynamic segmentation and generative
  supervision;
- the decoder works best when reconstruction and autonomous generation are
  balanced; and
- the latent representation and token generator benefit from co-adaptation.

The final checkpoint reports a **60.10** unweighted average over eight tasks.
It exceeds the reported Llama-3-8B values on HellaSwag and Story Cloze, while
remaining below stronger baselines on the overall average. The compared models
use different data, tokenizers, objectives, and token-counting conventions, so
the table contextualizes the checkpoint rather than establishing a
compute-matched comparison.

Each reported Cola2 task score is the mean of three evaluations with
independently sampled latent noise. Token decoding is greedy; the stochasticity
comes from the DiT's initial Gaussian noise. See the
[reproduction protocol](reproduction.md).

## Mapping to this repository

The paper discusses a broad hierarchical framework and controlled experiments
with autoregressive and masked-diffusion token decoders. The released
`Cola2ForConditionalGeneration` checkpoint exposes the concrete Full-DiT
latent generator and span-causal token realization path described in
[Architecture](architecture.md). Documentation of the released code takes
precedence when an experimental term does not map one-to-one to a public API.

## Citation

Cite the arXiv preprint using the entry in [CITATION.md](../CITATION.md).
Machine-readable paper and software metadata is available in
[`CITATION.cff`](../CITATION.cff).
