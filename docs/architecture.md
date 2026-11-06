# Architecture

[中文](architecture_zh.md) · [Documentation](README.md)

Cola2 factorizes language generation into a continuous semantic representation,
a latent prior, and a discrete token realization model. The released
implementation is inference-only.

## System view

```text
prompt tokens
    │
    ▼
dynamic-compression VAE ──► committed latent history
                                      │
                                      ▼
Gaussian noise ──► block-causal DiT ──► new latent block
                                      │
                                      ▼
                           duration + span-causal decoder
                                      │
                                      ▼
                              committed text spans
                                      │
                                      └── re-encode as latent feedback
```

## Dynamic-compression VAE

`Cola2CompressionVAE` maps contiguous token spans to continuous latent vectors.
The released configuration:

- dynamically chooses span lengths from 1 to 8 tokens;
- targets an average compression rate of 0.5;
- represents each span with a 64-dimensional variational latent;
- uses a 2560-dimensional token embedding, span pooling layers, and latent
  encoder layers; and
- expands latents back to token-aligned conditions for the decoder.

Boundary scores identify semantic transitions. A negative-feedback controller
adjusts the effective threshold to regulate compression without adding a rate
loss to the language-modeling objective.

## Latent DiT

`Cola2DiT` is an approximately 4B-parameter Diffusion Transformer in the
reported checkpoint. The public configuration uses 34 layers, hidden size
2560, 20 attention heads, 64-dimensional latents, and blocks of eight latent
positions.

The DiT attends block-causally: previously committed latent blocks form a clean
prefix, while all positions inside the new block are denoised jointly. The
official solver uses eight Euler steps with a low-noise-biased schedule:

\[
t_i = T(1-i/8)^2,\qquad i=0,\ldots,8.
\]

Classifier-free guidance combines conditional and unconditional velocity
predictions with scale 10.

## Token realization

The token decoder has a Qwen3-4B-shaped causal backbone: hidden size 2560,
36 layers, 32 query heads, and 8 key/value heads. Generated latents are expanded
according to predicted span durations and supplied through conditional layers.

The release path is span-causal: complete token spans are committed from left
to right. With the official preset, one latent span is realized per decoder
chunk. Decoder CFG uses a scale of -0.9 to balance latent reconstruction with
the decoder's autonomous language-modeling ability.

## Block generation loop

For each output block, `Cola2ForConditionalGeneration.generate()`:

1. encodes the prompt and already committed text;
2. samples Gaussian noise for the unfilled positions of an eight-latent block;
3. denoises the block with the DiT;
4. predicts a token duration for each new latent;
5. decodes and commits complete token spans;
6. stops at EOS or `max_new_tokens`; and
7. re-encodes committed spans under the fixed layout before generating the
   next block.

The last step is **latent feedback**. It prevents later blocks from conditioning
on a latent trajectory that no longer matches the discrete text actually
selected by the decoder.

## Paper and release terminology

The paper uses masked-diffusion terminology for controlled token-decoder
experiments and uses block-causal generation at multiple levels. In the public
implementation:

- **DiT** or **Full-DiT** refers to the continuous latent denoiser;
- **span-causal decoder** refers to the released discrete realization path; and
- **block-causal** describes the latent visibility pattern.

These terms should not be interpreted as a promise that every experimental
token-decoder factorization is exposed by the released checkpoint. Public class
names and configuration fields are the source of truth for the software API.

## Transformers interface

`Cola2ForConditionalGeneration` subclasses `PreTrainedModel` and contains
`Cola2CompressionVAE` and `Cola2DiT`. Importing `cola2` registers:

- `Cola2Config` with `AutoConfig`;
- the model with `AutoModel`; and
- the same conditional-generation model with `AutoModelForCausalLM`.

The main public operations are:

- `encode()` for dynamic semantic compression;
- `forward()` for reconstruction logits;
- `generate()` for latent-and-token continuation; and
- `from_pretrained()` / `save_pretrained()` for Transformers checkpoints.

Released checkpoint directories also contain `auto_map` and remote-code modules,
so users can load them with `trust_remote_code=True` without installing the
package separately. Review remote code before enabling that option in a
security-sensitive environment.

## Scope

This repository documents and tests inference behavior. It does not include
the three-stage training implementation or enough artifacts to reproduce
training. See the [paper overview](paper.md) for the training summary and the
[configuration reference](configuration.md) for exact public defaults.
