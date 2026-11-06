# Configuration

[中文](configuration_zh.md) · [Documentation](README.md)

Cola2 uses two Transformers configuration objects:

- `Cola2Config` / `config.json` defines tensor shapes and architecture.
- `Cola2GenerationConfig` / `generation_config.json` defines inference
  behavior and can be overridden per call.

## Model configuration

Architecture fields must match the checkpoint. Changing a shape-affecting
field does not resize or convert released weights.

### Tokens and context

| Field | Released default |
|---|---:|
| `vocab_size` | 151936 |
| `pad_token_id` / `mask_token_id` | 151936 |
| `bos_token_id` / `eos_token_id` | 151643 |
| `max_position_embeddings` | 32768 |

### Dynamic-compression VAE

| Field | Released default |
|---|---:|
| `encoder_embedding_size` | 2560 |
| `encoder_hidden_size` | 1536 |
| `encoder_intermediate_size` | 6144 |
| `encoder_num_attention_heads` | 12 |
| `encoder_num_hidden_layers` | 0 |
| `pool_num_hidden_layers` | 2 |
| `latent_encoder_num_hidden_layers` | 2 |
| `latent_decoder_num_hidden_layers` | 0 |
| `latent_dim` | 64 |
| `use_variational_encoder` | true |
| `target_compression_rate` | 0.5 |
| `min_span_length` / `max_span_length` | 1 / 8 |
| `latent_block_size` | 8 |

### Latent DiT

| Field | Released default |
|---|---:|
| `dit_hidden_size` | 2560 |
| `dit_intermediate_size` | 10240 |
| `dit_num_hidden_layers` | 34 |
| `dit_num_attention_heads` | 20 |
| `dit_head_dim` | 128 |
| `dit_rope_dim` | 96 |
| `dit_block_size` | 8 |
| `diffusion_time` | 1000.0 |

### Token decoder

| Field | Released default |
|---|---:|
| `decoder_hidden_size` | 2560 |
| `decoder_intermediate_size` | 9728 |
| `decoder_num_hidden_layers` | 36 |
| `decoder_num_attention_heads` | 32 |
| `decoder_num_key_value_heads` | 8 |
| `decoder_head_dim` | 128 |
| `condition_num_attention_heads` | 20 |
| `decoder_condition_num_hidden_layers` | 2 |
| `local_position_dim` | 128 |

The complete serialized `config.json` is authoritative. This table highlights
the fields most useful for inspection and does not replace the checkpoint
configuration.

`Cola2Config.tiny()` returns a CPU-friendly shape for tests. It exercises the
same implementation but is not compatible with released weights.

## Generation configuration

| Field | Default | Meaning |
|---|---:|---|
| `max_new_tokens` | 32 | Maximum generated token count |
| `do_sample` | false | Sample token logits instead of greedy selection |
| `temperature` | 1.0 | Token sampling temperature |
| `top_k` / `top_p` | 50 / 1.0 | Token sampling filters |
| `repetition_penalty` | 1.0 | Penalty applied before token selection |
| `dit_num_steps` | 8 | Euler solver steps per latent block |
| `dit_timestep_schedule` | `low_noise` | DiT solver-node schedule |
| `dit_timestep_power` | null | Positive exponent required by `power` |
| `dit_cfg_disable_empty_prefix` | false | Disable DiT CFG when no latent prefix exists |
| `dit_guidance_scale` | 10.0 | DiT classifier-free guidance |
| `decoder_guidance_scale` | -0.9 | Decoder classifier-free guidance |
| `span_causal_chunk_size` | 1 | Latent spans realized per decoder chunk |
| `dynamic_duration` | true | Predict token count per latent |
| `latent_feedback` | true | Re-encode committed text before the next block |

Supported schedules are `uniform`, `high_noise`, `low_noise`, and `power`.
The released low-noise schedule is equivalent to a power schedule with
exponent 2.0. Token sampling controls do not remove the latent DiT's own
Gaussian-noise stochasticity; pass `generator=` separately.

### Compatibility aliases

The loader accepts legacy evaluation-harness names and serializes canonical
fields:

| Alias | Canonical field |
|---|---|
| `dit_steps` | `dit_num_steps` |
| `dit_cfg_scale` | `dit_guidance_scale` |
| `decoder_cfg_scale` | `decoder_guidance_scale` |
| `span_causal_chunk_spans` / `chunk_spans` | `span_causal_chunk_size` |
| `latent_feedback_reencode` | `latent_feedback` |

New integrations should use canonical names.

## Checkpoint layout

A released snapshot is expected to include:

```text
cola2/
├── config.json
├── generation_config.json
├── tokenizer.json and tokenizer metadata
├── model-*.safetensors
├── model.safetensors.index.json
├── configuration_cola2.py
├── modeling_cola2.py
└── SHA256SUMS
```

The exact shard count may differ. Verify `SHA256SUMS`; do not combine files
from different checkpoint revisions.

## Environment variables

### Model sources

- `COLA2_MODEL_PATH`: local Cola2 checkpoint directory.
- `LLAMA3_MODEL_PATH`, `QWEN25_MODEL_PATH`, `QWEN3_MODEL_PATH`,
  `LLADA_MODEL_PATH`: optional baseline paths or Hub IDs.

### Benchmark

- `GENERATE_TASK_DATA_DIR`: directory containing externally downloaded
  normalized task JSONL files; defaults to `eval_data/`.
- `COLA2_BENCHMARK_ROOT`: prepared data and output root; defaults to
  `benchmark_artifacts/`.
- `COLA2_BENCHMARK_SEED`: base seed for deterministic per-record Cola2 latent
  noise; defaults to `42`.
- `COLA2_BENCHMARK_COMMAND`: optional external Cola2 adapter command.
- `PYTHON_BIN`: Python executable used by shell entry points.

See [Benchmark](benchmark.md) for CLI equivalents and artifact paths.
