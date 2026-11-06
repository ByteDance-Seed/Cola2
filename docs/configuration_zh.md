# 配置

[English](configuration.md) · [文档索引](README_zh.md)

Cola2 使用两个 Transformers 配置对象：

- `Cola2Config` / `config.json` 定义 tensor shape 和架构；
- `Cola2GenerationConfig` / `generation_config.json` 定义推理行为，可在
  每次调用时覆盖。

## 模型配置

架构字段必须与 checkpoint 匹配。修改影响 shape 的字段不会自动 resize 或转换
发布权重。

### Token 与上下文

| 字段 | 发布默认值 |
|---|---:|
| `vocab_size` | 151936 |
| `pad_token_id` / `mask_token_id` | 151936 |
| `bos_token_id` / `eos_token_id` | 151643 |
| `max_position_embeddings` | 32768 |

### 动态压缩 VAE

| 字段 | 发布默认值 |
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

| 字段 | 发布默认值 |
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

| 字段 | 发布默认值 |
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

完整序列化的 `config.json` 是权威来源。上表只列出最常用于检查的字段，不能替代
checkpoint 配置。

`Cola2Config.tiny()` 提供适合 CPU 测试的小型 shape。它使用相同实现路径，但与
发布权重不兼容。

## 生成配置

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `max_new_tokens` | 32 | 最大生成 token 数 |
| `do_sample` | false | 对 token logits 采样而非 greedy selection |
| `temperature` | 1.0 | Token sampling temperature |
| `top_k` / `top_p` | 50 / 1.0 | Token sampling filter |
| `repetition_penalty` | 1.0 | Token selection 前的重复惩罚 |
| `dit_num_steps` | 8 | 每个 latent block 的 Euler solver 步数 |
| `dit_timestep_schedule` | `low_noise` | DiT solver-node 调度 |
| `dit_timestep_power` | null | `power` 调度要求的正数指数 |
| `dit_cfg_disable_empty_prefix` | false | 无 latent prefix 时禁用 DiT CFG |
| `dit_guidance_scale` | 10.0 | DiT classifier-free guidance |
| `decoder_guidance_scale` | -0.9 | Decoder classifier-free guidance |
| `span_causal_chunk_size` | 1 | 每个 decoder chunk realization 的 latent span 数 |
| `dynamic_duration` | true | 预测每个 latent 的 token 数 |
| `latent_feedback` | true | 下一个 block 前重新编码已提交文本 |

支持 `uniform`、`high_noise`、`low_noise` 和 `power` schedule。发布的
low-noise schedule 等价于指数 2.0 的 power schedule。Token sampling 参数不会
消除 latent DiT 自身的高斯噪声随机性；generator 需要单独通过 `generator=`
传入。

### 兼容别名

Loader 接受旧评测 harness 字段，并序列化为 canonical field：

| 别名 | Canonical field |
|---|---|
| `dit_steps` | `dit_num_steps` |
| `dit_cfg_scale` | `dit_guidance_scale` |
| `decoder_cfg_scale` | `decoder_guidance_scale` |
| `span_causal_chunk_spans` / `chunk_spans` | `span_causal_chunk_size` |
| `latent_feedback_reencode` | `latent_feedback` |

新集成应使用 canonical name。

## Checkpoint 目录

发布 snapshot 预计包含：

```text
cola2/
├── config.json
├── generation_config.json
├── tokenizer.json 与 tokenizer metadata
├── model-*.safetensors
├── model.safetensors.index.json
├── configuration_cola2.py
├── modeling_cola2.py
└── SHA256SUMS
```

实际 shard 数可能不同。请校验 `SHA256SUMS`，不要混用不同 checkpoint revision
中的文件。

## 环境变量

### 模型来源

- `COLA2_MODEL_PATH`：本地 Cola2 checkpoint 目录。
- `LLAMA3_MODEL_PATH`、`QWEN25_MODEL_PATH`、`QWEN3_MODEL_PATH`、
  `LLADA_MODEL_PATH`：可选 baseline 路径或 Hub ID。

### Benchmark

- `GENERATE_TASK_DATA_DIR`：外部下载的归一化任务 JSONL 目录，默认为
  `eval_data/`。
- `COLA2_BENCHMARK_ROOT`：prepared data 与输出根目录，默认为
  `benchmark_artifacts/`。
- `COLA2_BENCHMARK_SEED`：Cola2 per-record latent noise 的 base seed，
  默认为 `42`。
- `COLA2_BENCHMARK_COMMAND`：可选外部 Cola2 adapter command。
- `PYTHON_BIN`：shell entry point 使用的 Python executable。

CLI 等价参数与产物路径见 [Benchmark](benchmark_zh.md)。
