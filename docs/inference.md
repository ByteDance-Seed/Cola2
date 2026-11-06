# Inference

[中文](inference_zh.md) · [Documentation](README.md)

## Requirements

Checkpoint inference requires the model snapshot hosted at
[ByteDance-Seed/Cola2](https://huggingface.co/ByteDance-Seed/Cola2). CUDA with
bfloat16 is recommended. CPU execution is useful for the tiny test
configuration but is not a practical target for the released checkpoint.

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "/path/to/cola2"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda").eval()
```

`trust_remote_code=True` executes Python shipped with the checkpoint. Review
that code first when loading an untrusted or mirrored snapshot. Installing this
repository registers the same classes locally.

## Official preset

The checkpoint stores the paper setting in `generation_config.json`:

```json
{
  "do_sample": false,
  "max_new_tokens": 32,
  "dit_num_steps": 8,
  "dit_timestep_schedule": "low_noise",
  "dit_guidance_scale": 10.0,
  "decoder_guidance_scale": -0.9,
  "span_causal_chunk_size": 1,
  "dynamic_duration": true,
  "latent_feedback": true
}
```

The low-noise schedule uses \(t_i=T(1-i/8)^2\), placing more Euler nodes near
zero noise. Calling `model.generate(**inputs)` uses the complete checkpoint
preset.

## Deterministic comparisons

`do_sample=false` controls token selection only. Each latent block still starts
from Gaussian noise, so calls are stochastic unless a generator is supplied.

```python
prompt = "Question: What is the capital of France? Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generator = torch.Generator(device=model.device).manual_seed(42)

with torch.inference_mode():
    output = model.generate(**inputs, generator=generator)
```

Create a new generator with the same seed before each call when exact replay is
required. Reusing one generator advances its state and intentionally produces a
different latent sample.

Exact equality across different devices, PyTorch versions, or kernels is not
guaranteed. The benchmark derives a stable seed per record so resume and
sharding do not change an already defined public run.

## Generation flow

For each block, Cola2:

1. encodes the prompt and committed context;
2. denoises eight latent positions with the block-causal DiT;
3. predicts a dynamic token duration for every latent;
4. decodes one complete latent span per decoder chunk;
5. commits the selected tokens; and
6. re-encodes the completed spans as latent feedback.

Decoder CFG -0.9 combines conditional and unconditional decoder logits. The
negative value preserves useful latent semantics while allowing the causal
decoder to supply token-level information not represented by the compressed
latent.

## Overrides

Generation fields can be supplied directly or through a copied
`Cola2GenerationConfig`:

```python
output = model.generate(
    **inputs,
    max_new_tokens=48,
    dit_num_steps=8,
    dit_timestep_schedule="low_noise",
    dit_guidance_scale=10.0,
    decoder_guidance_scale=-0.9,
    span_causal_chunk_size=1,
    dynamic_duration=True,
    latent_feedback=True,
    repetition_penalty=1.0,
    do_sample=False,
    generator=torch.Generator(device=model.device).manual_seed(42),
)
```

Supported DiT schedules are `uniform`, `high_noise`, `low_noise`, and `power`.
The `power` schedule also requires a positive `dit_timestep_power`.
`temperature`, `top_k`, and `top_p` affect token decoding only when
`do_sample=True`.

The release supports one return sequence and no beam search. Beam search is not
defined for the block-parallel span-causal generation path.

## Command line

Install the package and run:

```bash
cola2-chat \
  --model /path/to/cola2 \
  --device cuda \
  --dtype bfloat16 \
  --seed 42
```

The equivalent repository helper is:

```bash
COLA2_MODEL_PATH=/path/to/cola2 bash scripts/run_chat.sh
```

Useful options include `--prompt` for one-shot generation,
`--max-new-tokens`, `--max-context-tokens`, `--dit-steps`,
`--dit-timestep-schedule`, `--dit-cfg`, `--decoder-cfg`,
`--decoder-chunk-spans`, and `--repetition-penalty`.

The interactive wrapper formats local history as `User:` / `Assistant:` text.
This is only a convenience interface: Cola2 is a base model and has not been
instruction-tuned as a chat assistant.

## Troubleshooting

- **Missing model:** set `--model` or `COLA2_MODEL_PATH`.
- **Out of memory:** shorten the prompt or output, close other GPU processes,
  or choose a supported lower-precision dtype. Do not silently change model
  architecture fields.
- **Different output with the same prompt:** pass a seeded generator and check
  device and package versions.
- **Unused generation argument:** use the canonical names in
  [Configuration](configuration.md); unknown fields fail instead of being
  ignored.
- **Remote-code error:** install this repository or ensure the snapshot
  includes the advertised remote-code files.
