# Cola2

Cola2 is a continuous-latent language model that co-adapts semantic
compression and discrete text generation. It combines a dynamic-compression
VAE, a block-causal Diffusion Transformer (DiT), and a span-causal token
decoder with latent feedback.

[中文](README_zh.md) ·
[Project page](https://hongcanguo.github.io/Cola2) ·
Paper: **Coming soon** ·
Model and benchmark files: **Coming soon**

## Overview

The accompanying paper is titled **“Cola2: Co-Adaptive Hierarchical Language
Modeling with Continuous Semantic Compression and Balanced Discrete
Generation.”** Cola2 dynamically groups text into semantic spans, represents
each span with a compact continuous latent, generates latent blocks with a
DiT, and realizes those latents as discrete tokens.

The reported checkpoint was trained on 346.03B source tokens and reaches a
60.10 average across eight evaluation tasks. These numbers are reported paper
results, not a claim that every local run will produce identical values.

This repository provides:

- an inference-only, Hugging Face Transformers-compatible implementation;
- configuration, CLI, examples, and CPU unit tests;
- a resumable eight-task benchmark and scorer; and
- machine-readable and Markdown copies of the reported paper results.

It does not contain training code, model weights, tokenizer assets, or
benchmark JSONL files. The checkpoint and normalized benchmark files will be
published in the same Hugging Face repository.

## Installation

From a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10+, PyTorch 2.2+, Transformers 4.51–4.x, and safetensors 0.4+ are
required. CUDA with bfloat16 is recommended for checkpoint inference. Install
development and test dependencies with `python -m pip install -e ".[test]"`.
The wheel contains the inference package; benchmark modules and scripts are
run from a source checkout.

## Model files

The Hugging Face release is **coming soon**. After downloading its Transformers
snapshot, point `COLA2_MODEL_PATH` to the local directory:

```bash
export COLA2_MODEL_PATH=/path/to/cola2
```

The model directory is expected to contain sharded safetensors weights,
tokenizer files, `config.json`, `generation_config.json`, remote-code modules,
and `SHA256SUMS`. Verify the downloaded files against `SHA256SUMS` before use.

## Quick start

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "/path/to/cola2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda").eval()

prompt = "Question: What is the capital of France? Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generator = torch.Generator(device=model.device).manual_seed(42)

with torch.inference_mode():
    output = model.generate(**inputs, generator=generator)

answer = tokenizer.decode(
    output[0, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print(answer)
```

Without overrides, `generate()` uses the released preset: an 8-step
low-noise-biased Euler DiT schedule, DiT CFG 10, decoder CFG -0.9, one latent
span per decoder chunk, dynamic duration, latent feedback, greedy token
selection, and `max_new_tokens=32`. Greedy token selection does not make the
model deterministic because latent generation starts from Gaussian noise; pass
a seeded `torch.Generator` when comparing outputs.

## Command line

```bash
COLA2_MODEL_PATH=/path/to/cola2 bash scripts/run_chat.sh
```

Use `/reset` to clear the local history and `/quit` to exit. Cola2 is a base
model, not an instruction-tuned chat assistant.

## Evaluation

Normalized benchmark JSONL files will be distributed with the Hugging Face
model release. Download them to `eval_data/`, or set
`GENERATE_TASK_DATA_DIR=/path/to/eval_data`. Until that release, the data URL
is intentionally left as **Coming soon**.

Run a 200-example smoke benchmark:

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export GENERATE_TASK_DATA_DIR=/path/to/eval_data
bash scripts/run_benchmark_quick.sh --models cola2
```

Run the complete fixed benchmark grid:

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export LLAMA3_MODEL_PATH=meta-llama/Meta-Llama-3-8B
export QWEN25_MODEL_PATH=Qwen/Qwen2.5-7B
export QWEN3_MODEL_PATH=Qwen/Qwen3-8B-Base
export LLADA_MODEL_PATH=GSAI-ML/LLaDA-8B-Base
bash scripts/run_benchmark_all.sh --max-samples 0
```

The paper reports each Cola2 task score as the mean of three latent-noise
sampled evaluations. Use `scripts/run_benchmark_paper.sh` for the corresponding
three-seed public protocol. The original paper seeds are not published, so
small numerical differences from the [reference results](eval_output/full_test_results.md)
are expected.

## Documentation

- [Documentation index](docs/README.md)
- [Paper overview](docs/paper.md)
- [Architecture](docs/architecture.md)
- [Inference](docs/inference.md)
- [Configuration](docs/configuration.md)
- [Benchmark](docs/benchmark.md)
- [Reproduction protocol](docs/reproduction.md)
- [Model card](docs/model_card.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Citation

See [CITATION.md](CITATION.md) and `CITATION.cff`. Formal paper publication
metadata will be added when it becomes available.

## License

The repository code is licensed under the
[Apache License 2.0](LICENSE). Model weights, tokenizer assets, benchmark
datasets, and other third-party materials remain subject to their respective
terms. See [NOTICE](NOTICE) for attribution details.
