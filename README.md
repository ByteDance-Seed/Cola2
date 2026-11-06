# Cola2

Cola2 is a continuous-latent language model that co-adapts semantic
compression and discrete text generation. It combines a dynamic-compression
VAE, a block-causal Diffusion Transformer (DiT), and a span-causal token
decoder with latent feedback.

[中文](README_zh.md) ·
[Paper](https://arxiv.org/abs/2610.31305) ·
[Project page](https://hongcanguo.github.io/Cola2/) ·
[Model and benchmark files](https://huggingface.co/ByteDance-Seed/Cola2)

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
benchmark JSONL files. The checkpoint and normalized benchmark files are
hosted in the same
[Hugging Face repository](https://huggingface.co/ByteDance-Seed/Cola2).

## Project overview

[![Animated preview of the Cola2 method and results.](assets/videos/cola2-overview-preview.gif)](assets/videos/cola2-v2-motion-bgm-1080p.mp4)

Click the preview to watch the full video ·
[Open the interactive project page](https://hongcanguo.github.io/Cola2/)

## Method

![Cola2 method overview: semantic compression, balanced discrete decoding, detached latent-prior training, and block-wise inference.](https://hongcanguo.github.io/Cola2/static/images/cola2-workflow.svg)

Stage I jointly trains the dynamic-compression encoder and the token decoder.
Stage II continues adapting that VAE with token-side supervision while an
approximately 4B DiT learns the detached latent distribution through Flow
Matching. During inference, Cola2 repeatedly proposes a latent block, decodes
tokens, and re-encodes the committed text under the fixed span layout so the
continuous and discrete histories stay aligned.

## Controlled scaling

The controlled study compares AR, LLaDA, Cola DLM, uncompressed Cola-MDM, and
compressed Cola2-MDM at matched model size, data, and evaluation settings.
Cola-MDM has the strongest aggregate scaling; Cola2-MDM converges more slowly
but continues closing its gap to LLaDA after roughly 150B tokens and remains
unsaturated at about 210B tokens.

![Controlled scaling trends for AR, LLaDA, Cola DLM, Cola-MDM, and Cola2-MDM.](assets/figures/scaling-scatter-trend.svg)

The optimal decoder classifier-free-guidance weight remains near `-0.9` for
most of training and gradually moves toward zero.

![Optimal decoder CFG weight over the controlled scaling run.](assets/figures/best-decoder-cfg-trend.svg)

## Post-pretraining results

The resource-constrained checkpoint uses 346.03B source tokens. Its latent DiT
is still unconverged, yet Cola2 reaches a 60.10 eight-task average and exceeds
Llama-3-8B on HellaSwag and StoryCloze with more than 43× fewer reported
pretraining tokens.

| Task | Llama-3-8B | Qwen2.5-7B | Qwen3-8B-Base | LLaDA-8B-Base | **Cola2** |
|---|---:|---:|---:|---:|---:|
| Pretraining tokens | 15T+ | 18T | ~36T | 2.3T | **346B** |
| RACE | 75.77 | 89.95 | 90.65 | 87.40 | **69.91** |
| NarrativeQA | 42.70 | 32.90 | 37.30 | 36.00 | **32.70** |
| HellaSwag | 39.21 | 75.67 | 68.13 | 69.65 | **41.75** |
| StoryCloze | 79.49 | 95.09 | 97.44 | 95.94 | **80.42** |
| LAMBADA | 75.70 | 71.82 | 71.20 | 67.90 | **63.00** |
| ARC-Easy | 90.86 | 96.00 | 97.35 | 93.85 | **87.62** |
| TruthfulQA | 63.04 | 69.03 | 72.58 | 64.87 | **54.10** |
| MMLU | 63.19 | 70.16 | 72.31 | 61.27 | **51.32** |
| **8-task average** | **66.25** | **75.08** | **75.87** | **72.11** | **60.10** |

Values are accuracies (%) except for the token row. Models use different
training recipes, tokenizers, and token-counting conventions, so this table
contextualizes capability rather than isolating the generation paradigm.
Each Cola2 task value is the mean of three latent-noise sampled evaluations;
see the [reference results](eval_output/full_test_results.md) and
[reproduction protocol](docs/reproduction.md).

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

Download the Transformers snapshot from
[ByteDance-Seed/Cola2](https://huggingface.co/ByteDance-Seed/Cola2), then point
`COLA2_MODEL_PATH` to the local directory:

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

Normalized benchmark JSONL files are distributed with the Hugging Face
model release. Download them from
[ByteDance-Seed/Cola2](https://huggingface.co/ByteDance-Seed/Cola2) to
`eval_data/`, or set `GENERATE_TASK_DATA_DIR=/path/to/eval_data`.

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

Please cite the [Cola2 arXiv paper](https://arxiv.org/abs/2610.31305). BibTeX
and machine-readable metadata are provided in [CITATION.md](CITATION.md) and
`CITATION.cff`.

## License

The repository code is licensed under the
[Apache License 2.0](LICENSE). Model weights, tokenizer assets, benchmark
datasets, and other third-party materials remain subject to their respective
terms. See [NOTICE](NOTICE) for attribution details.
