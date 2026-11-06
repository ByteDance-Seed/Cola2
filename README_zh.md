# Cola2

Cola2 是一个让语义压缩与离散文本生成协同适配的连续 latent 语言模型。模型由
动态压缩 VAE、block-causal Diffusion Transformer（DiT）、span-causal token
decoder 和 latent feedback 组成。

[English](README.md) ·
[项目主页](https://hongcanguo.github.io/Cola2) ·
论文：**即将发布** ·
模型与 benchmark 文件：**即将发布**

## 项目简介

配套论文题为 **“Cola2: Co-Adaptive Hierarchical Language Modeling with
Continuous Semantic Compression and Balanced Discrete Generation”**。
Cola2 将文本动态划分为语义 span，用紧凑的连续 latent 表示每个 span，通过 DiT
生成 latent block，再将 latent 实现为离散 token。

论文所报告的 checkpoint 使用 346.03B source tokens 训练，在八项评测上的平均
成绩为 60.10。这些数字是论文报告结果，并不表示每次本地运行都会得到完全相同的
数值。

本仓库提供：

- 仅用于推理、兼容 Hugging Face Transformers 的实现；
- 配置、命令行工具、示例和 CPU 单元测试；
- 支持断点续跑的八任务 benchmark 与 scorer；
- 论文结果的 Markdown 和机器可读副本。

本仓库不包含训练代码、模型权重、tokenizer 或 benchmark JSONL。Checkpoint 和
归一化 benchmark 文件将在同一个 Hugging Face 仓库中发布。

## 安装

进入源码目录后执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

项目要求 Python 3.10+、PyTorch 2.2+、Transformers 4.51–4.x 和
safetensors 0.4+。Checkpoint 推理建议使用支持 bfloat16 的 CUDA 环境。开发与
测试依赖可通过 `python -m pip install -e ".[test]"` 安装。
Wheel 仅包含推理包；benchmark 模块与脚本需要从源码 checkout 中运行。

## 模型文件

Hugging Face 发布地址**即将提供**。下载 Transformers snapshot 后，将
`COLA2_MODEL_PATH` 指向本地目录：

```bash
export COLA2_MODEL_PATH=/path/to/cola2
```

模型目录应包含 safetensors 权重分片、tokenizer、`config.json`、
`generation_config.json`、remote-code 模块和 `SHA256SUMS`。使用前请根据
`SHA256SUMS` 校验下载文件。

## 快速调用

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

不传覆盖参数时，`generate()` 使用发布 preset：8 步 low-noise-biased Euler
DiT、DiT CFG 10、decoder CFG -0.9、每个 decoder chunk 一个 latent span、
dynamic duration、latent feedback、greedy token selection 和
`max_new_tokens=32`。Greedy token selection 不代表整个模型是确定性的，因为
latent generation 从高斯噪声开始；比较输出时应传入固定 seed 的
`torch.Generator`。

## 命令行

```bash
COLA2_MODEL_PATH=/path/to/cola2 bash scripts/run_chat.sh
```

输入 `/reset` 清空本地历史，输入 `/quit` 退出。Cola2 是 base model，不是经过
instruction tuning 的聊天助手。

## 评测

归一化 benchmark JSONL 将随 Hugging Face 模型一起发布。下载后放入
`eval_data/`，或设置 `GENERATE_TASK_DATA_DIR=/path/to/eval_data`。发布前数据
链接明确保留为**即将发布**。

运行每项任务 200 条样本的 smoke benchmark：

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export GENERATE_TASK_DATA_DIR=/path/to/eval_data
bash scripts/run_benchmark_quick.sh --models cola2
```

运行完整固定评测矩阵：

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export LLAMA3_MODEL_PATH=meta-llama/Meta-Llama-3-8B
export QWEN25_MODEL_PATH=Qwen/Qwen2.5-7B
export QWEN3_MODEL_PATH=Qwen/Qwen3-8B-Base
export LLADA_MODEL_PATH=GSAI-ML/LLaDA-8B-Base
bash scripts/run_benchmark_all.sh --max-samples 0
```

论文中每项 Cola2 成绩是三次 latent-noise sampled evaluation 的均值。可使用
`scripts/run_benchmark_paper.sh` 执行对应的三 seed 公开协议。论文原始 seed
尚未公布，因此本地结果与[参考结果](eval_output/full_test_results.md)之间可能
存在小幅数值差异。

## 文档

- [文档索引](docs/README_zh.md)
- [论文概览](docs/paper_zh.md)
- [架构](docs/architecture_zh.md)
- [推理](docs/inference_zh.md)
- [配置](docs/configuration_zh.md)
- [评测](docs/benchmark_zh.md)
- [复现协议](docs/reproduction_zh.md)
- [模型卡](docs/model_card_zh.md)
- [贡献指南](CONTRIBUTING.md)
- [安全政策](SECURITY.md)

## 引用

引用方式见 [CITATION.md](CITATION.md) 和 `CITATION.cff`。正式论文发表信息
将在可用后补充。

## 许可证

仓库代码采用 [Apache License 2.0](LICENSE)。模型权重、tokenizer、
benchmark 数据集和其他第三方素材仍分别遵循其适用条款。归属信息见
[NOTICE](NOTICE)。
