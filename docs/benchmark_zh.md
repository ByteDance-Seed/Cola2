# Benchmark

[English](benchmark.md) · [文档索引](README_zh.md) ·
[复现协议](reproduction_zh.md)

公开 benchmark 固定论文八任务对比使用的数据归一化、prompt template、生成长度、
输出 schema 和 scorer。它面向可审计复现，不是通用评测框架。

## 外部数据

归一化 source JSONL 不提交到本仓库，与 Cola2 checkpoint 一起托管在
[ByteDance-Seed/Cola2](https://huggingface.co/ByteDance-Seed/Cola2)。

下载后，将 8 个文件放入 `eval_data/`，或设置：

```bash
export GENERATE_TASK_DATA_DIR=/path/to/eval_data
```

文件必须与 [eval_data/manifest.json](../eval_data/manifest.json) 匹配。准备阶段
会拒绝不完整数据，并记录 source 和 prepared SHA-256。不要重排或去重：
NarrativeQA 有意保留重复 source ID，准备代码会通过输入位置进行区分。

## 任务

| 任务 | Source 行数 | 评测重点 |
|---|---:|---|
| RACE | 4,887 | 阅读理解 |
| NarrativeQA | 1,000 | 抽取式/开放回答 |
| HellaSwag | 10,042 | 常识补全 |
| Story Cloze | 468 | 故事结尾 |
| LAMBADA | 5,153 | 末词补全 |
| ARC-Easy | 2,375 | 科学多项选择 |
| TruthfulQA | 817 | 真实性多项选择 |
| MMLU | 14,042 | 多领域知识 |

除 LAMBADA 外，所有任务使用固定 few-shot prompt。Prompt 构造位于
`benchmarks/prompt_template.py`，答案提取与 scoring 位于
`benchmarks/task_scoring.py`。

报告 aggregate 是八项任务准确率的等权算术平均，不按样本数加权。

## 模型与 decoding 契约

论文表格包含 Meta-Llama-3-8B、Qwen2.5-7B、Qwen3-8B-Base、
LLaDA-8B-Base 和 Cola2。所有列共同使用：

- 相同的归一化行与固定 prompt；
- greedy token selection；
- `max_new_tokens=32`；
- 公开 runner 中 2048 的最大 prompt length；
- 相同 prediction schema 与 task scorer。

Cola2 还固定使用 low-noise 8-step DiT、DiT CFG 10、decoder CFG -0.9、
每个 decoder chunk 一个 latent span、dynamic duration 和 latent feedback。
虽然 token decoding 为 greedy，DiT 仍从采样的高斯噪声开始，因此 base seed
属于公开 run configuration。

Meta Llama checkpoint 受访问控制，需要接受许可证并登录 Hugging Face。
硬件和显存需求取决于模型；runner 默认使用 CUDA 和 bfloat16，不执行自动
offloading。

## 命令

每项所选任务 200 行的 Cola2 smoke run：

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export GENERATE_TASK_DATA_DIR=/path/to/eval_data

bash scripts/run_benchmark_quick.sh \
  --models cola2 \
  --tasks "lambada arc_easy"
```

通过 `COLA2_QUICK_SAMPLES` 修改 quick 数量：

```bash
COLA2_QUICK_SAMPLES=100 bash scripts/run_benchmark_quick.sh --models cola2
```

完整五模型 grid：

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export LLAMA3_MODEL_PATH=meta-llama/Meta-Llama-3-8B
export QWEN25_MODEL_PATH=Qwen/Qwen2.5-7B
export QWEN3_MODEL_PATH=Qwen/Qwen3-8B-Base
export LLADA_MODEL_PATH=GSAI-ML/LLaDA-8B-Base

bash scripts/run_benchmark_all.sh --max-samples 0
```

论文式三 seed Cola2 运行：

```bash
COLA2_PAPER_SEEDS="42 43 44" \
  bash scripts/run_benchmark_paper.sh --max-samples 0
```

论文没有公开原始随机 seed。以上公开 seed 定义了使用同一三次评测均值规则的可
重复协议，但不保证与历史表格数值完全相同。

## Prepared data 与产物

默认 `COLA2_BENCHMARK_ROOT=benchmark_artifacts` 时，单次运行写入：

```text
benchmark_artifacts/
├── data/
│   ├── manifest.json
│   └── {task}.jsonl
├── outputs/{model}/{task}/
│   ├── run_config.json
│   ├── predictions.jsonl
│   ├── scored.jsonl
│   ├── metrics.json
│   └── complete.json
└── summary/
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

论文脚本将每个 seed 保存在独立 run 目录，并在 `paper_summary/` 下写入含
mean/standard deviation 的 aggregate summary。

`run_config.json` 记录 model source、dtype、device、decoding 设置与 base seed。
Completion marker 包含 prediction、score、metrics 和 configuration hash，避免
静默复用过期产物。

## Resume 与 sharding

Inference 追加 prediction，并可恢复尾部不完整的 JSONL record。Resume 默认
开启。如果包括 `sample_seed` 在内的任何推理设置改变，runner 会要求
`--force`，不会混用产物。

独立 job 可拆分 grid：

```bash
python -m benchmarks.orchestrate \
  --model-shard-index 0 --model-shard-count 2 \
  --task-shard-index 0 --task-shard-count 4
```

Cola2 的确定性 seed 由 base seed、task 和 canonical sample ID 派生，因此 task
顺序、resume 和 sharding 不会改变单条 sample。

## 外部 Cola2 adapter

`COLA2_BENCHMARK_COMMAND` 可替换已安装的 Cola2 backend。Command template
必须消费文档定义的 placeholder，包括 `{input}`、`{output}`、模型/设置字段和
`{sample_seed}`、`{sample_seed_scheme}`。Adapter 必须按
`benchmarks.sample_rng.derive_sample_seed(base_seed, task, sample_id)`
实现的同一 `sha256-v1` 算法，为每条 record 派生 generator seed。完整契约见
`python -m benchmarks.run_inference --help`。

## 结果解释

[eval_output/full_test_results.md](../eval_output/full_test_results.md) 保存论文
报告值。原始 Cola2 seed 未公开，并且 device 或依赖版本可能影响数值，因此本地
结果可能不同。报告结果时应比较协议与 per-task trend，并保留生成的
`run_config.json`。
