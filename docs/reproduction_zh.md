# 复现协议

[English](reproduction.md) · [文档索引](README_zh.md) ·
[Benchmark 参考](benchmark_zh.md)

本指南区分三种目标：

1. 快速验证公开 pipeline；
2. 完成一次确定性的完整 benchmark；
3. 按论文协议汇总三次独立采样的 Cola2 运行。

模型 checkpoint 与归一化 source JSONL 是外部产物，Hugging Face 链接目前为
**即将发布**。

## 1. 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

export COLA2_MODEL_PATH=/path/to/cola2
export GENERATE_TASK_DATA_DIR=/path/to/eval_data
```

请校验 checkpoint 的 `SHA256SUMS`，并确认 8 个 source 文件与
`eval_data/manifest.json` 匹配。Runner 会检查行数，并在 prepared-data
manifest 中记录 hash。

公开默认环境为 CUDA + bfloat16。发布数值时应记录 GPU、driver、CUDA、
PyTorch、Transformers 和 checkpoint revision。

## 2. Smoke test

运行小型 Cola2-only 子集：

```bash
COLA2_QUICK_SAMPLES=20 bash scripts/run_benchmark_quick.sh \
  --models cola2 \
  --tasks "lambada arc_easy"
```

该命令验证数据准备、模型加载、prediction、scoring 与 summary generation，
不能与完整论文表直接比较。

## 3. 单次确定性完整运行

```bash
export COLA2_BENCHMARK_SEED=42
bash scripts/run_benchmark_all.sh \
  --models cola2 \
  --max-samples 0
```

Runner 为每条 record 调用
`benchmarks.sample_rng.derive_sample_seed(base_seed, task, sample_id)`。
`sha256-v1` 会对 UTF-8 payload
`"sha256-v1\n{base_seed}\n{task}\n{canonical_sample_id}"` 计算 SHA-256，
将 digest 前 8 个字节解释为无符号大端整数，再对 `2**63 - 1` 取模。

因此单条输出不受 task 顺序、resume 和 model/task sharding 影响。改变 base seed
会定义一个新 run，并使旧 run configuration 失效。

## 4. 论文式三次均值

论文将每项 Cola2 成绩报告为三次独立 latent noise 评测的均值，token selection
仍然使用 greedy。

```bash
COLA2_PAPER_SEEDS="42 43 44" \
  bash scripts/run_benchmark_paper.sh \
  --max-samples 0
```

脚本为三个 seed 分别保存产物，并写入：

```text
benchmark_artifacts/
├── paper_runs/seed-42/
├── paper_runs/seed-43/
├── paper_runs/seed-44/
└── paper_summary/
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

`paper_summary` 包含每项任务的各 run 值、算术平均与总体标准差。最终八任务结果
是八个 aggregate task mean 的等权平均。

论文原始 seed 不可用。42、43、44 定义公开可复现协议，但不保证与历史输出逐
bit 相同，也不保证恰好得到论文报告的 60.10。

## 5. Baseline grid

运行完整对比前配置全部模型来源：

```bash
export COLA2_MODEL_PATH=/path/to/cola2
export LLAMA3_MODEL_PATH=meta-llama/Meta-Llama-3-8B
export QWEN25_MODEL_PATH=Qwen/Qwen2.5-7B
export QWEN3_MODEL_PATH=Qwen/Qwen3-8B-Base
export LLADA_MODEL_PATH=GSAI-ML/LLaDA-8B-Base

bash scripts/run_benchmark_all.sh --max-samples 0
```

Meta Llama 需要受控访问。论文 baseline 列是报告值；独立重跑时应固定 dependency
revision 和 model snapshot。

## 6. Resume、force 与 shard

- Resume 默认开启，会跳过有效 completion marker。
- 模型 snapshot、seed 或推理设置改变后，用 `--force` 替换产物。
- `--no-resume` 会在输出已经存在时失败。
- `--model-shard-index/count` 与 `--task-shard-index/count` 可把独立 cell
  分配给多个 job。
- `--dry-run` 可在不加载模型的情况下检查待运行 cell。

不要合并 `run_config.json` 不同的 prediction 文件。

## 7. 独立校验

Pipeline 会校验：

- source 与 prepared sample count；
- duplicate-source-ID disambiguation 后 benchmark ID 的唯一性；
- prompt、ground truth、choices 和 task/model label；
- scored record 数量与 accuracy 计算；
- prediction、scored output、metrics 与 run configuration 的 hash。

下游分析以机器可读输出为准：

- 单次运行使用 `summary.json`；
- 三次 aggregate 使用 `paper_summary/summary.json`；
- 论文报告表使用
  [eval_output/full_test_results.json](../eval_output/full_test_results.json)。

## 8. 复现边界

公开仓库复现推理 preset、prompt/scoring 契约和三次 aggregate 规则，不复现训练，
也无法保证历史随机采样。若本地数值不同，应同时发布生成的 run configuration
与环境信息，不要覆盖仓库中的论文参考表。
